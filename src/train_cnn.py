import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Cấu hình để tiết kiệm memory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Giảm log của TensorFlow
tf.config.experimental.enable_memory_growth = True

# Cấu hình GPU nếu có (nhưng với memory growth để không chiếm hết VRAM)
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detected: {len(gpus)} GPU(s) available")
    else:
        print("No GPU detected, using CPU")
        # Giới hạn số threads cho CPU để không quá tải
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError as e:
    print(f"GPU configuration error: {e}")

# Kích thước ảnh nhỏ hơn để tiết kiệm RAM
IMG_WIDTH = 160  # Giảm từ 224 xuống 160
IMG_HEIGHT = 160
IMG_CHANNELS = 3
BATCH_SIZE = 16  # Giảm batch size để tiết kiệm memory

# Lấy đường dẫn tuyệt đối của thư mục gốc
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
train_dir = os.path.join(project_root, 'data', 'train')
validation_dir = os.path.join(project_root, 'data', 'valid')
test_dir = os.path.join(project_root, 'data', 'test')

# Tạo thư mục models nếu chưa có
models_dir = os.path.join(project_root, 'models')
os.makedirs(models_dir, exist_ok=True)

# Data augmentation nhẹ để tiết kiệm memory
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode='nearest'
)

# Validation và test chỉ cần rescale
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Danh sách classes
classes = ['DangerousDriving', 'Distracted', 'Drinking', 'SafeDriving', 'SleepyDriving', 'Yawn']

# Tạo generators với batch size nhỏ
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=classes,
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=classes,
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=classes,
    shuffle=False
)

# In thông tin về dataset
print("Dataset Information:")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Test samples: {test_generator.samples}")
print(f"Classes: {train_generator.class_indices}")
print(f"Image size: {IMG_HEIGHT}x{IMG_WIDTH}")
print(f"Batch size: {BATCH_SIZE}")

# Tính class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

# Lấy số lớp
num_classes = len(train_generator.class_indices)


def create_lightweight_model():
    """Tạo model nhẹ phù hợp với RAM 8GB"""

    # Sử dụng MobileNetV2 với alpha=0.75 để giảm parameters
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS),
        alpha=0.75  # Giảm width của model
    )

    # Đóng băng base model
    base_model.trainable = False

    # Architecture đơn giản hơn
    inputs = base_model.input
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)  # Giảm từ 512 xuống 128
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    print(f"Model parameters: {model.count_params():,}")
    return model, base_model


# Tạo model
model, base_model = create_lightweight_model()

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks được tối ưu cho training nhanh
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=8,  # Giảm patience để training nhanh hơn
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(models_dir, "best_model.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
]

# Phase 1: Training với frozen base model (ít epochs hơn)
print("\n" + "=" * 50)
print("Phase 1: Training with frozen base model...")
print("=" * 50)

history1 = model.fit(
    train_generator,
    epochs=15,  # Giảm từ 25 xuống 15
    validation_data=validation_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# Clear memory sau phase 1
gc.collect()

# Phase 2: Fine-tuning nhẹ
print("\n" + "=" * 50)
print("Phase 2: Fine-tuning...")
print("=" * 50)

# Unfreeze một phần của base model
base_model.trainable = True

# Chỉ fine-tune các layer cuối
for layer in base_model.layers[:-30]:  # Freeze tất cả trừ 30 layer cuối
    layer.trainable = False

# Compile với learning rate rất thấp
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Learning rate thấp hơn
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tuning với ít epochs
fine_tune_callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-8,
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(models_dir, "best_model_finetuned.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
]

history2 = model.fit(
    train_generator,
    epochs=8,  # Giảm từ 15 xuống 8
    validation_data=validation_generator,
    class_weight=class_weight_dict,
    callbacks=fine_tune_callbacks,
    verbose=1
)

# Clear memory
gc.collect()

# Đánh giá model cuối cùng
print("\n" + "=" * 50)
print("Final Evaluation")
print("=" * 50)

# Load best model
best_model_path = os.path.join(models_dir, "best_model_finetuned.h5")
if os.path.exists(best_model_path):
    print("Loading best fine-tuned model...")
    model = tf.keras.models.load_model(best_model_path)

# Evaluate trên test set
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"Final Test Loss: {test_loss:.4f}")
print(f"Final Test Accuracy: {test_accuracy:.4f}")

# Predictions và classification report (xử lý memory-efficient)
print("\nGenerating predictions...")
test_generator.reset()

# Predict theo batch để tiết kiệm memory
predictions = []
true_labels = []

for i in range(len(test_generator)):
    batch_images, batch_labels = test_generator[i]
    batch_predictions = model.predict(batch_images, verbose=0)

    predictions.extend(np.argmax(batch_predictions, axis=1))
    true_labels.extend(np.argmax(batch_labels, axis=1))

    # Clear memory định kỳ
    if i % 10 == 0:
        gc.collect()

predictions = np.array(predictions)
true_labels = np.array(true_labels)
class_labels = list(train_generator.class_indices.keys())

print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=class_labels))

# Confusion Matrix (đơn giản hóa)
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(models_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()

# Lưu model cuối cùng
final_model_path = os.path.join(models_dir, "drowsiness_detector_final.h5")
model.save(final_model_path)
print(f"Final model saved to: {final_model_path}")


def plot_training_history_simple(history1, history2):
    """Vẽ biểu đồ training history đơn giản"""

    # Kết hợp history từ 2 phases
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']

    epochs_range = range(len(acc))
    phase1_end = len(history1.history['accuracy'])

    plt.figure(figsize=(12, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy')
    plt.axvline(x=phase1_end - 1, color='green', linestyle='--', alpha=0.7, label='Fine-tuning starts')
    plt.legend()
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'b-', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'r-', label='Validation Loss')
    plt.axvline(x=phase1_end - 1, color='green', linestyle='--', alpha=0.7, label='Fine-tuning starts')
    plt.legend()
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.show()


# Vẽ biểu đồ training history
plot_training_history_simple(history1, history2)

# Clear memory cuối cùng
gc.collect()

print("\n" + "=" * 50)
print("Training completed successfully!")
print("=" * 50)
print(f"System specs: RAM 8GB, Image size: {IMG_HEIGHT}x{IMG_WIDTH}")
print(f"Model parameters: {model.count_params():,}")
print(f"Best model: {best_model_path}")
print(f"Final model: {final_model_path}")
print("=" * 50)

# Memory usage summary
print("\nMemory optimization features enabled:")
print("✓ Reduced image size (160x160 instead of 224x224)")
print("✓ Small batch size (16)")
print("✓ Lightweight MobileNetV2 (alpha=0.75)")
print("✓ Simplified architecture")
print("✓ Fewer training epochs")
print("✓ Memory cleanup with gc.collect()")
print("✓ Batch-wise prediction to save memory")