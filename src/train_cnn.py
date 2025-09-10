import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_CHANNELS = 3

script_dir = os.path.dirname(os.path.abspath(__file__)) #Lấy folder chứa file .py đang chạy
project_root = os.path.dirname(script_dir) #Lấy thư mục cha của thằng bên trên (script_dir)

split_dir = os.path.join(project_root, 'data', 'split')
train_dir = os.path.join(split_dir, 'train')
test_dir = os.path.join(split_dir, 'test')

# print(split_dir)
# print(train_dir)
# print(test_dir)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

num_classes = len(train_generator.class_indices)

model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=25,
    validation_data=test_generator,
    validation_steps=test_generator.n // test_generator.batch_size
)

# Đánh giá mô hình trên tập kiểm tra
print("\nĐánh giá hiệu suất cuối cùng trên tập kiểm tra...")
loss, accuracy = model.evaluate(test_generator, steps=test_generator.n // test_generator.batch_size)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")

save_path = os.path.join(project_root, "models", "drowsiness_detector.h5")
model.save(save_path)
print(f"Model saved in {save_path}")
