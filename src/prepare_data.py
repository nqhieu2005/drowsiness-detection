import os
import shutil
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(os.path.abspath(__file__)) #Lấy folder chứa file .py đang chạy
project_root = os.path.dirname(script_dir) #Lấy thư mục cha của thằng bên trên (script_dir)

processed_dir = os.path.join(project_root, 'data', 'processed')
split_dir = os.path.join(project_root, 'data', 'split')
labels = ['eyeclose', 'yawn', 'awake', 'happy']

#Tạo cấu trúc thư mục đầu ra
for split_type in ['train', 'test']:
    for label in labels:
        os.makedirs(os.path.join(split_dir, split_type, label), exist_ok=True)

#Chia dữ liệu
print("Start splitting data...")
for label in labels:
    all_images = os.listdir(os.path.join(processed_dir, label))
    train_images, test_images = train_test_split(all_images, test_size=0.2, random_state=42)

    print(f"Label '{label}' has {len(all_images)} images, split into {len(train_images)} train, {len(test_images)} test")

    for img in train_images:
        src_path = os.path.join(processed_dir, label, img)
        destination_path = os.path.join(split_dir, 'train', label, img)
        shutil.copy(src_path, destination_path)
    for img in test_images:
        src_path = os.path.join(processed_dir, label, img)
        destination_path = os.path.join(split_dir, 'test', label, img)
        shutil.copy(src_path, destination_path)
print("Split done.")