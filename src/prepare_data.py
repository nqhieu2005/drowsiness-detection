import os
import shutil


def organize_dataset(base_path):
    """
    Tổ chức lại dữ liệu từ annotations file sang cấu trúc thư mục
    """
    for split_name in ['train', 'test', 'valid']:
        split_path = os.path.join(base_path, split_name)
        if not os.path.exists(split_path):
            print(f"Skipping {split_name} because the directory does not exist.")
            continue

        annotation_path = os.path.join(split_path, '_annotations.txt')
        classes_path = os.path.join(split_path, '_classes.txt')

        if not os.path.exists(annotation_path) or not os.path.exists(classes_path):
            print(f"Skipping {split_name} due to missing annotation or classes file.")
            continue

        # Read class names
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        # Sửa lỗi: Thêm lại lệnh tạo thư mục con cho mỗi lớp
        for cls in classes:
            os.makedirs(os.path.join(split_path, cls), exist_ok=True)

        # Di chuyển các tệp ảnh vào các thư mục con tương ứng
        with open(annotation_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                image_name = parts[0]

                try:
                    last_part = parts[-1]
                    class_index = int(last_part.split(',')[-1])
                except (ValueError, IndexError):
                    continue

                class_name = classes[class_index]

                src_image_path = os.path.join(split_path, image_name)
                dst_image_path = os.path.join(split_path, class_name, image_name)

                if os.path.exists(src_image_path):
                    shutil.move(src_image_path, dst_image_path)
                else:
                    print(f"Warning: Image {src_image_path} not found.")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data')

    organize_dataset(data_path)
    print("Dataset organization complete.")