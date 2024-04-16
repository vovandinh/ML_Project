import os
import cv2

# Thư mục chứa dữ liệu ảnh
data_dir = 'D:/ML_Project/Project_CK/data/animals/train'

# Thư mục để lưu trữ ảnh đã chuyển đổi kích thước
output_dir = 'D:/ML_Project/Project_CK/data/animals/output_train'

# Kích thước mong muốn
desired_size = (150, 150)

# Duyệt qua từng thư mục trong thư mục chứa dữ liệu
for animal_folder in os.listdir(data_dir):
    # Tạo thư mục tương ứng trong thư mục lưu trữ ảnh đã chuyển đổi
    os.makedirs(os.path.join(output_dir, animal_folder), exist_ok=True)

    # Duyệt qua từng file trong thư mục con
    for filename in os.listdir(os.path.join(data_dir, animal_folder)):
        # Đọc ảnh từ file
        img = cv2.imread(os.path.join(data_dir, animal_folder, filename))

        # Chuyển đổi kích thước ảnh
        resized_img = cv2.resize(img, desired_size)

        # Lưu ảnh đã chuyển đổi vào thư mục tương ứng
        cv2.imwrite(os.path.join(output_dir, animal_folder, filename), resized_img)