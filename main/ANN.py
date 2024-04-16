import os
import cv2
import numpy as np

from sklearn.decomposition import PCA

image_folder = 'D:/ML_Project/Project_CK/data/animals/output_train'

images = []

# Duyệt qua từng thư mục trong thư mục chứa dữ liệu
for animal_folder in os.listdir(image_folder):
    # Duyệt qua từng file trong thư mục con
    for filename in os.listdir(os.path.join(image_folder, animal_folder)):
        img = cv2.imread(os.path.join(image_folder, animal_folder, filename), 0)  # Đọc ảnh dưới dạng ảnh đen trắng
        images.append(img)

X = np.array(images).reshape(len(images), -1)

# Giảm chiều dữ liệu bằng PCA
pca = PCA(n_components=225)
X_pca = pca.fit_transform(X)





