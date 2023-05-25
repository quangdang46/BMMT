'''
-sử dụng thuật toán SIFT (Scale-Invariant Feature Transform) để tìm kiếm điểm đặc trưng và so khớp giữa một ảnh đầu vào và các ảnh trong một thư mục chứa cơ sở dữ liệu các ảnh.
-đọc ảnh đầu vào bằng hàm cv2.imread() và lưu vào biến source_image. Sau đó, ta duyệt qua từng file trong thư mục chứa cơ sở dữ liệu bằng hàm os.listdir(). Đối với mỗi file, ta đọc ảnh bằng cv2.imread() và lưu vào biến target_image.
-sử dụng hàm cv2.SIFT_create() để tạo một đối tượng SIFT và áp dụng thuật toán SIFT để tìm kiếm các điểm đặc trưng và tính toán mô tả của chúng trong cả ảnh đầu vào và ảnh trong cơ sở dữ liệu. Ta sử dụng hàm cv2.FlannBasedMatcher() để so khớp các điểm đặc trưng giữa hai ảnh và áp dụng bộ lọc ratio để lọc bỏ các khớp không đáng tin cậy.
-có danh sách các điểm khớp giữa hai ảnh, ta tính số lượng điểm khớp hợp lệ và tính điểm số của sự khớp bằng cách chia số lượng điểm khớp cho tổng số lượng điểm đặc trưng trong ảnh nhỏ hơn (vì số điểm đặc trưng của hai ảnh có thể khác nhau). Nếu điểm số mới tính được lớn hơn điểm số tốt nhất hiện tại, ta cập nhật điểm số và lưu lại tên file ảnh và ảnh đó.
-sử dụng hàm cv2.drawMatches() để vẽ các điểm đặc trưng khớp và lưu lại kết quả vào biến result. Ta hiển thị kết quả trên cửa sổ mới và in ra tên file ảnh và điểm số tốt nhất.
'''

import cv2
import os

source_image = cv2.imread("test.BMP")
score = 0
file_name = None
image = None
kp1, kp2, mp = None, None, None
db_name = "db"

for file in os.listdir(db_name):
    target_image = cv2.imread(os.path.join(db_name, file))
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(source_image, None)
    kp2, des2 = sift.detectAndCompute(target_image, None)
    matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), dict()).knnMatch(des1, des2, k=2)
    mp = []
    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            mp.append(p)

    keypoints = min(len(kp1), len(kp2))
    if keypoints > 0:
        current_score = len(mp) / keypoints * 100
        if current_score > score:
            score = current_score
            file_name = file
            image = target_image

result = cv2.drawMatches(source_image, kp1, image, kp2, mp, None)
result = cv2.resize(result, None, fx=2.5, fy=2.5)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"The best match: {file_name}")
print(f"The score: {score}")

