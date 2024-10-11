import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기
image = cv2.imread('sample.jpeg')  # 파일 경로에 맞게 수정
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 그레이스케일로 변환

# 2. GaussianBlur 적용하여 노이즈 제거 및 별 감지 강화
blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)

# 3. Thresholding: 밝은 별을 감지하기 위한 이진화 처리
_, thresholded_image = cv2.threshold(blurred_image, 80, 255, cv2.THRESH_BINARY)

# 4. Morphological Erosion으로 빛 번짐 제거 (작은 커널 사용)
kernel = np.ones((3, 3), np.uint8)
eroded_image = cv2.erode(thresholded_image, kernel, iterations=1)

# 5. Contour 찾기
contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 6. 별을 흑백 이미지에 빨간색 원과 번호로 표시
annotated_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)  # 원을 그리기 위해 흑백 이미지를 BGR로 변환

# 인식된 별들의 정보를 저장할 리스트
star_info = []

for idx, contour in enumerate(contours):
    # 최소 외접 원 찾기 (별을 감쌀 수 있는 최소 원)
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))  # 별의 중심 좌표
    radius = int(radius)  # 반지름을 정수로 변환

    # 면적 기반 필터링 (너무 큰 영역을 줄임)
    area = cv2.contourArea(contour)
    if area > 2000:  # 면적이 너무 크면 반지름을 줄임
        radius = int(radius * 0.7)  # 70%로 줄임

    # 원을 그리기 (빨간색)
    cv2.circle(annotated_image, center, radius, (0, 0, 255), 2)
    # 별 번호 표시
    cv2.putText(annotated_image, f'{idx + 1}', (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 별 번호, 중심 좌표, 크기 정보 저장
    star_info.append((idx + 1, center, radius))

# 별 정보 출력
print("인식된 별 정보 (번호, 위치(중심 좌표), 크기(반지름)):")
for info in star_info:
    print(f"별 {info[0]}: 위치 = {info[1]}, 크기(반지름) = {info[2]}")

# 7. 흑백 이미지에 빨간색 원과 번호 표시 후 시각화
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))  # BGR에서 RGB로 변환하여 표시
plt.title("Detected Stars with Reduced Halo Effect (Erosion + Contour)")
plt.show()
