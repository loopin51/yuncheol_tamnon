import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기
image = cv2.imread('Pleiades.jpg')  # 파일 경로에 맞게 수정
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 그레이스케일로 변환

# 2. GaussianBlur 적용하여 노이즈 제거 및 별 감지 강화
blurred_image = cv2.GaussianBlur(gray_image, (15, 15), 0)

# 3. Thresholding: 더 낮은 임계값을 설정하여 더 많은 별 감지
_, thresholded_image = cv2.threshold(blurred_image, 200, 255, cv2.THRESH_BINARY) #최솟값을 낮출수록 큰 별의 인식 영역이 퍼짐

# 4. Blob Detection을 위한 파라미터 설정
params = cv2.SimpleBlobDetector_Params()

# 밝은 별을 감지할 수 있도록 밝기 필터 사용
params.filterByColor = True
params.blobColor = 255  # 밝은 별 감지

# 별의 크기 설정 (작은 별과 큰 별 모두 감지되도록)
params.filterByArea = True
params.minArea = 30  # 작은 별도 감지할 수 있도록 값 낮춤
params.maxArea = 10000  # 큰 별도 감지될 수 있도록 값 크게 설정

# 별의 원형성을 필터링 (작은 별도 허용, 큰 별은 더 완화)
params.filterByCircularity = True
params.minCircularity = 0.7  # Circularity 완화 (큰 별의 비대칭성을 허용)

# 별이 긴 형태가 아닌, 균일한 모양으로 필터링
params.filterByInertia = True
params.minInertiaRatio = 0.5  # 큰 별의 긴 형태도 허용할 수 있도록 완화

# 블롭 감지기 생성
detector = cv2.SimpleBlobDetector_create(params)

# 감지된 별(블롭)을 추출
keypoints = detector.detect(thresholded_image)

# 5. 별의 위치와 크기 인덱싱 및 흑백 이미지에 별 표시
annotated_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)  # 원을 그리기 위해 흑백 이미지를 BGR로 변환

for idx, keypoint in enumerate(keypoints):
    x, y = np.round(keypoint.pt).astype(int)  # 별의 중심 좌표
    size = keypoint.size  # 별의 크기
    # 별 주위에 빨간색 원 그리기
    cv2.circle(annotated_image, (x, y), int(size // 2), (0, 0, 255), 2)
    # 별 번호를 이미지에 표시
    cv2.putText(annotated_image, f'{idx + 1}', (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# 6. 흑백 이미지에 빨간색 원과 번호 표시 후 시각화
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))  # BGR에서 RGB로 변환하여 표시
plt.title("Detected Stars with Red Circles and Numbers")
plt.show()
