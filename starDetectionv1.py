import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기
image = cv2.imread('Pleiades.jpg')  # 제공한 이미지 파일 경로
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 그레이스케일로 변환

# 2. GaussianBlur 적용하여 노이즈 제거 및 별 감지 강화
blurred_image = cv2.GaussianBlur(gray_image, (15, 15), 0)

# 3. Thresholding: 밝은 별을 감지하기 위한 이진화 처리
_, thresholded_image = cv2.threshold(blurred_image, 220, 255, cv2.THRESH_BINARY)

# 4. Blob Detection을 위한 파라미터 설정
params = cv2.SimpleBlobDetector_Params()

# 밝은 별을 감지할 수 있도록 밝기 필터 사용
params.filterByColor = True
params.blobColor = 255  # 밝은 별 감지

# 별의 최소 크기 설정 (작은 잡음을 제거)
params.filterByArea = True
params.minArea = 50  # 너무 작은 노이즈 제거
params.maxArea = 10000  # 너무 큰 객체 제거

# 별의 원형성을 필터링
params.filterByCircularity = True
params.minCircularity = 0.7  # 별의 원형에 가까운 객체만 필터링

# 별이 긴 형태가 아닌, 균일한 모양으로 필터링 (긴 물체나 잡음 제거)
params.filterByInertia = True
params.minInertiaRatio = 0.5

# 블롭 감지기 생성
detector = cv2.SimpleBlobDetector_create(params)

# 감지된 별(블롭)을 추출
keypoints = detector.detect(thresholded_image)

# 5. 별의 위치와 크기 인덱싱
for keypoint in keypoints:
    x, y = np.round(keypoint.pt).astype(int)  # 별의 중심 좌표
    size = keypoint.size  # 별의 크기
    cv2.circle(image, (x, y), int(size // 2), (0, 255, 0), 2)  # 원으로 별 표시
    print(f"별 위치: ({x}, {y}), 크기: {size}")

# 6. 결과 이미지와 감지된 별 시각화
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # BGR에서 RGB로 변환하여 표시
plt.title("Detected Stars with Recommended Parameters")
plt.show()
