import torch
from ultralytics import YOLO

"""
# 사전 학습된 YOLOv8 모델 불러오기 (COCO 80클래스)
model = YOLO("yolov8n.pt")

# 이미지 예측
results = model("C:/Users/porsche/Documents/GitHub/Today-I-Learned/codingtest/2025/yolo_pytorch_image_test.jpg")  # 결과는 내부적으로 torch.Tensor 사용

# 결과 확인
for r in results:
    boxes = r.boxes.xyxy   # 바운딩박스 좌표 (torch.Tensor)
    scores = r.boxes.conf  # confidence (torch.Tensor)
    classes = r.boxes.cls  # class indices (torch.Tensor)

    print("박스 좌표:", boxes)
    print("신뢰도:", scores)
    print("클래스:", classes)
"""
import cv2

# 모델 로드
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # PyTorch 모델로 추론
    results = model(frame)

    # 시각화 (plot() 함수가 OpenCV BGR 이미지를 반환)
    annotated_frame = results[0].plot()

    # 출력
    cv2.imshow("YOLOv8 with PyTorch", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
