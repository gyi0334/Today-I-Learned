from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

video_path = r"C:\Users\porsche\Documents\GitHub\Today-I-Learned\codingtest\2025\test_video.mp4"
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*"mp4v") # 코덱을 지정함
out = cv2.VideoWriter("output.mp4", fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 추론
    results = model(frame)

    # 시각화 (바운딩 박스 그려진 프레임 반환)
    annotated_frame = results[0].plot()

    # 화면 출력
    cv2.imshow("YOLO Video", annotated_frame)

    # 결과 저장
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()