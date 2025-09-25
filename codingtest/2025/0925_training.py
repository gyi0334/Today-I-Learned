import cv2
import os

# 파일이 실제로 위치한 곳만 경로 지정
base_path = r"C:\Users\porsche\Documents\GitHub\Today-I-Learned\models"
configFile = os.path.join(base_path, "deploy.prototxt")
modelFile = os.path.join(base_path, "res10_300x300_ssd_iter_140000_fp16.caffemodel")

net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


# 웹캠 열기 (Windows에서는 CAP_DSHOW 권장)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다.")
        break

    (h, w) = frame.shape[:2]

    # DNN 입력 준비 (300x300으로 변환)
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    # 탐지 결과 반복
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # 신뢰도 50% 이상만 표시
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")

            # 얼굴 영역 바운딩 박스 + 신뢰도 표시
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{confidence*100:.1f}%"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 출력
    cv2.imshow("Face Detection (DNN)", frame)

    # q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
