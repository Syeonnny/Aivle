import cv2
import numpy as np
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO("C:/Users/User/Desktop/project4/custom_model_10.pt")


# opencv에서 사용하려는 카메라
cap = cv2.VideoCapture(0)

# Haar Cascades, Viola–Jones object detection framework
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 카메라 동작 확인용
if not cap.isOpened():
    print('웹캠 실행 불가')
    exit()

# 매 프레임마다 동작시킬 것이므로 무한 반복문
while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 로드 불가')
        break

    frame = frame.astype(np.uint8)
    frame = cv2.flip(frame, 1)  # 좌우 반전

    # 흑백으로 변환하여 얼굴 탐지
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame)

    # 탐지된 얼굴에 대해 반복
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # YOLO 모델에 맞게 RGB로 변환

        # 얼굴 부분을 YOLO 모델로 예측
        results = model.predict(source=face_rgb, save=False, save_txt=False)  # 이미지 단일 예측

        # 예측 결과에 따른 사각형 및 텍스트 설정
        for r in results:
            if r.probs.top1 == 0:  # 예측 클래스가 'My Face'일 때
                color = (0, 255, 0)  # 초록색
                prob = float(r.probs.top1conf) * 100
                label_text = f'My Face: {prob:.2f}%'
            else:  # 예측 클래스가 'Other Face'일 때
                color = (0, 0, 255)  # 빨간색
                prob = float(r.probs.top1conf) * 100
                label_text = f'Other Face: {prob:.2f}%'

        # 탐지된 얼굴에 사각형과 텍스트 추가
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color, 2)

    # 프레임 표시
    cv2.imshow('Face Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
