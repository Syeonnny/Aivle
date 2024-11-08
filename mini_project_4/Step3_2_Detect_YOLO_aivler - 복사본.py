import cv2
import numpy as np
from ultralytics import YOLO

###############################################
## 모델 불러오기
model = YOLO("c:/Users/User/Desktop/project4/model_seo.pt")
###############################################

## opencv에서 사용하려는 카메라
cap = cv2.VideoCapture(0)

## 카메라 동작 확인용
if not cap.isOpened():
    print('웹캠 실행 불가')
    exit()

## 매 프레임마다 동작시킬 것이므로 무한 반복문
while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 로드 불가')
        break
    
    frame = frame.astype(np.uint8)
    ## 카메라 좌우 전환
    frame = cv2.flip(frame, 1)
    
    ## 예측값 생성 : 이미지 프레임을 입력하여 예측 수행
    results = model.predict(frame, verbose=False, stream=True, conf=0.5)

    ## 예측한 것을 뜯어봅시다
    for r in results:
        ## r_b는 매 프레임의 바운딩 박스'들'의 정보를 가지고 있음
        r_b = r.boxes
        ## r_b의 클래스 예측값이 없는게 아니라면
        if r_b.cls is not None:
            ## r_b가 클래스 예측한만큼 반복 수행
            for idx in range(len(r_b)):
                ## 점 2개의 좌표를 가져옴
                x1, y1, x2, y2 = int(r_b.xyxy[idx][0]), int(r_b.xyxy[idx][1]), int(r_b.xyxy[idx][2]), int(r_b.xyxy[idx][3])
                
                ## 클래스와 신뢰 점수에 따라 다르게 처리
                class_id = int(r_b.cls[idx])
                conf = r_b.conf[idx] * 100  # 신뢰도 점수
                if class_id == 0:  # Other Face
                    color = (0, 0, 255)
                    label_text = f'Other Face conf-score: {conf:.2f}'
                elif class_id == 1:  # My Face
                    color = (0, 255, 0)
                    label_text = f'My Face conf-score: {conf:.2f}'
                elif class_id == 2:  # Seo Face
                    color = (255, 165, 0)  # Seo를 위한 색상 추가 (주황색)
                    label_text = f'Seo Face conf-score: {conf:.2f}'
                
                ## 프레임에 꼭짓점 2개의 좌표를 이용하여 박스를 표현
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                ## 박스 위에 텍스트 추가
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            color, 2)

    ## 프레임을 확인할 수 있다
    cv2.imshow('Face_Detection', frame)
    
    ## 키보드 q 키를 누르면 반복문 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
