import os
import cv2

#########################################################
## 이미지가 있는 폴더와 Annotation 파일을 저장할 폴더 경로 설정
img_folder = "C:/Users/User/Desktop/project4/seo/images"
output_folder = "C:/Users/User/Desktop/project4/seo/labels"
#########################################################

def load_face_detector():
    #####################################################################
    ## 아래 링크 두 곳에서 파일을 다운로드 받고, 현재 py 파일의 경로에 배치하세요
    ## https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
    ## https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt
    # DNN 얼굴 검출 모델 로드
    model_path = "C:/Users/User/Desktop/project4/res10_300x300_ssd_iter_140000.caffemodel"
    config_path = "C:/Users/User/Desktop/project4/deploy.prototxt"
    #####################################################################
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)
    return net

def yolo_format(x, y, w, h, img_width, img_height):
    # YOLO 형식 변환 (클래스, x_center, y_center, width, height)
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return f"0 {x_center} {y_center} {width} {height}"

## conf_threshold의 값을 변경하여서 annotate 하는 기준을 높일 수 있다!
def annotate_faces(img_folder, output_folder, face_net, conf_threshold=0.5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_images = len(os.listdir(img_folder))
    annotated_count = 0

    for img_name in os.listdir(img_folder):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(img_folder, img_name)
            image = cv2.imread(img_path)

            # 이미지가 제대로 로드되었는지 확인
            if image is None:
                print(f"Failed to load {img_name}")
                continue

            img_height, img_width = image.shape[:2]

            # DNN 모델로 얼굴 검출
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
            face_net.setInput(blob)
            detections = face_net.forward()

            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence >= conf_threshold:
                    box = detections[0, 0, i, 3:7] * [img_width, img_height, img_width, img_height]
                    (x, y, w, h) = box.astype("int")
                    faces.append((x, y, w - x, h - y))

            # if len(faces) == 0:
            #     print(f"No faces detected in {img_name}")
            #     continue  # 얼굴이 없으면 다음 이미지로

            annotation_content = "\n".join(
                yolo_format(x, y, w, h, img_width, img_height) for (x, y, w, h) in faces
            )

            # 주석 파일 저장
            output_path = os.path.join(output_folder, os.path.splitext(img_name)[0] + '.txt')
            try:
                with open(output_path, 'w') as f:
                    f.write(annotation_content)
                annotated_count += 1
                print(f"Annotated {img_name} and saved to {output_path}")
            except Exception as e:
                print(f"Failed to write {output_path}: {e}")

    print(f"Total images: {total_images}, Annotated files created: {annotated_count}")

# 얼굴 검출 모델 로드 및 주석 생성
face_net = load_face_detector()
annotate_faces(img_folder, output_folder, face_net)