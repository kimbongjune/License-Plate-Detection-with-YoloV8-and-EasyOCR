from flask import Flask, render_template, request, redirect, url_for
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO
from ultralytics import YOLO
import cv2
import easyocr
import uuid
import os
import re

app = Flask(__name__)

# 모델 경로 설정
LICENSE_MODEL_DETECTION_DIR = './models/license_plate_detector.pt'
COCO_MODEL_DIR = "./models/yolov8n.pt"
folder_path = "./licenses_plates_imgs_detected/"

# OCR 리더 초기화
ocr = PaddleOCR(lang='korean')

# 차량 분류 번호 설정
vehicles = [2]

# 모델 로드
coco_model = YOLO(COCO_MODEL_DIR)
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            image = np.array(Image.open(file))
            results = model_prediction(image)
            if len(results) == 3:
                prediction = numpy_array_to_base64(results[0])  # 이미지 배열을 Base64로 인코딩
                texts = results[1]
                license_plate_crop = results[2]
                return render_template('results.html', prediction=prediction, texts=texts, license_plate_crop=license_plate_crop)
            else:
                prediction = numpy_array_to_base64(results[0])
                return render_template('results.html', prediction=prediction)
    return render_template('upload.html')

def numpy_array_to_base64(img_array):
    pil_img = Image.fromarray(img_array)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return img_str

def model_prediction(img):
    license_numbers = 0
    results = {}
    licenses_texts = []
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    object_detections = coco_model(img)[0]
    license_detections = license_plate_detector(img)[0]

    if len(object_detections.boxes.cls.tolist()) != 0 :
        for detection in object_detections.boxes.data.tolist() :
            xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection

            if int(class_id) in vehicles :
                cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 3)
    else :
            xcar1, ycar1, xcar2, ycar2 = 0, 0, 0, 0
            car_score = 0

    if len(license_detections.boxes.cls.tolist()) != 0 :
        license_plate_crops_total = []
        for license_plate in license_detections.boxes.data.tolist() :
            x1, y1, x2, y2, score, class_id = license_plate

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

            license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]

            img_name = '{}.jpg'.format(uuid.uuid1())
            cv2.imwrite(os.path.join(folder_path, img_name), license_plate_crop)

            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY) 
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop, img)
            licenses_texts.append(license_plate_text)

            if license_plate_text is not None and license_plate_text_score is not None:
                license_plate_crops_total.append(license_plate_crop)
                results[license_numbers] = {}
                results[license_numbers][license_numbers] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2], 'car_score': car_score},
                    'license_plate': {'bbox': [x1, y1, x2, y2], 'text': license_plate_text, 'bbox_score': score, 'text_score': license_plate_text_score}
                } 
                license_numbers += 1

        write_csv(results, f"./csv_detections/detection_results.csv")
        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return [img_wth_box, licenses_texts, license_plate_crops_total]
    
    else: 
        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return [img_wth_box]
    
def is_skewed(rect):
    # rect[2]는 최소 외접 사각형의 각도입니다.
    angle = abs(rect[2])
    if angle > 5 and angle < 85:  # ±5도 이상의 기울기만 보정
        return True
    return False
    
def correct_skew(image):
    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 이진화하여 이미지 내의 전경과 배경을 분리
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 컨투어 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 가장 큰 컨투어를 기준으로 처리
    max_contour = max(contours, key=cv2.contourArea)

    # 최소 외접 사각형 찾기
    rect = cv2.minAreaRect(max_contour)
    
    # 기울어져 있는지 확인
    if not is_skewed(rect):
        return image  # 기울어져 있지 않으면 원본 이미지를 반환

    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 박스 좌표를 상하좌우 순서로 정렬
    rect_pts = sorted(box, key=lambda x: x[1])  # 상단과 하단 기준으로 정렬
    top_pts = sorted(rect_pts[:2], key=lambda x: x[0])
    bottom_pts = sorted(rect_pts[2:], key=lambda x: x[0])

    ordered_pts = np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype="float32")

    # 대상 사각형 좌표를 설정
    width = int(rect[1][0])
    height = int(rect[1][1])

    dst_pts = np.array([[0, 0],
                        [width-1, 0],
                        [width-1, height-1],
                        [0, height-1]], dtype="float32")

    # 원근 변환 매트릭스 계산
    M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)

    # 이미지를 원근 변환하여 직선화
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped

    
def upscale_and_enhance_image(image, scale_factor=2):
    # 이미지 업스케일링
    upscaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # 샤프닝 커널 적용 (선명도 개선)
    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]])
    sharpened_image = cv2.filter2D(upscaled_image, -1, kernel)
    
    return sharpened_image

def read_license_plate(license_plate_crop, img):
    # 기울기 보정
    license_plate_crop = correct_skew(license_plate_crop)

    # 이미지 업스케일링 및 화질 개선
    license_plate_crop = upscale_and_enhance_image(license_plate_crop)

    # PaddleOCR로 텍스트 인식
    tests = ocr.ocr(license_plate_crop, cls=True)

    # 필터링된 텍스트, 박스, 점수를 저장할 리스트
    filtered_texts = []
    filtered_boxes = []
    filtered_scores = []

    # OCR 결과에서 신뢰도 점수가 0.7 이상인 텍스트만 필터링
    for line in tests[0]:
        text = line[1][0]
        score = line[1][1]
        box = line[0]

        if score >= 0.7:
            filtered_texts.append((text, box, score))

    # 좌표를 기준으로 텍스트 정렬 (좌측에서 우측 순)
    filtered_texts = sorted(filtered_texts, key=lambda x: x[1][0][0])

    # 정렬된 결과에서 텍스트, 박스, 점수 분리
    texts = [item[0] for item in filtered_texts]
    boxes = [item[1] for item in filtered_texts]
    scores = [item[2] for item in filtered_texts]

    # 특수문자와 영문자를 제거한 후 텍스트 연결
    cleaned_text = re.sub(r'[a-zA-Z!@#$%^&*(),.?":{}|<>\-]', '', "".join(texts))

    # 이미지에 텍스트와 점수 그리기
    pil_img = Image.fromarray(cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_path = "font/NanumGothicBold.ttf"  # 폰트 파일 경로 설정
    font = ImageFont.truetype(font_path, 12)  # 폰트 크기 설정

    for (box, text, score) in zip(boxes, texts, scores):
        (top_left, top_right, bottom_right, bottom_left) = box
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # 텍스트 및 확률 출력
        print(f"Detected text: {text} (Probability: {score:.2f})")

        # 텍스트와 점수를 표시할 문자열
        display_text = f"{text} ({score:.2f})"

        # 박스 그리기
        draw.rectangle([top_left, bottom_right], outline=(0, 255, 0), width=2)
        draw.text((bottom_left[0], bottom_left[1] + 5), display_text, font=font, fill=(255, 0, 0))  # 빨간색 텍스트

    # 결과 이미지를 저장
    pil_img.save("a.jpg")

    # 필터링된 점수들로 점수 계산
    total_score = sum(scores)
    average_score = total_score / len(scores) if len(scores) > 0 else 0

    # 최종 텍스트와 평균 점수를 반환
    return cleaned_text, average_score
    
def read_license_plate_paddle(license_plate_crop):
    result = ocr.ocr(license_plate_crop, cls=True)

    # PIL의 이미지를 수정하기 위한 객체 생성
    pil_img = Image.fromarray(cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_path = "font/NanumGothicBold.ttf"  # 폰트 파일 경로 설정
    font = ImageFont.truetype(font_path, 12)  # 폰트 크기 설정

    # 결과 시각화
    boxes = [line[0] for line in result[0]]
    texts = [line[1][0] for line in result[0]]
    scores = [line[1][1] for line in result[0]]

    # 이미지에 박스와 텍스트 그리기
    for (box, text, score) in zip(boxes, texts, scores):
        (top_left, top_right, bottom_right, bottom_left) = box
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # 박스 그리기
        draw.rectangle([top_left, bottom_right], outline=(0, 255, 0), width=2)
        draw.text(bottom_right, f'{text} ({score:.2f})', font=font, fill=(255, 0, 0))  # 빨간색 텍스트
    pil_img.save("a.jpg")
    # 시각화된 이미지를 다시 넘파이 배열로 변환
    img_with_text = np.array(pil_img)
    
    return img_with_text, texts, scores

def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score\n')
        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                if 'car' in results[frame_nmr][car_id].keys() and 'license_plate' in results[frame_nmr][car_id].keys() and 'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(
                        frame_nmr,
                        car_id,
                        '[{} {} {} {}]'.format(
                            results[frame_nmr][car_id]['car']['bbox'][0],
                            results[frame_nmr][car_id]['car']['bbox'][1],
                            results[frame_nmr][car_id]['car']['bbox'][2],
                            results[frame_nmr][car_id]['car']['bbox'][3]
                        ),
                        '[{} {} {} {}]'.format(
                            results[frame_nmr][car_id]['license_plate']['bbox'][0],
                            results[frame_nmr][car_id]['license_plate']['bbox'][1],
                            results[frame_nmr][car_id]['license_plate']['bbox'][2],
                            results[frame_nmr][car_id]['license_plate']['bbox'][3]
                        ),
                        results[frame_nmr][car_id]['license_plate']['bbox_score'],
                        results[frame_nmr][car_id]['license_plate']['text'],
                        results[frame_nmr][car_id]['license_plate']['text_score']
                    ))
        f.close()

if __name__ == '__main__':
    app.run(debug=True)
