from flask import Flask, render_template, request, redirect, url_for, jsonify
from paddleocr import PaddleOCR
import numpy as np
from realesrgan import RealESRGANer
from PIL import Image, ImageDraw, ImageFont, ExifTags
import base64
from io import BytesIO
from ultralytics import YOLO
import cv2
import re
import os
from flask_cors import CORS
# import torch
# from basicsr.archs.rrdbnet_arch import RRDBNet

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Real-ESRGAN 모델 경로
# model_path = './models/RealESRGAN_x4plus.pth'

# model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

# # RealESRGANer 클래스 인스턴스화
# upscaler = RealESRGANer(
#     scale=2,  # 업스케일링 배율 설정
#     model_path=model_path,  # 모델 가중치 파일 경로
#     model=model,  # 생성한 모델 전달
#     tile=0,  # 타일 크기, 메모리 문제를 피하기 위해 사용할 수 있음
#     tile_pad=10,
#     pre_pad=10,
#     half=False,
#     device=device
# )



app = Flask(__name__)
CORS(app)

# 모델 경로 설정
LICENSE_MODEL_DETECTION_DIR = './models/license_plate_detector.pt'
COCO_MODEL_DIR = "./models/yolov8n.pt"
folder_path = "./licenses_plates_imgs_detected/"

# OCR 리더 초기화
ocr = PaddleOCR(use_angle_cls=True,lang='korean')

# 차량 분류 번호 설정
vehicles = [2]

# 모델 로드
coco_model = YOLO(COCO_MODEL_DIR)
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_api():
    print("call /api/upload")
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    image = Image.open(file)
    image = correct_image_orientation(image)

    image = np.array(image)
    cv2.imwrite("after_processing.jpg", image)
    results = model_prediction(image)

    if len(results) >= 3:
        prediction = numpy_array_to_base64(results[0])
        texts = list(set(results[1]))  # 중복된 텍스트 제거
        pil_img_base64 = results[3]

        response = {
            "prediction": prediction,
            "texts": texts,
            "license_plate_image": pil_img_base64  # Base64 인코딩된 이미지
        }
        return jsonify(response)
    else:
        prediction = numpy_array_to_base64(results[0])
        return jsonify({"prediction": prediction})
    
@app.route('/api/ocr', methods=['GET', 'POST'])
def ocr_api():
    if request.method == "POST":
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        image = np.array(Image.open(file))
        ocr_text, original_img, boxed_img = ocr_image(image)

        response = {
            "original_image": original_img,
            "boxed_image": boxed_img,
            "ocr_text": ocr_text
        }
        return jsonify(response)
    else:
        return render_template('ocr_upload.html')

def correct_image_orientation(image):
    try:
        # EXIF 데이터에서 orientation 태그를 찾습니다.
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = image._getexif()

        if exif is not None:
            orientation = exif.get(orientation, 1)

            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # EXIF 정보가 없거나 처리 중 문제가 발생해도 그대로 진행
        pass

    return image
    
def pil_image_to_base64(pil_img):
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return img_str
    
def ocr_image(image):
    # PaddleOCR로 텍스트 인식
    result = ocr.ocr(image, cls=True)
    texts = [line[1][0] for line in result[0]]
    
    # 결과 이미지를 생성 (박스 처리된 이미지)
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    for line in result[0]:
        box = line[0]
        draw.rectangle([tuple(box[0]), tuple(box[2])], outline=(0, 255, 0), width=2)
    
    # 원본 이미지를 Base64로 인코딩
    original_img_base64 = numpy_array_to_base64(image)
    # 박스 처리된 이미지를 Base64로 인코딩
    boxed_img_base64 = pil_image_to_base64(pil_img)
    
    # 텍스트를 한 줄로 연결
    cleaned_text = " ".join(texts)

    # 결과를 반환
    return cleaned_text, original_img_base64, boxed_img_base64

def numpy_array_to_base64(img_array):
    # RGBA 이미지를 RGB로 변환
    if img_array.shape[2] == 4:  # 채널이 4개인 경우
        img_array = Image.fromarray(img_array).convert('RGB')
    else:
        img_array = Image.fromarray(img_array)
    
    buff = BytesIO()
    img_array.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return img_str

def model_prediction(img):
    license_numbers = 0
    results = {}
    licenses_texts = []

    #img, _ = upscaler.enhance(img, outscale=4)

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

            # license_plate_crop, _ = upscaler.enhance(license_plate_crop, outscale=2)

            license_plate_text, license_plate_text_score, pil_img_base64  = read_license_plate(license_plate_crop, img)
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
        return [img_wth_box, licenses_texts, license_plate_crops_total, pil_img_base64]
    
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
    #license_plate_crop = correct_skew(license_plate_crop)

    # 이미지 업스케일링 및 화질 개선
    print("read license")
    license_plate_crop = upscale_and_enhance_image(license_plate_crop)

    # PaddleOCR로 텍스트 인식
    print("read text")
    tests = ocr.ocr(license_plate_crop, cls=True)

    print("read text complete")

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

    # Y 좌표를 기준으로 먼저 정렬하고, 같은 Y 좌표 내에서는 X 좌표를 기준으로 정렬
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
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # 필터링된 점수들로 점수 계산
    total_score = sum(scores)
    average_score = total_score / len(scores) if len(scores) > 0 else 0

    # 최종 텍스트와 평균 점수를 반환
    return cleaned_text, average_score, img_base64

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
    app.run(debug=True, host='0.0.0.0', port=5500)
