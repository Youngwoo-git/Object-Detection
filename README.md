# Object_Detection
객체 검출 모듈 연구

많은 소타모델이 있지만, 현실적으로 관리되지 않는 디텍션 모듈(안정되지 않은)을 운영단계에서 도입할수 없음. 오픈소스로 잘 관리되고 있는 yolov5를 채택하는것이 맞을것으로 보임.

|모델명                                                  |데이터셋|성능|eval 결과 (mAP)|비고|
|---                                                    |---|---|---|---|
|[DAB-DETR](https://github.com/IDEA-opensource/DAB-DETR)|[coco test-dev](https://paperswithcode.com/sota/object-detection-on-coco)|box AP 63.3|mAP50 61.6 </br> mAP 44.8|전체 코드 없음|
|[SwinV2+HTC](https://github.com/microsoft/Swin-Transformer)|[coco test-dev](https://paperswithcode.com/sota/object-detection-on-coco)|box AP 63.1||swinv2 아직 mmdetection 지원 x|
|[YOLOv5x6 + TTA]([https://github.com/ultralytics/yolov5])|[coco eval](https://github.com/ultralytics/yolov5)|mAP 55.8|mAP50 70.8 </br> mAP 55.5|무|
|[YOLOR-D6](https://github.com/WongKinYiu/yolor)|coco dataset|mAP 	57.3|mAP50 68.5 </br> mAP 52.6|무|
|[YOLOX-s](https://github.com/Megvii-BaseDetection/YOLOX)|[Argoverse-HD dataset](https://paperswithcode.com/dataset/argoverse)|AP 40.5|mAP50 61.5 </br> mAP 45.3|무|
|[YOLOX-x](https://github.com/Megvii-BaseDetection/YOLOX)|[Argoverse-HD dataset](https://paperswithcode.com/dataset/argoverse)|AP 51.1|mAP50 70.7 </br> mAP 55.5|무|
|[DynamicHead](https://github.com/microsoft/DynamicHead)|coco dataset|COCO mAP 49.8|mAP50 68.4998 </br> mAP 49.6351|무|
|[GLIP-L](https://github.com/microsoft/GLIP)|FourODs,GoldG,CC3M+12M,SBU|COCO mAP 51.4||inference 속도가 약 0.25s 이기에 매우 느림. real time으로 활용하기에는 부적합|
|[YOLOv7](https://github.com/WongKinYiu/yolov7)|coco dataset|AP 51.4|mAP50 70.7 </br> mAP 55.1|무|
|[YOLOv7-e6e](https://github.com/WongKinYiu/yolov7)|coco dataset|AP 56.8|mAP50 72.1 </br> mAP 57.1|무|




위의 모델들에 대해 테스트하는 데이터셋에 대한 공개 여부는 비공개입니다.


# 성능 평가

## 평가 정보
- mp: mean precision
- mr: mean recall
- map50: mean average precision (IOU Threshold = 0.5)
- map:  mean average precision (0.05의 단계 크기로 0.5부터 0.95까지의 IoU에 대한 평균 AP)

## 실행 방법
```bash
python calc_map.py target_path=디텍션 결과가 저장된 json 파일 위치
```

## 입력 데이터


해당 코드는 아래 dictionary 형태로 detection 툴의 출력을 넘겨받아 작업합니다.

```python
# x1, y1: left top (float)
# x2, y2: right bottom (float)
# conf: 예측 confidence (float)
# cls: class index (float)
detection 결과 리스트 = [[x1, y1, x2, y2, conf, cls], [x1, y1, x2, y2, conf, cls]]

# 해당 딕셔너리들을 담은 리스트를 json 파일로 dump 하여주세요

files = sorted(glob('path/to/directory/*'))
dict_list = []

for file in files:
    detection_result = model(Image.open(file))
    pred_dict = {"img_name": 이미지 절대경로(string), "pred": detection 결과}
    dict_list.append(pred_dict)
with open('test.json', 'w') as f:
    json.dump(dict_list, f)
```

## 출력 예시

![출력](./figures/for_readme.jpg)
