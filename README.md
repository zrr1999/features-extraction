# Face detection, extraction and recognition

![实验进程架构图](archetecture.png)

## Face Detection
- Opencv
- SSD
- Dlib
- Mediapipe
- RetinaFace

## Face Recognition (Extraction)
- OpenFace
- DeepFace
- DeepID
- ArcFace

## Facial Feature Recognition
- VGG-Face

## classification
- VGG-Face

## deepface package
"opencv": OpenCvWrapper.build_model,
"ssd": SsdWrapper.build_model,
"dlib": DlibWrapper.build_model,
"mtcnn": MtcnnWrapper.build_model,
"retinaface": RetinaFaceWrapper.build_model,
"mediapipe": MediapipeWrapper.build_model,

"VGG-Face": VGGFace.loadModel,
"OpenFace": OpenFace.loadModel,
"Facenet": Facenet.loadModel,
"Facenet512": Facenet512.loadModel,
"DeepFace": FbDeepFace.loadModel,
"DeepID": DeepID.loadModel,

"Dlib": DlibWrapper.loadModel,
"ArcFace": ArcFace.loadModel,
"SFace": SFace.load_model,
"Emotion": Emotion.loadModel,
"Age": Age.loadModel,
"Gender": Gender.loadModel,
"Race": Race.loadModel,


## 数据集

### UCEC-Face
数据集由若干图片组成，每张图片包含一个人的脸部，图片的命名格式为如下
```sh
$DATASETS/Face-Dataset/UCEC-Face/subject{i}/subject{i}.{j}.png
```
其中i表示人的编号，j表示图片的编号，i的范围是1-130，j的范围是0-59，每个人有60张图片，共130个人，共7800张图片。


