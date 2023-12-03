# Face detection, extraction and recognition

## 模型
### Face Detection
- Opencv
- SSD
- ✅Dlib
- ✅Mediapipe
- RetinaFace

### Face Recognition (Feature Extraction)
- ✅OpenFace
- ✅DeepFace
- DeepID
- ✅ArcFace
- ✅Facenet

### Facial Feature Recognition
- VGG-Face

### Classification
- Fully Connected Layer
- VGG-Face

## 实验
### Deepface Package
支持的检测模型：
- opencv
- ssd
- dlib
- mtcnn
- retinaface
- mediapipe

支持的识别模型：
- VGG-Face
- OpenFace
- Facenet
- Facenet512
- DeepFace
- DeepID


### 特征提取
使用每个检测模型和识别模型（特征提取部分）分别提取特征，命名成 `f"{detection_method}_{recognition_method}.pkl"`。

## 结果


## 数据集

### UCEC-Face
数据集由若干图片组成，每张图片包含一个人的脸部，图片的命名格式为如下
```sh
$DATASETS/Face-Dataset/UCEC-Face/subject{i}/subject{i}.{j}.png
```
其中i表示人的编号，j表示图片的编号，i的范围是1-130，j从0开始，范围并不固定。


## 参考资料
![实验进程架构图](archetecture.png)
[Chinese Face Dataset for Face Recognition in an Uncontrolled Classroom Environment](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10210367)