# Face detection, extraction and recognition

## 方法和模型
### Face Detection
- Opencv
- SSD
- ✅ Dlib
- ✅ Mediapipe
- RetinaFace

### Face Recognition (Feature Extraction)
- ✅ OpenFace
- ✅ DeepFace
- DeepID
- ✅ ArcFace
- ✅ Facenet
- ✅ Facenet512

### Facial Feature Recognition
- ✅ VGG-Face

### Classification
- ✅ Fully Connected Layer
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

### 分类模型训练
对每类特征分别进行 `kfold-10` 的交叉 `10000` 批次的训练，在 `10` 次损失不下降的情况下停止训练（这个数值可能有点小，后续调整）。
目前给出了每次训练的精度和 `10` 次训练的平均精度。

早熟机制：
- ✅ Loss
- ✅ Accuracy
- ❌ Precision
- ❌ Recall
- ❌ F1
- 🚧 Weighted-F1

## 指标
- ✅ Accuracy
- ✅ Weighted-F1

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
