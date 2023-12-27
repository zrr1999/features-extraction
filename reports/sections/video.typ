#import emoji: checkmark, crossmark, construction

= 视频模态特征提取对比实验
== 方法和模型
=== 特征类型
- #checkmark r21d
- #checkmark s3d
- resnet

=== 分类模型
- #checkmark conv1d
- #checkmark conv1d + softmax

== 实验
=== 特征提取
+ 下载并使用 `video_features` 项目使用多个不同方法提取特征，并将每个视频的特征保存为一个文件（不同方法在不同文件夹中分类）。
+ 使预处理生成的特征文件，每个方法保存为一个文件，命名成`f"{method}_features.pkl"`。
+ 在特征上训练分类模型。

=== 分类模型训练
对每类特征分别进行 `kfold-10` 的交叉 `10000` 批次的训练，在 `10` 次损失不下降的情况下停止训练。 目前给出了每次训练的精度和 `10`
次训练的平均精度和`weighted-F1`。

== 结果
// typstfmt::off
#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 10pt,
  align: horizon,
  [*特征类型*], [*特征对齐长度*], [*网络模型*], [*精度*], [*weighted-F1*],
  [r21d], [10], [conv1d],[31.54\%], [20.93\%] ,
  [r21d], [10], [conv1d + softmax],[46.94\%], [21.82\%] ,
  [r21d], [10], [lstm+ softmax],[46.94\%], [21.82\%] ,
  [s3d], [10],  [conv1d], [30.10\%], [28.98\%] ,
  [s3d], [10],  [conv1d + softmax], [45.07\%], [28.01\%] ,
  [s3d], [10],  [lstm+ softmax], [45.07\%], [28.01\%] ,

  [r21d], [20], [conv1d],[32.86\%], [21.31\%] ,
  [s3d], [20],  [conv1d], [30.77\%], [29.60\%] ,

  [r21d], [20], [conv1d + logsoftmax],[36.51\%], [22.62\%] ,
  [s3d], [20],  [conv1d + logsoftmax], [31.80\%], [30.00\%] ,
  [r21d], [20], [lstm + logsoftmax],[40.57\%], [22.27\%] ,
  [s3d], [20],  [lstm + logsoftmax], [35.95\%], [28.73\%] ,
  [r21d], [20], [lstm(bi) + logsoftmax],[38.33\%], [21.39\%] ,
  [s3d], [20],  [lstm(bi) + logsoftmax], [33.10\%], [28.99\%] ,
  [r21d], [20], [gru(bi) + logsoftmax],[34.51\%], [21.65\%] ,
  [s3d], [20],  [gru(bi) + logsoftmax], [29.79\%], [28.84\%] ,
)
// typstfmt::on
== 数据集
使用 MELD 数据集，共7个类别。

== 参考资料
- #link("https://github.com/v-iashin/video_features")[video_features]
