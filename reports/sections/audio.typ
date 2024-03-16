#import emoji: checkmark, crossmark, construction

= 音频模态特征提取对比实验
== 方法和模型
=== 特征提取方法
- ComParE_2016
- #checkmark GeMAPSv01a
- #checkmark GeMAPSv01b
- #checkmark eGeMAPSv01a
- #checkmark eeGeMAPSv01b
- #checkmark eGeMAPSv02
- #checkmark emobase

=== Classification
- #checkmark Conv1d Network

== 实验
=== 特征提取
使用每个检测模型和识别模型（特征提取部分）分别提取特征，命名成
`f"{detection_method}_{recognition_method}.pkl"`。

=== 分类模型训练
对每类特征在训练集分别进行 `10000` 批次的训练，在满足早退条件情况下停止训练。 目前给出了精度和`weighted-F1`。

== 结果
// typstfmt::off
#table(
  columns: (auto, auto, auto, auto,auto),
  inset: 10pt,
  align: horizon,
  [*特征提取方法*], [*特征层级*],[*网络模型*], [*精度*], [*weighted-F1*],
  [GeMAPSv01a], [lld],[conv1d], [52.40\%], [52.37\%],
  [GeMAPSv01b], [lld], [conv1d],[49.60\%], [48.55\%],
  [eGeMAPSv01a], [lld],[conv1d], [55.97\%], [56.39\%],
  [eeGeMAPSv01b], [lld],[conv1d], [56.87\%], [57.20\%],
  [eGeMAPSv02], [lld],[conv1d], [59.10\%], [59.02\%],
  [emobase], [lld],[conv1d], [69.90\%], [70.01\%],
  [emobase], [func], [Conv1d], [76.43%], [38.01%],
  [eGeMAPSv01b], [func], [Conv1d], [63.67%], [15.44%],
  [eGeMAPSv01a], [func], [conv1d], [61.13%], [31.68%],
  [eGeMAPSv02], [Func], [Conv1d], [61.03%], [14.81%],
  [GeMAPSv01a], [func], [conv1d], [60.1%], [31.6%],
  [GeMAPSv01b], [func], [Conv1d], [59.87%], [30.5%],
  [emobase], [lld], [GRU], [77.87%], [28.27%],
  [eGeMAPSv01a], [func], [GRU], [61.5%], [23.83%],
  [eGeMAPSv01b], [func], [GRU], [60.27%], [23.15%],
  [eGeMAPSv02], [func], [GRU], [59.7%], [23.15%],
  [eGeMAPSv01a], [func], [GRU], [56.63%], [21.56%],
  [emobase], [func], [GRU], [56.2%], [22.16%],
  [GeMAPSv01b], [func], [GRU], [56.03%], [21.65%],
  [eGeMAPSv01a], [lld], [GRU], [42.9%], [18.13%],
  [GeMAPSv01a], [lld], [GRU], [36.27%], [14.77%],
  [eGeMAPSv01b], [lld], [GRU], [35.73%], [14.14%],
  [eGeMAPSv02], [lld], [GRU], [35.17%], [14.42%],
  [GeMAPSv01b], [lld], [GRU], [34.63%], [14.64%],
  [emobase], [lld], [LSTM], [69.93%], [25.52%],
  [eGeMAPSv02], [func], [LSTM], [59.2%], [22.47%],
  [eGeMAPSv01b], [func], [LSTM], [58.13%], [22.44%],
  [GeMAPSv01a], [func], [LSTM], [56.47%], [22.36%],
  [eGeMAPS], [func], [LSTM], [56.37%], [21.22%],
  [GeMAPSv01b], [func], [LSTM], [54.87%], [22.14%],
  [emobase], [func], [LSTM], [51.07%], [16.06%],
  [eGeMAPSv01b], [lld], [LSTM], [35.43%], [13.49%],
  [GeMAPSv01b], [lld], [LSTM], [33.33%], [11.44%],
  [eGeMAPSv02], [lld], [LSTM], [32.07%], [11.54%],
  [eGeMAPS], [lld], [LSTM], [29.1%], [10.83%]
)
// typstfmt::on

== 数据集

=== ESD
数据集由若干音频组成，音频的命名格式为如下
```sh
$DATASETS/Face-Dataset/ESD/{i:04d}/{emotion}/{set_type}/{i:04d}_{j:06d}.wav
```
其中i表示人的编号，$j$表示音频的编号，$i$的范围是$1-20$，$j$从$0$开始。


== 参考资料
- #link("https://github.com/audeering/opensmile-python")[opensmile-python]
