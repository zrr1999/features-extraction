#import emoji: checkmark, crossmark, construction
#import "@local/bone-document:0.1.0": document-init
#import "utils.typ": include-section
#show: document-init.with(title: "多模态特征提取研究情况", author: "詹荣瑞，蔡小慧")

= 整体情况

== 模态
- 文字
- #checkmark 语音
- #checkmark 图像面部表情
- #checkmark 视频

== 早熟机制
- #checkmark Loss
- #checkmark Accuracy
- #checkmark Weighted-F1

== 指标
- #checkmark Accuracy
- #checkmark Weighted-F1

#include-section("face")
#include-section("audio")
#include-section("video")
