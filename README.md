## 甲骨文图像处理与识别项目

### 项目简介
本项目旨在对甲骨文原始拓片进行图像预处理，提取图像特征，建立甲骨文图像预处理模型，实现对甲骨文图像干扰元素的初步识别和处理。

### 项目目标
1. 实现甲骨文图像预处理，提取图像特征
2. 建立甲骨文图像分析模型，实现甲骨文图像分割和单字分割
3. 对甲骨文原始拓片进行自动文字识别，并呈现类似拍照翻译的效果

### 项目内容

#### Q1: 图像预处理与干扰元素识别
- 环境要求：openCV库
- 实现图像的灰度转换、二值化处理
- 使用中值滤波、均值滤波和非局部均值去噪技术进行图像去噪处理
- 在预处理过程中显示并保存处理后的图像
- `1_Pre_test/`: 存放待处理的原始图像的文件夹
- `1_After_test/`: 存放处理后图像的输出文件夹

#### Q2: 甲骨文图像分割模型
- 基于u-net模型
- 使用注意力机制改善模型效果
- 分析甲骨文原始拓片图像
- 建立快速准确的甲骨文图像分割模型
- 实现自动单字分割，并进行模型评估

#### Q3: 单字分割实验
- 利用甲骨文图像分割模型对原始拓片图像进行自动单字分割
- 将分割结果保存在“Test_results.xlsx”中

#### Q4: 文字自动识别
- 基于卷积神经网络
- 利用已标注的甲骨文字形数据集进行文字自动识别
- 呈现类似拍照翻译的效果

### 注意事项
- 代码结构复杂（shishan），谨慎运行
- 使用大量类似if(1==2):的控制方式，请注意
