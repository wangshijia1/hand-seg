# pytorch-medical-image-segmentation

#### 配置要求：
1.Python 3.6
2.Pytorch 0.4
#### 如何运行：
1. 运行 ‘kflod.py’ 采用k折交叉验证划分数据集
2. 运行 ‘test_file_create.py’ 生成测试文件，将测试文件的所有文件名读取到txt文件中
3. 运行 ‘bladder.py’ 对数据集进行预处理
4. 运行 ‘train_bladder.py’ 对数据集进行训练
5. 运行 ‘test_bladder.py’ 预测,，结果存放在test/result中
6. 运行 ‘performance.py’ 计算评价指标，计算结果存放在test/result/performance.txt中
#### 预训练权重：
预训练好的权重将存放在/checkpoint
#### 数据集：
数据集存放在/media中
#### 网络模型：
本文选用的Unet、FCN、Laddernet、deeplabv3+网络模型存放在/network中
