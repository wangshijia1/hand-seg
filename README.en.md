# pytorch-hand-image-segmentation

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
#### 安装包：
Torch,os,numpy,cv2,sys,PIL,tqdm,logging,shutil,sklearn,tensorboardX,random,time
#### 目录结构描述：
|——readme.md			//使用说明文档  <br />
|——.idea			//配置文件<br />
|——checkpoint				//训练好的权重文件<br />
|——dataprepare            //数据准备<br />
	|——.pytest_cache       //缓存测试 <br />			
	|——kflod.py            //采用k折交叉验证划分数据集<br />
|——test_file_create.py    //生成测试文件，将测试图片的所有图片名读取到txt文件中<br />
|——datasets          <br />
|——bladder.py          //对数据集进行预处理<br />
|——log                //日志文件，记录训练日志<br />
|——Loss               //各种损失函数的计算<br />
|——dice_loss.py<br />
|——focal_loss.py<br />
|——IoU_loss.py<br />
|——log_loss.py<br />
|——lovasz_softmax_loss.py<br />
|——ohem_loss.py<br />
|——weighted_wce_loss.py<br />
|——media<br />
|——Datasets         //数据集存放在此处<br />
|——networks            //各种网络模型<br />
|——base_model<br />
|——custom_modules<br />
|——sync_batchnorm<br />
|——PSPNet<br />
|——test<br />
|——result               //存放测试结果<br />
|——test_bladder.py          //测试程序<br />
|——train<br />
|——train_bladder.py           //训练程序<br />
|——validate<br />
|——result<br />
|——validate_bladder.py        //验证程序<br />
|——utils              //一些功能函数<br />
|——performance.py    //计算各项性能指标<br />
