import torch
import torch.nn as nn
import torch.nn.functional as F


'''
L = -log(y') , y = 1
L = -log(1-y'), y = 0

二分类： L = -1/N * sum(yi * log(pi) + (1-yi) * log(1-pi))
其中，yi  为输入实例 xi 的真实类别, pi为预测输入实例 xi 属于类别 1 的概率. 
对所有样本的对数损失表示对每个样本的对数损失的平均值, 对于完美的分类器, 对数损失为 0。
此loss function每一次梯度的回传对每一个类别具有相同的关注度！
所以极易受到类别不平衡的影响，在图像分割领域尤其如此。
'''

#二值交叉熵，这里输入要经过sigmoid处理
nn.BCELoss(F.sigmoid(input), target)


#多分类交叉熵, 用这个 loss 前面不需要加 Softmax 层
nn.CrossEntropyLoss(input, target)
