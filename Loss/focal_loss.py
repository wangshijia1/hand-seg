"""
Focal loss是何恺明针对训练样本不平衡提出的loss 函数。公式：
FL（Pt） = - α * (1-Pt)^β * log(Pt)
        = - 1/N * sum(α * yi * (1-Pi)^β * log(Pi) + (1-α) * (1-yi) * Pi^β * log(1-Pi))

可以认为，focal loss是交叉熵上的变种，

针对以下两个问题设计了两个参数 α\ alpha、β\beta：
1.正负样本不平衡，比如负样本太多；
2.存在大量的简单易分类样本。

第一个问题，容易想到可以在loss函数中，给不同类别的样本loss加权重，正样本少，就加大正样本loss的权重，这就是focal loss里面参数 α \alphaα的作用；
第二个问题，设计了参数β \betaβ，从公式里就可以看到，当样本预测值pt比较大时，也就是易分样本，（1-pt)^beta 会很小，这样易分样本的loss会显著减小，模型就会更关注难分样本loss的优化。

目前在图像分割上只是适应于二分类。

"""
import torch
import torch.nn as nn
# --------------------------- BINARY LOSSES ---------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)

    def forward(self, preds, labels):
        if self.ignore_index is not None:
            mask = labels != self.ignore
            labels = labels[mask]
            preds = preds[mask]

        logpt = -self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss
# --------------------------- MULTICLASS LOSSES ---------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss

