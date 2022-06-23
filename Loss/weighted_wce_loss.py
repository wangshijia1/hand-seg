"""
带权重的交叉熵loss — Weighted cross-entropy (WCE)
R为标准的分割图，其中rn为label 分割图中的某一个像素的GT。
P为预测的概率图，pn为像素的预测概率值，背景像素图的概率值就为1-P。
只有两个类别的带权重的交叉熵为：
WCE = -1/N * sum(wrn * log(pn) + (1-rn) * log(1-pn))
w为权重,
w = (N-sum(pn) )/ sum(pn)
缺点是需要人为的调整困难样本的权重，增加调参难度。
"""