import logging
import os
import os.path as osp

import numpy as np
import torch
import sys

from PIL import Image
from tqdm import tqdm


if __name__ == '__main__':
    #logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')  # 初始化日志对象
    # file = open("./test/attu_net_small_vgg16_pretrain_522_results/performance_attu_net_small_vgg16_pretrain_522.txt", 'w')
    # output = sys.stdout
    outputfile = open("./test/laddernet_results/performance_laddernet.txt", 'w')
    sys.stdout = outputfile

    GT_paths = "./test/laddernet_results/gt"  # 真实值
    R_paths = "./test/laddernet_results/pred" # 预测值
    imgs_GT = os.listdir(GT_paths)   #真实值列表

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 定义device
    # 其中需要注意的是“cuda:0”代表起始的device_id为0，如果直接是“cuda”，同样默认是从0开始。可以根据实际需要修改起始位置，如“cuda:1”。
    # logging.info(f'Using device {device}')  # 输出日志的信息，logging.info('输出信息')
    img_num = len(imgs_GT)
    sumAcc, sumSen, sumPre, sumSpe, sumNPV, sumDSC, sumJacc, sumFPR, sumFNR, sumFDR, sumFOR, sumLR_plus, sumLR_sub, sumDOR = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i, mask_name in tqdm(enumerate(imgs_GT)):
        #logging.info("\nPredicting image {} ...".format(mask_name))

        GT_path = osp.join(GT_paths, mask_name)    #真实值路径
        R_path = osp.join(R_paths, mask_name)  #预测值路径

        img_GT = Image.open(GT_path)    #真实值
        h1,w1 = img_GT.size

        img_R = Image.open(R_path)    #预测值
        h2,w2 = img_R.size
        if h1 != h2:
            img_R = img_R.resize((h1, w1))
        #转为数组
        GT_array = np.array(img_GT)
        R_array = np.array(img_R)
        (h, w) = GT_array.shape     #高，宽

        TP, FP, TN, FN = 0, 0, 0, 0
        for p in range(h):
            for q in range(w):
                if GT_array[p, q]==255:
                    if R_array[p, q]==255 or R_array[p,q]==1:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if R_array[p, q] == 255 or R_array[p,q]==1:
                        FP += 1
                    else:
                        TN += 1
        #(1) Accuracy(Man)
        Acc = (TP + TN) / (TP + TN + FP + FN)
        sumAcc += Acc
        #(2) Recall(Sensitivity/TPR)
        Sen = TP / (TP + FN)
        sumSen += Sen
        #(3) Precision/PPV
        Pre = TP / (TP + FP)
        sumPre += Pre
        #(4) Specificity(TNR/Selectivity)
        Spe = TN / (TN + FP)
        sumSpe += Spe
        #(5) NPV
        NPV = TN / (TN + FN)
        sumNPV += NPV
        #(6) DSC(F1/F-score)
        DSC = 2 * TP / (2 * TP + FN + FP)
        sumDSC += DSC
        #(7) Jaccard(IoU/FS/CSI)
        Jacc = TP / (TP + FN + FP)
        sumJacc += Jacc
        #(8) FPR
        FPR = FP / (FP + TN+1e-7)
        sumFPR += FPR
        #(9) FNR
        FNR = FN / (FN + TP+1e-7)
        sumFNR += FNR
        #(10) FDR
        FDR = FP / (FP + TP+1e-7)
        sumFDR += FDR
        #(11) FOR
        FOR = FN / (FN + TN+1e-7)
        sumFOR += FOR
        #(12) LR+
        LR_plus = Sen / (FPR+1e-7)
        sumLR_plus += LR_plus
        #(13) LR-
        LR_sub = FNR / (Spe++1e-7)
        sumLR_sub += LR_sub
        #(14) DOR
        DOR = LR_plus / (LR_sub+1e-7)
        sumDOR += DOR
        # print(mask_name, Acc, Sen, Pre, Spe, NPV, DSC, Jacc, file=outputfile)
        print(mask_name, Acc, Sen, Pre, Spe, NPV, DSC, Jacc, FPR, FNR, FDR, FOR, LR_plus, LR_sub, DOR, file=outputfile)

        # print(mask_name,Acc, Sen, Pre, Spe, NPV, DSC, Jacc, FPR, FNR, FDR, FOR, LR_plus,LR_sub, DOR,sumAcc, sumSen, sumPre, sumSpe, sumNPV, sumDSC, sumJacc, sumFPR, sumFNR, sumFDR, sumFOR, sumLR_plus, sumLR_sub, sumDOR, file=outputfile)


    avgAcc = sumAcc / img_num           #(1)
    avgSen = sumSen / img_num           #(2)
    avgPre = sumPre / img_num           #(3)
    avgSpe = sumSpe / img_num           #(4)
    avgNPV = sumNPV / img_num           #(5)
    avgDSC = sumDSC / img_num           #(6)
    avgJacc = sumJacc / img_num         #(7)
    avgFPR = sumFPR / img_num           #(8)
    avgFNR = sumFNR / img_num           #(9)
    avgFDR = sumFDR / img_num           #(10)
    avgFOR = sumFOR / img_num           #(11)
    avgLR_plus = sumLR_plus / img_num   #(12)
    avgLR_sub = sumLR_sub / img_num     #(13)
    avgDOR = sumDOR / img_num           #(14)

    #输出结果
    print('1  Accuracy       {0:.4}'.format(avgAcc), file=outputfile)
    print('2  Sensitivity    {0:.4}'.format(avgSen), file=outputfile)
    print('3  Precision      {0:.4}'.format(avgPre), file=outputfile)
    print('4  Specificity    {0:.4}'.format(avgSpe), file=outputfile)
    print('5  NPV            {0:.4}'.format(avgNPV), file=outputfile)
    print('6  DSC(F1)        {0:.4}'.format(avgDSC), file=outputfile)
    print('7  IoU            {0:.4}'.format(avgJacc), file=outputfile)
    print('8  FPR            {0:.4}'.format(avgFPR), file=outputfile)
    print('9  FNR            {0:.4}'.format(avgFNR), file=outputfile)
    print('10 FDR            {0:.4}'.format(avgFDR), file=outputfile)
    print('11 FOR            {0:.4}'.format(avgFOR), file=outputfile)
    print('12 LR+            {0:.4}'.format(avgLR_plus), file=outputfile)
    print('13 LR-            {0:.4}'.format(avgLR_sub), file=outputfile)
    print('14 DOR            {0:.4}'.format(avgDOR), file=outputfile)
    print(img_num, file=outputfile)
    outputfile.close()  # close后才能看到写入的数据








