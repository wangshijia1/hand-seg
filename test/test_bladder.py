import os
import cv2
import torch
import shutil
import copy
from PIL import Image
import utils.image_transforms as joint_transforms
from torch.utils.data import DataLoader
import utils.transforms as extended_transforms
from datasets import bladder
from utils.loss import *
from networks.u_net import Baseline
from tqdm import tqdm

# from networks.LadderNetv65_newtry1 import LadderNetv6
from networks.LadderNetv65 import LadderNetv6
# from networks.segnet import Baseline
# from networks.FCN_ResNet import Baseline
# from network.modeling import deeplabv3plus_resnet50
from networks.DeeplabV3Plus import Deeplabv3plus_res50
# from networks.unet_small import UNet
# from networks.unet_vgg16_pretain import Baseline
# from networks.unet_small_vgg16_pretrain import UNet
# from networks.my_model import AttU_Net
# from networks.attu_net_small import AttU_Net
# from networks.attunet_small_vgg16_pretrain import AttU_Net
# from networks.attunet_vgg16_pretrain import AttU_Net
# from  networks.unet_without_att_vgg16_pretrain import U_Net
# from networks.unet_small import UNet
# from networks.attunet_vgg16_pretrain_withnewDAC import AttU_Net
# from networks.attunet_vgg16_pretrain_withnewDAC_xg1 import AttU_Net
from networks.unet_vgg16pretain import Baseline


# from tensorboardX import SummaryWriter
# import torchvision.utils as vutils

# 指定使用gpu1
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

crop_size = 112
test_path = r'..\media/Datasets/Leaf/test'
center_crop = joint_transforms.Resize(crop_size)
test_input_transform = extended_transforms.NpyToTensor()
target_transform = extended_transforms.MaskToTensor()

test_set = bladder.Dataset(test_path, 'test', 1,
                              joint_transform=None, transform=test_input_transform, center_crop=center_crop,
                              target_transform=target_transform)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

palette = [[0], [255]]
palette_gray = [[0], [1]]

num_classes = 1

# net = Baseline(img_ch=3, num_classes=num_classes, depth=2).cuda() #unet
# net = UNet(n_channels=3,n_classes=bladder.num_classes,bilinear=True).cuda() # unet_small
# net = Baseline(img_ch=3,num_classes=bladder.num_classes,depth=2).cuda() #unet_vgg16_pretrain
# net = Baseline(num_classes=num_classes).cuda()
# net = Baseline(num_classes=num_classes).cuda()  #segnet
# net = Baseline(num_classes=num_classes).cuda() # FCN_ResNet
# net = deeplabv3plus_resnet50(num_classes=bladder.num_classes).cuda()  # deeplabv3plus_resnet50
# net = Deeplabv3plus_res50(num_classes=bladder.num_classes).cuda()  # deeplabv3plus_resnet50
# net = LadderNetv6(layers=4, filters=10, num_classes=num_classes, inplanes=3).cuda()
# net = LadderNetv6(layers=4,filters=10,num_classes=num_classes,inplanes=3).cuda()
# net = UNet(n_channels=3,n_classes=bladder.num_classes,bilinear=True).cuda() # unet_small_vgg16_pretrain
# net = AttU_Net(img_ch=3,output_ch=bladder.num_classes).cuda()  #attu_net
# net = AttU_Net(n_channels=3, n_classes=bladder.num_classes, bilinear=True).cuda() # attu_net_small
# net = AttU_Net(n_channels=3,n_classes=bladder.num_classes,bilinear=True).cuda() # attu_net_small_vgg16_pretrain
# net = AttU_Net(img_ch=3,output_ch=bladder.num_classes).cuda() # attu_net_vgg16_pretrain
# net = U_Net(img_ch=3,output_ch=bladder.num_classes).cuda()  # unet_without_att_vgg16_pretrain
# net = UNet(n_channels=3,n_classes=bladder.num_classes,bilinear=False).cuda() # unet_small_deconv
# net =AttU_Net(img_ch=3,output_ch=bladder.num_classes).cuda() # attunet_vgg16_pretrain_withnewDAC
# net =AttU_Net(img_ch=3,output_ch=bladder.num_classes).cuda() # attunet_vgg16_pretrain_withnewDAC
net = Baseline(img_ch=3,num_classes=bladder.num_classes,depth=2).cuda()  #unet-vgg16


net.load_state_dict(torch.load("../checkpoint/u_net_depth=2_fold_1_dice_801717_batch1.pth"))
net.eval()


def auto_test(net):
    # feature map 可视化
    # log_dir = r'..\results1'
    # writer = SummaryWriter(log_dir=log_dir, filename_suffix="_feature map")

    save_path = './unet_results2'
    img_path = os.path.join(save_path, 'images')
    pred_path = os.path.join(save_path, 'pred')
    gt_path = os.path.join(save_path, 'gt')

    for (input, mask), file_name in tqdm(test_loader):
        file_name = file_name[0].split('.')[0]

        X = input.cuda()
        pred = net(X)
        pred = torch.sigmoid(pred)
        pred = pred.cpu().detach()
        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1

        # # 可视化预处理
        # feature_map1 = pred.transpose_(0, 1)  # 交换第0维和第1维
        # feature_map_grid1 = vutils.make_grid(feature_map1, normalize=True, scale_each=True, nrow=2)
        # writer.add_image('feature map1', feature_map_grid1, global_step=2)
        # # writer.close()
        #
        # pred = pred.transpose_(0, 1)  # 交换第0维和第1维,还原
        ##   pred = torch.sigmoid(pred)
        #
        # feature_map2 = pred.transpose_(0, 1)  # 交换第0维和第1维
        # feature_map_grid2 = vutils.make_grid(feature_map2, normalize=True, scale_each=True, nrow=2)
        # writer.add_image('feature map2', feature_map_grid2, global_step=2)
        #
        # mask_map = mask.transpose_(0, 1)  # 交换第0维和第1维
        # mask_map_grid = vutils.make_grid(mask_map, normalize=True, scale_each=True, nrow=2)
        # writer.add_image('mask map', mask_map_grid, global_step=2)
        #
        # writer.close()
        #
        # pred = pred.transpose_(0, 1)  # 交换第0维和第1维,还原
        ## pred = pred.cpu().detach()

        # # 原图
        # m1 = np.array(input.squeeze())
        # m1 = helpers.array_to_img(m1,data_format='channels_first',scale=False)
        #
        # # gt
        # z = np.array(mask.squeeze()).transpose(1, 2, 0)
        # gt = helpers.onehot_to_mask(z, palette)
        # gt = helpers.array_to_img(gt,data_format='channels_last',scale=False)
        #
        # # pred
        # z = np.array(pred.squeeze()).transpose(1, 2, 0)
        # save_pred = helpers.onehot_to_mask(z, palette)
        # save_pred_png = helpers.array_to_img(save_pred,data_format='channels_last',scale=False)
        #
        # # png格式
        # m1.save(os.path.join(img_path, file_name + '.png'))
        # gt.save(os.path.join(gt_path, file_name + '.png'))
        # save_pred_png.save(os.path.join(pred_path, file_name + '.png'))

        # # 原图
        m1 = np.array(input.squeeze())
        m1 = helpers.array_to_img(m1, data_format='channels_first', scale=False)

        # gt
        gt = np.array(mask.squeeze())
        gt = np.asarray(gt, dtype='float32')
        gt[gt==1] = 255
        gt = Image.fromarray(gt.astype('uint8'))

        # pred
        pred = np.array(pred.squeeze())
        pred = np.asarray(pred, dtype='float32')
        pred[pred == 1] = 255
        pred = Image.fromarray(pred.astype('uint8'))

        # png格式
        m1.save(os.path.join(img_path, file_name + '.png'))
        gt.save(os.path.join(gt_path, file_name + '.png'))
        pred.save(os.path.join(pred_path, file_name + '.png'))

if __name__ == '__main__':
    np.set_printoptions(threshold=9999999)

    auto_test(net)