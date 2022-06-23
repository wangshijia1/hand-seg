import os
import cv2
import numpy as np
from PIL import Image
from torch.utils import data
from utils import helpers

'''
1 = vein
0   = background 
'''
# palette = [[0], [1]]  # one-hot的颜色表
num_classes = 1  # 分类数


def make_dataset(root, mode, fold):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        img_path = os.path.join(root, 'image')
        mask_path = os.path.join(root, 'label')

        if 'Augdata' in root:  # 当使用增广后的训练集
            data_list = os.listdir(os.path.join(root, 'label'))
        else:
            data_list = [l.strip('\n') for l in open(os.path.join(root, 'train{}.txt'.format(fold))).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it))
            items.append(item)
    elif mode == 'val':
        img_path = os.path.join(root, 'image')
        mask_path = os.path.join(root, 'label')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'val{}.txt'.format(fold))).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it))
            items.append(item)
    else:
        img_path = os.path.join(root, 'image')
        mask_path = os.path.join(root, 'label')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'test.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it))
            items.append(item)
    return items


class Dataset(data.Dataset):
    def __init__(self, root, mode, fold, joint_transform=None, center_crop=None, transform=None, target_transform=None):
        self.imgs = make_dataset(root, mode, fold)
        # self.palette = palette
        self.mode = mode
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 image, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.center_crop = center_crop
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        img_path, mask_path = self.imgs[index]
        file_name = mask_path.split('\\')[-1]

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.center_crop is not None:
            img, mask = self.center_crop(img, mask)
        img = np.array(img)
        mask = np.array(mask)
        # Image.open读取灰度图像时shape=(H, W) 而非(H, W, 1)
        # 因此先扩展出通道维度，以便在通道维度上进行one-hot映射
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)
        # mask = helpers.mask_to_onehot(mask, self.palette)
        # shape from (H, W, C) to (C, H, W)
        img = img.transpose([2, 0, 1])
        mask = mask.transpose([2, 0, 1])
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return (img, mask), file_name

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    np.set_printoptions(threshold=9999999)

    from torch.utils.data import DataLoader
    import utils.image_transforms as joint_transforms
    import utils.transforms as extended_transforms


    # 测试加载数据类
    def demo():
        train_path = r'../media/Datasets/Leaf/train'
        val_path = r'../media/Datasets/Leaf/train'
        test_path = r'../media/Datasets/Leaf/test'

        center_crop = joint_transforms.Resize(112)
        test_center_crop = joint_transforms.SingleResize(112)
        train_input_transform = extended_transforms.NpyToTensor()
        target_transform = extended_transforms.MaskToTensor()

        train_set = Dataset(train_path, 'train', 1,
                            joint_transform=None, center_crop=center_crop,
                            transform=train_input_transform, target_transform=target_transform)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=False)

        # for (input, mask), file_name in train_loader:
        #     print(input.shape)
        #     print(mask.shape)
        #     y = input.squeeze()
        #     img = helpers.array_to_img(y,data_format='channels_first',scale=False)
        #     # 将gt反one-hot回去以便进行可视化
        #     palette = [[0, 0, 0], [255, 0, 0], [0, 0, 255],[255, 255, 0]]
        #     z = np.array(mask.squeeze()).transpose(1, 2, 0)
        #     gt = helpers.onehot_to_mask(z, palette)
        #     gt = helpers.array_to_img(gt,data_format='channels_last',scale=False)
        #     # cv2.imshow('img GT', np.uint8(np.hstack([img, gt])))
        #     # cv2.imshow('img GT', np.uint8(img))
        #     # cv2.waitKey(1000)
        #     # cv2.imshow('img GT', np.uint8(gt))
        #     # cv2.waitKey(1000)
        #     gt.save(os.path.join('E:/毕业论文/多分类/pytorch-medical-image-segmentation-master/media/Datasets/Leaf/train/label_rgb',file_name[0]))


    demo()
