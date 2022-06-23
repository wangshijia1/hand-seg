import os, shutil
from sklearn.model_selection import KFold


# 按K折交叉验证划分数据集
def dataset_kfold(dataset_dir, save_path):
    data_list = os.listdir(dataset_dir)

    kf = KFold(5, True, 12345)  # 使用5折交叉验证
    # kf = KFold(1, False, 1)  # 使用5折交叉验证

    for i, (tr, val) in enumerate(kf.split(data_list), 1):
        if i == 1:
            print(len(tr), len(val))
            if os.path.exists(os.path.join(save_path, 'train{}.txt'.format(i))):
                # 若该目录已存在，则先删除，用来清空数据
                print('清空原始数据中...')
                os.remove(os.path.join(save_path, 'train{}.txt'.format(i)))
                os.remove(os.path.join(save_path, 'val{}.txt'.format(i)))
                print('原始数据已清空。')

            for item in tr:
                file_name = data_list[item]
                with open(os.path.join(save_path, 'train{}.txt'.format(i)), 'a') as f:
                    f.write(file_name)
                    f.write('\n')

            for item in val:
                file_name = data_list[item]
                with open(os.path.join(save_path, 'val{}.txt'.format(i)), 'a') as f:
                    f.write(file_name)
                    f.write('\n')


if __name__ == '__main__':
    # 膀胱数据集划分
    # 首次划分数据集或者重新划分数据集时运行
    # dataset_kfold(os.path.join('..\media\Datasets\Bladder', 'raw_data\Labels'),
    #               os.path.join('..\media\Datasets\Bladder', 'raw_data'))

    # dataset_kfold(os.path.join('..\media\Datasets\Leaf', 'train\label'),
    #               os.path.join('..\media\Datasets\Leaf', 'train'))

    dataset_kfold(os.path.join('..\media\Datasets\Leaf_cut', 'train\label'),
                  os.path.join('..\media\Datasets\Leaf_cut', 'train'))