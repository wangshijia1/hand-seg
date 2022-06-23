import os, shutil

# 创建测试集的文件名清单
def create_testfile(dataset_dir, save_path):
    if os.path.exists(os.path.join(save_path, 'test.txt')):
        # 若该目录已存在，则先删除，用来清空数据
        print('清空原始数据中...')
        os.remove(os.path.join(save_path, 'test.txt'))
        print('原始数据已清空。')

    data_list = os.listdir(dataset_dir)
    for i, test in enumerate(data_list):
        with open(os.path.join(save_path, 'test.txt'), 'a') as f:
            f.write(test)
            f.write('\n')


if __name__ == '__main__':
    # 膀胱测试数据集
    # 首次划分数据集或者重新划分数据集时运行
    # dataset_kfold(os.path.join('..\media\Datasets\Bladder', 'raw_data\Labels'),
    #               os.path.join('..\media\Datasets\Bladder', 'raw_data'))
    create_testfile(os.path.join('..\media\Datasets\Leaf', 'test\label'),
                  os.path.join('..\media\Datasets\Leaf', 'test'))