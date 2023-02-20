import os, random, shutil
import numpy as np


def make_dir(source, target):
    '''
    创建和源文件相似的文件路径函数
    :param source: 源文件位置
    :param target: 目标文件位置
    '''
    dir_names = os.listdir(source)
    # for names in dir_names:
    for i in ['trainA', 'trainB', 'valA', 'valB', 'testA', 'testB']:
        path = target + '/' + i
        if not os.path.exists(path):
            os.makedirs(path)


def split_and_save_AB(source, target, type,  patient, slice):
    src_path_AB = source + '/' + patient + '/' + slice
    # print(src_path_AB)
    if type == 0:
        save_path_A = target + '/trainA/' + patient + '/' + slice
        save_path_B = target + '/trainB/' + patient + '/' + slice
    elif type == 1:
        save_path_A = target + '/valA/' + patient + '/' + slice
        save_path_B = target + '/valB/' + patient + '/' + slice
    elif type == 2:
        save_path_A = target + '/testA/' + patient + '/' + slice
        save_path_B = target + '/testB/' + patient + '/' + slice

    AB = np.load(src_path_AB)
    AB = AB[0]
    w, h = AB.shape
    A = AB[0: w//2, :]
    B = AB[w//2: w, :]

    np.save(save_path_A, A)
    np.save(save_path_B, B)


def divideTrainValiTest(source, target):
    '''
        创建和源文件相似的文件路径
        :param source: 源文件位置
        :param target: 目标文件位置
    '''

    # 得到源文件下的种类
    patients = os.listdir(source)
    random.shuffle(patients)
    print(int(0.8 * len(patients)), int(0.9 * len(patients)), int(0.9 * len(patients)))
    train_list = patients[0:int(0.8 * len(patients))]
    valid_list = patients[int(0.8 * len(patients)):int(0.9 * len(patients))]
    test_list = patients[int(0.9 * len(patients)):]
    print(train_list)
    print(valid_list)
    print(test_list)
    # 对于每个图片，移入到对应的文件夹里面
    for train_patient in train_list:
        if not os.path.exists(target + '/trainA/' + train_patient):
            os.makedirs(target + '/trainA/' + train_patient)
        if not os.path.exists(target + '/trainB/' + train_patient):
            os.makedirs(target + '/trainB/' + train_patient)
        for slice in os.listdir(source + '/' + train_patient):
            # print(slice)
            # shutil.copyfile(source + '/' + train_patient + '/' + slice, target + '/train/' + train_patient + '/' + slice)
            split_and_save_AB(source, target, 0, train_patient, slice)

    for validation_patient in valid_list:
        if not os.path.exists(target + '/valA/' + validation_patient):
            os.makedirs(target + '/valA/' + validation_patient)
        if not os.path.exists(target + '/valB/' + validation_patient):
            os.makedirs(target + '/valB/' + validation_patient)
        for slice in os.listdir(source + '/' + validation_patient):
            # print(slice)
            # shutil.copyfile(source + '/' + validation_patient + '/' + slice, target + '/val/' + validation_patient + '/' + slice)
            split_and_save_AB(source, target, 1, validation_patient, slice)

    for test_patient in test_list:
        if not os.path.exists(target + '/testA/' + test_patient):
            os.makedirs(target + '/testA/' + test_patient)
        if not os.path.exists(target + '/testB/' + test_patient):
            os.makedirs(target + '/testB/' + test_patient)
        for slice in os.listdir(source + '/' + test_patient):
            # print(slice)
            # shutil.copyfile(source + '/' + test_patient + '/' + slice, target + '/test/' + test_patient + '/' + slice)
            split_and_save_AB(source, target, 2, test_patient, slice)


if __name__ == '__main__':
    # filepath = r"D:\style_transfer_test\AB"
    # dist = r"D:\style_transfer_divided_unpaired"

    filepath = r"/media/NAS02/BraTS2020/pix2pix_cycle/AB"
    dist = r"/media/NAS02/BraTS2020/pix2pix_cycle/style_transfer_divided_unpaired"
    make_dir(filepath, dist)
    divideTrainValiTest(filepath, dist)
