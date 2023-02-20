import os, random, shutil


def make_dir(source, target):
    '''
    创建和源文件相似的文件路径函数
    :param source: 源文件位置
    :param target: 目标文件位置
    '''
    dir_names = os.listdir(source)
    #for names in dir_names:
    for i in ['train', 'val', 'test']:
        path = target + '/' + i
        if not os.path.exists(path):
            os.makedirs(path)


def divideTrainValiTest(source, target):
    '''
        创建和源文件相似的文件路径
        :param source: 源文件位置
        :param target: 目标文件位置
    '''

    # 得到源文件下的种类
    patients = os.listdir(source)
    random.shuffle(patients)
    # print(int(0.8 * len(patients)), int(0.9 * len(patients)), int(0.9 * len(patients)))
    train_list = patients[0:int(0.8 * len(patients))]
    valid_list = patients[int(0.8 * len(patients)):int(0.9 * len(patients))]
    test_list = patients[int(0.9 * len(patients)):]
    print(train_list)
    print(valid_list)
    print(test_list)
    # 对于每个图片，移入到对应的文件夹里面
    for train_patient in train_list:
        if not os.path.exists(target + '/train/' + train_patient):
            os.makedirs(target + '/train/' + train_patient)
        for slice in os.listdir(source + '/' + train_patient):
            # print(slice)
            shutil.copyfile(source + '/' + train_patient + '/' + slice, target + '/train/' + train_patient + '/' + slice)
    for validation_patient in valid_list:
        if not os.path.exists(target + '/val/' + validation_patient):
            os.makedirs(target + '/val/' + validation_patient)
        for slice in os.listdir(source + '/' + validation_patient):
            #print(slice)
            shutil.copyfile(source + '/' + validation_patient + '/' + slice, target + '/val/' + validation_patient + '/' + slice)
    for test_patient in test_list:
        if not os.path.exists(target + '/test/' + test_patient):
            os.makedirs(target + '/test/' + test_patient)
        for slice in os.listdir(source + '/' + test_patient):
            #  print(slice)
            shutil.copyfile(source + '/' + test_patient + '/' + slice, target + '/test/' + test_patient + '/' + slice)


if __name__ == '__main__':
    # filepath = r"D:\style_transfer_test\AB"
    # dist = r"D:\style_transfer_divided"

    filepath = "/media/NAS02/BraTS2020/pix2pix_cycle/AB"
    dist = "/media/NAS02/BraTS2020/pix2pix_cycle/style_transfer_divided"
    make_dir(filepath, dist)
    divideTrainValiTest(filepath, dist)
