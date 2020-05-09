
import os
import random
import shutil

'''
    随机拆分数据集为训练集和测试集：按照8:2比例拆分
'''
def splitData2TrainAndTest(base_path, train_path, test_path):
    if os.path.exists(train_path) is False:
        os.makedirs(train_path)
    if os.path.exists(test_path) is False:
        os.makedirs(test_path)

    for person_type in os.listdir(base_path):
        type_dir = os.path.join(base_path, person_type)
        os.makedirs(os.path.join(train_path, person_type))    # 创建训练集和测试集的目录结构
        os.makedirs(os.path.join(test_path, person_type))

        for file in os.listdir(type_dir):
            if random.randint(0, 9) < 8:
                shutil.copy(os.path.join(type_dir, file), os.path.join(train_path, person_type))
            else:
                shutil.copy(os.path.join(type_dir, file), os.path.join(test_path, person_type))
    print("测试集分割完成")

def rename_jpg():
    base_dir = "./training_set/"

    for second_dir in os.listdir(base_dir):
        full_dir = os.path.join(base_dir, second_dir)
        count = 1
        for file in os.listdir(full_dir):
            print(file)
            old_fullname = os.path.join(full_dir, file)
            new_filename = second_dir + "-" + str(count) + ".jpg"
            count = count + 1
            print(os.path.join(full_dir, new_filename))
            os.rename(old_fullname, os.path.join(full_dir, new_filename))
    print("批量rename完成")

'''
    统计目录下文件个数
'''
def statistics_file_nums(path, endfix='.jpg'):
    file_nums = 0
    for person_type in os.listdir(path):
        full_dir = os.path.join(path, person_type)
        for file in os.listdir(full_dir):
            if file.endswith(endfix):
                file_nums = file_nums + 1
    return file_nums



if __name__ == '__main__':
    base_path = "F:/workspace/dataset/UTKFace/"
    train_path = "./UTKFace/train/"
    test_path = "./UTKFace/test/"

    splitData2TrainAndTest(base_path, train_path, test_path)    # 拆分训练集和测试集
    print("train images:", statistics_file_nums(train_path))
    print("test images:", statistics_file_nums(test_path))