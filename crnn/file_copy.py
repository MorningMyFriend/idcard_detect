# coding=utf-8
import os
import shutil
import random
import cv2


def copy_files():
    source_dir = '/home/wurui/idcard/data/TrainCRNN-prob0.99-ratio4.0/aug-img'
    target_dir = '/home/wurui/Downloads/rar/images'
    COUNT = 0
    for fil in os.listdir(source_dir):
        shutil.copy(os.path.join(source_dir, fil),
                    os.path.join(target_dir, fil))
        COUNT += 1
        print(COUNT)


def merge_txt():
    txt_files = [
        '/home/wurui/idcard/data/TrainCRNN-prob0.99-ratio4.0/train-real.txt',
        '/home/wurui/idcard/data/TrainCRNN-prob0.99-ratio4.0/aug-train.txt'
    ]

    result_txt_path = '/home/wurui/idcard/data/TrainCRNN-prob0.99-ratio4.0/train.txt'

    with open(result_txt_path, 'w') as outf:
        for txt_fil in txt_files:
            infile = open(txt_fil, 'r')
            line = infile.readline()

            while True:
                if line is None or line == '':
                    break
                outf.write(line)
                line = infile.readline()

            infile.close()


def mix_txt():
    # 每个batch里混合　真实／合成数据，　train.txt要关闭shuffle
    real_txt_path = '/home/wurui/idcard/data/TrainCRNN-prob0.99-ratio4.0/train-real-aug.txt'
    arti_txt_path = '/home/wurui/Downloads/Synthetic Chinese String Dataset/train.txt'
    result_txt_path = '/home/wurui/idcard/data/TrainCRNN-prob0.99-ratio4.0/train.txt'

    # 先对 真实数据的　train.txt　随机打乱
    real_txt_fil = open(real_txt_path, 'r')
    real_lines = real_txt_fil.readlines()
    real_txt_fil.close()

    random.seed(1234)
    random.shuffle(real_lines)

    # 每个batch　按照比例　列出　真实／混合数据 = 2:1
    outf = open(result_txt_path, 'w')
    artif = open(arti_txt_path, 'r')
    arti_lines = artif.readlines()
    artif.close()

    count, i, j = 0, 0, 0
    while i < len(real_lines) and j < len(arti_lines):
        if count % 3 == 0:
            outf.write(real_lines[i])
            i += 1
        else:
            outf.write(arti_lines[j])
            j += 1

        count += 1
        print(count)

    while i < len(real_lines):
        outf.write(real_lines[i])
        i += 1
        count += 1
        print(count)

    while j < len(arti_lines):
        outf.write(arti_lines[j])
        j += 1
        count += 1
        print(count)

    outf.close()


def select_test():
    train_txt = '/home/wurui/idcard/data/TrainCRNN-prob0.99-ratio4.0/train.txt'
    test_txt = '/home/wurui/idcard/data/TrainCRNN-prob0.99-ratio4.0/test.txt'

    test_fil = open(test_txt, 'w')

    train_fil = open(train_txt, 'r')
    line = train_fil.readline()
    count = 0
    while True:
        if line is None or line == '':
            break
        if count % 5 == 0:
            test_fil.write(line)

        line = train_fil.readline()
        count += 1


if __name__ == '__main__':
    # copy_files()
    # select_test()
    # merge_txt()
    # mix_txt()
    

    img = cv2.imread(
        '/home/wurui/Downloads/rar/images/20456343_4045240981.jpg')
    cv2.imshow('', img)
    cv2.waitKey()
