# coding=utf-8

import os
import cv2
import utils
import shutil
import random
import numpy as np


def cut_exRect_from_img(img, x1, y1, w, h, exw=5, exh=5):
    x2 = min(img.shape[1], x1 + w + exw)
    y2 = min(img.shape[0], y1 + h + exh)
    x1 = max(0, x1 - exw)
    y1 = max(0, y1 - exh)
    return img[y1:y2, x1:x2]


def cut_txtline_from_front_img(thresh_avg=0.99, thresh_min=0.96, DEBUG=False):
    ''' 
    产生 crnn 训练样本
    百度api, 含位置版本标注好
    从身份证正面读取label,切割成条, 并产生txt

    筛选条件：识别阈值／文本区域长宽比
    '''

    # root_img_dir = '/home/wurui/Desktop/test/img'
    root_img_dir = '/home/wurui/idcard/data/org/img'
    # root_label_dir = '/home/wurui/Desktop/test/label'
    root_label_dir = '/home/wurui/idcard/data/org/labels-git'

    result_img_dir = '/home/wurui/idcard/data/TrainCRNN-prob0.99-ratio4.0/img'
    result_label_dir = '/home/wurui/idcard/data/TrainCRNN-prob0.99-ratio4.0/label'

    # source_parts = ['part5', 'part2', 'part3', 'part4', 'part1']
    source_parts = ['part1']

    ex_names = ['.jpg', '.jpeg', '.JPG', '.JPEG']

    count = 0

    # 登记已经被裁剪过的图
    already_cut_files = []
    for source_part in source_parts:
        label_dir = os.path.join(result_label_dir, source_part)

        if not os.path.exists(label_dir):
            continue

        for label_name in os.listdir(label_dir):
            name = source_part+'/'+label_name[0:-6]
            if name not in set(already_cut_files):
                already_cut_files.append(name)

    # 根据已标注的label裁剪原图
    for source_part in source_parts:
        img_dir = os.path.join(root_img_dir, source_part)
        label_dir = os.path.join(root_label_dir, source_part)

        result_img_part_dir = os.path.join(result_img_dir, source_part)
        result_label_part_dir = os.path.join(result_label_dir, source_part)

        if not os.path.exists(result_img_part_dir):
            os.mkdir(result_img_part_dir)
        if not os.path.exists(result_label_part_dir):
            os.mkdir(result_label_part_dir)

        # 从已标注的label文件中产生样本
        if not os.path.exists(label_dir):
            continue

        for label_name in os.listdir(label_dir):
            # 跳过已经裁剪过的图像
            name = source_part+'/'+label_name[0:-5]
            if name in set(already_cut_files):
                continue

            label_path = os.path.join(label_dir, label_name)
            img_name = [os.path.join(img_dir, label_name.split('.')[0] + ex_name) for ex_name in ex_names if
                        os.path.exists(
                            os.path.join(
                                img_dir, label_name.split('.')[0] + ex_name)
            )][0]
            if not os.path.exists(img_name) or not os.path.exists(label_path):
                continue

            # 读取图像
            img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            if img.shape[0] < 1 or img.shape[1] < 1:
                continue

            # 读取 label
            label_dict = utils.read_json(label_path)
            words_result = label_dict['words_result']

            # 方向
            direction = str(label_dict['direction'])

            if DEBUG:
                print('\n\n file: %s\n direction: %s ' %
                      (label_path, direction))

            for id, item in enumerate(words_result):
                top, w, h, left = item['location']['top'], item['location']['width'], \
                    item['location']['height'], item['location']['left']

                # 文本内容
                content = item['words']

                # 识别概率 筛选
                probability_avg = float(item['probability']['average'])
                probability_min = float(item['probability']['min'])

                # print('avg: %.2f  min: %.2f' %
                #       (probability_avg, probability_min))

                if probability_avg > thresh_avg or (probability_avg > 0.98 and probability_min > 0.96):

                    # 切割出 txtline, 并转正
                    txtline_img = cut_exRect_from_img(
                        img, left, top, w, h, exw=0, exh=0)

                    if direction == '1':
                        txtline_img = cv2.rotate(
                            txtline_img, cv2.ROTATE_90_CLOCKWISE)
                    elif direction == '2':
                        txtline_img = cv2.rotate(
                            txtline_img, cv2.ROTATE_180)
                    elif direction == '3':
                        txtline_img = cv2.rotate(
                            txtline_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    # 图像比例筛选
                    if float(txtline_img.shape[1])/float(txtline_img.shape[0]) < 4.0:
                        continue

                    # 保存裁剪图
                    cv2.imwrite(os.path.join(result_img_part_dir, label_name.split('.')[0] + '_' + str(id) + '.jpg'),
                                txtline_img)

                    # 保存txt内容label
                    with open(os.path.join(result_label_part_dir, label_name.split('.')[0] + '_' + str(id) + '.txt'), 'w',
                              encoding='utf-8') as txtfile:
                        txtfile.write(content)
                        txtfile.close()

                    count += 1
                    print(count)

                    # debug
                    if DEBUG:
                        print('content:', content)
                        cv2.imshow('txtline', txtline_img)
                        cv2.waitKey()


def binarilize(img):
    img = cv2.resize(img, (int(img.shape[1] * 70.0 / img.shape[0]), 70))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, bi = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU |
                          cv2.THRESH_BINARY_INV)

    # bi = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize=9, C=5)
    # bi = cv2.adaptiveThreshold(gray, 255, cv2.CALIB_CB_ADAPTIVE_THRESH, cv2.THRESH_BINARY_INV, blockSize=5, C=10)
    # bi = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=15, C=5)

    return bi


def binary():
    img_dir = '/home/wurui/idcard/data/TrainGanCRNN/img'
    bimg_dir = '/home/wurui/idcard/data/TrainGanCRNN/bimg'

    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if img is None or img.shape[0] < 1 or img.shape[1] < 1:
            os.remove(img_path)
            continue

        bi = binarilize(img)
        bi = cv2.cvtColor(bi, cv2.COLOR_GRAY2BGR)
        # cv2.imshow('bi',bi)
        # cv2.waitKey()

        cv2.imwrite(os.path.join(bimg_dir, img_name), bi)


def crop_resize(img):
    # 1. 直接resize
    # nimg = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)

    # 2. 上下补黑
    # bkg = np.ones((img.shape[1], img.shape[1], 3), dtype=np.uint8)
    # bkg = bkg*255
    # # bkg = cv2.cvtColor(bkg, cv2.COLOR_GRAY2BGR)
    # y1 = int((img.shape[1] - 35) / 2.0) - 1
    # y2 = y1 + 70
    # bkg[y1:y2, :, :] = img
    # # cv2.imshow('bkg', bkg)
    # # cv2.waitKey()
    #
    #
    # 3. 重复拼接
    repeat = 4
    bkg = np.zeros((70 * repeat, img.shape[1], 3), dtype=np.uint8)
    for i in range(repeat):
        y1 = i * 70
        y2 = y1 + 70
        bkg[y1:y2, :, :] = img
    bkg = cv2.resize(bkg, (256, 256))

    # cv2.imshow('bkg', bkg)
    # cv2.waitKey()

    return bkg


def gan_train_data():
    img_dir = '/home/wurui/idcard/data/TrainGanCRNN/img'
    bimg_dir = '/home/wurui/idcard/data/TrainGanCRNN/bimg'
    count = 0

    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        img = cv2.resize(img, (int(img.shape[1] * 70.0 / img.shape[0]), 70))
        img = crop_resize(img)
        cv2.imwrite(
            os.path.join('/home/wurui/idcard/pix2pix-tensorflow-master/photos/resized', img_name[0:-4] + '.png'), img)

    for img_name in os.listdir(bimg_dir):
        img_path = os.path.join(bimg_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        img = crop_resize(img)
        cv2.imwrite(os.path.join('/home/wurui/idcard/pix2pix-tensorflow-master/photos/blank', img_name[0:-4] + '.png'),
                    img)


def resize_256():
    # imgdir = '/home/wurui/idcard/pix2pix-tensorflow-master/photos/resized'
    imgdir = '/home/wurui/idcard/pix2pix-tensorflow-master/photos/blank'
    for im in os.listdir(imgdir):
        img_path = os.path.join(imgdir, im)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(img_path, img)


def convert_label_train_crnn():
    root_img_dir = '/home/wurui/idcard/data/TrainCRNN-prob0.99-ratio4.0/img'
    root_label_dir = '/home/wurui/idcard/data/TrainCRNN-prob0.99-ratio4.0/label'

    # 所有part的图片重新命名移动到一起
    result_img_dir = '/home/wurui/idcard/data/TrainCRNN-prob0.99-ratio4.0/crnn'
    # 制作 lmdb 的文件
    result_train_txt_path = '/home/wurui/idcard/data/TrainCRNN-prob0.99-ratio4.0/train.txt'
    result_test_txt_path = '/home/wurui/idcard/data/TrainCRNN-prob0.99-ratio4.0/test.txt'
    # 中文和数字标签对应文件
    old_label_map_path = '/home/wurui/idcard/data/TrainCRNN-prob0.99-ratio4.0/label_map.txt'
    dict_txt = '/home/wurui/idcard/data/TrainCRNN-prob0.99-ratio4.0/label_dict.txt'

    source_parts = ['part5', 'part2', 'part3', 'part4', 'part1']
    # source_parts = ['part3', 'part5']
    ex_names = ['.jpg', '.jpeg', '.JPG', '.JPEG']

    # 是否全部图像写入train.txt
    IF_ALL = False

    # 测试集比例
    RATIO = 20 / 1

    result_train_txt = open(result_train_txt_path, 'w')
    result_test_txt = open(result_test_txt_path, 'w')

    # 中文--数字 label对应表
    word_map = dict()  # dict = { a:0, b:1, ...}

    with open(old_label_map_path, 'r') as fil:
        lines = fil.readlines()
        for line in lines:
            line = line.strip('\n')
            if line not in word_map.keys():
                word_map[line] = len(word_map.keys())

    # 重新命名图片
    COUNT = 0

    # 标注没有label的图像
    for source_part in source_parts:
        # part1 的　训练测试比例调整
        if source_part == 'part1':
            RATIO = 56/1
        elif source_part == 'part3':
            RATIO = 15/1
        else:
            RATIO = 20/1

        part_label_dir = os.path.join(root_label_dir, source_part)
        part_img_dir = os.path.join(root_img_dir, source_part)

        if not os.path.exists(part_label_dir) or not os.path.exists(part_img_dir):
            continue

        # 对每个 part 中的图像, 随机打乱
        shuffiled_labels = list(os.listdir(part_label_dir))
        random.shuffle(shuffiled_labels)
        for label_file in shuffiled_labels:
            # 检查 label 和 img 都有
            img_path = None
            try:
                img_path = [os.path.join(part_img_dir, label_file.split('.')[0] + ex_name) for ex_name in ex_names if
                            os.path.exists(
                                os.path.join(
                                    part_img_dir, label_file.split('.')[0] + ex_name)
                )][0]

            except:
                pass

            # 如果 label 没有对应的 图片, 就跳过 label 文件
            if img_path is None:
                print('>>>> warning: img not exsist: ', os.path.join(
                    part_label_dir, label_file), '  vs ', os.path.join(part_img_dir))

                # os.remove(os.path.join(part_label_dir, label_file))
                continue

            # 复制图片, 并用英文数字重命名
            new_img_path = os.path.join(result_img_dir, str(COUNT)+'-real.jpg')
            shutil.copy(img_path, new_img_path)

            # 写入对应的汉字标签  转成  数字标签
            label_path = os.path.join(part_label_dir, label_file)

            line = None
            with open(label_path, 'r') as fi:
                line = fi.readline()
                line = line.strip('\n')
                fi.close()

            label = str(os.path.split(new_img_path)[-1])

            for ch in line:
                if ch not in word_map.keys():
                    word_map[ch] = len(word_map)
                label = label + ' ' + str(word_map[ch])

            # 写入 train.txt / test.txt
            if int(COUNT % RATIO) == 0:
                result_test_txt.writelines(label+'\n')

            if IF_ALL:
                result_train_txt.writelines(label+'\n')

            else:
                if int(COUNT % RATIO) != 0:
                    result_train_txt.writelines(label+'\n')

            COUNT += 1
            print(COUNT)

    result_train_txt.close()
    result_test_txt.close()

    # 保存　word_map　中文字符序号对照表
    with open(dict_txt, 'w') as fi:
        word_map = sorted(word_map.items(),
                          key=lambda x: x[1], reverse=False)
        for item in word_map:
            fi.write(item[0]+'\n')


def check_train_test_txt(img_folder, txt_path, result_path, max_num=5000):
    '''
    逐行读取　train.txt　或　test.txt　的图片名，
    删除不在　img_folder　中的行
    '''
    img_names = set(os.listdir(img_folder))

    count = 0

    result_f = open(result_path, 'w')

    with open(txt_path, 'r') as f:
        line = f.readline()
        while True:
            if line is None or line == '':
                break

            name = line.split(' ')[0]
            if name in img_names:
                result_f.write(line)
            count += 1
            if count > max_num:
                break
            line = f.readline()

    result_f.close()


if __name__ == '__main__':
    # 生成ＣＲＮＮ训练数据

    # 1. 从标签文件中把文字行 cut 出来,生成对应的 txt
    # 用原图像名, 分辨跳过已经裁剪过的图像
    # cut_txtline_from_front_img(DEBUG=False)

    # 2. 从 txtline 图像和 txt 标签, 生成 lmdb 原始材料
    # 把图像和txt改名成英文, 生产汉字的对照序列号标注文件train.txt 和 对应的汉字--数字dict.txt
    convert_label_train_crnn()

    # 数据检查
    # img_folder = '/home/wurui/idcard/Synthetic_Chinese_String_Dataset/Dataset/images'
    # txt = '/home/wurui/idcard/Synthetic_Chinese_String_Dataset/Dataset/train-org.txt'
    # rtxt = '/home/wurui/idcard/Synthetic_Chinese_String_Dataset/Dataset/train.txt'
    # check_train_test_txt(img_folder, txt, rtxt)

    # 生成　ＧＡＮ　训练数据
    # gan_train_data()
