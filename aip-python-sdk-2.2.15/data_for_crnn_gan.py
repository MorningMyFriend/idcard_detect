# coding=utf-8

import os
import cv2
import utils
import numpy as np


def cut_exRect_from_img(img, x1, y1, w, h, exw=5, exh=5):
    x2 = min(img.shape[1], x1 + w + exw)
    y2 = min(img.shape[0], y1 + h + exh)
    x1 = max(0, x1 - exw)
    y1 = max(0, y1 - exh)
    return img[y1:y2, x1:x2]


def cut_txtline_from_front_img():
    # 从身份证正面读取label,切割成条, 并产生txt
    root_img_dir = '/home/wurui/idcard/data/normal-front-0dree/img'
    root_label_dir = '/home/wurui/idcard/data/normal-front-0dree/label'

    result_img_dir = '/home/wurui/idcard/data/TrainGanCRNN/img'
    result_label_dir = '/home/wurui/idcard/data/TrainGanCRNN/label'

    source_parts = ['part5', 'part2', 'part3', 'part4', 'part1']
    source_parts = ['part3']
    ex_names = ['.jpg', '.jpeg', '.JPG', '.JPEG']

    items = ['姓名', '性别', '民族', '出生', '住址', '公民身份号码']

    for source_part in source_parts:
        img_dir = os.path.join(root_img_dir, source_part)
        label_dir = os.path.join(root_label_dir, source_part)

        result_img_part_dir = os.path.join(result_img_dir, source_part)
        result_label_part_dir = os.path.join(result_label_dir, source_part)

        if not os.path.exists(result_img_part_dir):
            os.mkdir(result_img_part_dir)
        if not os.path.exists(result_label_part_dir):
            os.mkdir(result_label_part_dir)

        for label_name in os.listdir(label_dir):
            label_path = os.path.join(label_dir, label_name)
            img_name = [os.path.join(img_dir, label_name.split('.')[0] + ex_name) for ex_name in ex_names if
                        os.path.exists(
                            os.path.join(img_dir, label_name.split('.')[0] + ex_name)
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

            for id, item in enumerate(items):
                if item not in words_result.keys():
                    continue

                top, w, h, left = words_result[item]['location']['top'], words_result[item]['location']['width'], \
                                  words_result[item]['location']['height'], words_result[item]['location']['left']

                content = words_result[item]['words']

                # 切割出 txtline
                txtline_img = cut_exRect_from_img(img, left, top, w, h, exw=5, exh=5)
                cv2.imwrite(os.path.join(result_img_part_dir, label_name.split('.')[0] + '_' + str(id) + '.jpg'),
                            txtline_img)

                # 保存txt内容label
                with open(os.path.join(result_label_part_dir, label_name.split('.')[0] + '_' + str(id) + '.txt'), 'w',
                          encoding='utf-8') as txtfile:
                    txtfile.write(content)
                    txtfile.close()


def binarilize(img):
    img = cv2.resize(img, (int(img.shape[1] * 70.0 / img.shape[0]), 70))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, bi = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

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


def test():
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


def resize():
    # imgdir = '/home/wurui/idcard/pix2pix-tensorflow-master/photos/resized'
    imgdir = '/home/wurui/idcard/pix2pix-tensorflow-master/photos/blank'
    for im in os.listdir(imgdir):
        img_path = os.path.join(imgdir, im)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(img_path, img)


if __name__ == '__main__':
    # cut_txtline_from_front_img()
    # binary()
    test()
    # resize()
