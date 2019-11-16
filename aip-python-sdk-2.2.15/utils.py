# coding=utf-8

import os
import sys
import json
import shutil

import cv2


def save_json(dic, save_path):
    with open(save_path, "w") as f:
        json.dump(dic, f)


def read_json(json_path):
    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
        return load_dict
    return None


def rotate_coordinates(xmin, ymin, xmax, ymax, imgH, imgW, flag):
    # 计算旋转后的坐标
    if flag == 'clockwise90':
        xmin, ymin = imgH - ymin, xmin
        xmax, ymax = imgH - ymax, xmax
    elif flag == 'clockwise180':
        xmin, ymin = imgW - xmin, imgH - ymin
        xmax, ymax = imgW - xmax, imgH - ymax
    elif flag == 'clockwise270':
        xmin, ymin = ymin, imgW - xmin
        xmax, ymax = ymax, imgW - xmax

    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin

    return xmin, ymin, xmax, ymax


def resizeImgToScreen(img):
    # 缩放到适合屏幕显示
    if img.shape[0] > img.shape[1]:
        img = cv2.resize(
            img, (int(img.shape[1] * 960 / img.shape[0]), 960), img)
    else:
        img = cv2.resize(
            img, (1080, int(img.shape[0] * 1080 / img.shape[1])), img)
    return img


def convert_img_file_to_jpg():
    # 图片格式统一到jpeg
    root_img_dir = '/home/wurui/idcard/data/org/img'
    root_label_dir = '/home/wurui/idcard/data/org/label'
    source_parts = ['part2', 'part3', 'part4', 'part5', 'part1']

    for source_part in source_parts:
        part_img_dir = os.path.join(root_img_dir, source_part)
        for fil in os.listdir(part_img_dir):
            if fil.split('.')[1] not in ['jpg', 'JPG', 'jpeg', 'JPEG']:
                image = cv2.imread(os.path.join(
                    part_img_dir, fil), cv2.IMREAD_COLOR)
                # cv2.imshow('', image)
                # cv2.waitKey()
                cv2.imwrite(os.path.join(
                    part_img_dir, fil.split('.')[0] + '.jpg'), image)
            else:
                os.rename(os.path.join(part_img_dir, fil), os.path.join(
                    part_img_dir, fil.split('.')[0] + '.jpg'))


def legal_rect_xy(img, xmin, xmax, ymin, ymax):
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img.shape[1]-1, xmax)
    ymax = min(img.shape[0]-1, ymax)
    return xmin, xmax, ymin, ymax


def convert_imgs(img_root_folder, target='.jpg'):
    for name in os.listdir(img_root_folder):
        img_path = os.path.join(img_root_folder, name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        new_path = os.path.join(img_root_folder, name.split('.')[0]+target)
        cv2.imwrite(new_path, img)


def delete＿files():
    target_dir = '/home/wurui/idcard/data/org/labels-git/part2'
    refer_dir = '/home/wurui/idcard/data/front-result/area-100000-0.7/img/part2'
    for fil in os.listdir(refer_dir):
        json_name = os.path.join(target_dir, fil.split('.')[0]+'.json')
        if os.path.exists(json_name):
            print(json_name)
            os.remove(json_name)


def merge_img():
    # 检测重复命名
    root_img_dir = '/home/wurui/idcard/data/org/img'
    root_label_dir = '/home/wurui/idcard/data/org/labels-git'

    source_parts = ['part1', 'part2', 'part3', 'part4', 'part5']

    names = list()
    count = 0
    dup_count = 0

    for source_part in source_parts:
        part_img_dir = os.path.join(root_img_dir, source_part)
        part_label_dir = os.path.join(root_label_dir, source_part)

        for fil in os.listdir(part_img_dir):

            name = fil[0:-4]
            if name not in set(names):
                names.append(name)
            else:
                print(name)
                dup_count += 1

            count += 1

            print('count:%d   dup:%d' % (count, dup_count))


if __name__ == '__main__':

    # source = '/home/wurui/idcard/data/org/img/part3'
    # convert_imgs(source, '.jpg')

    merge_img()

    # delete＿files()
