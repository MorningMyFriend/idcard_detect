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
        img = cv2.resize(img, (int(img.shape[1] * 960 / img.shape[0]), 960), img)
    else:
        img = cv2.resize(img, (1080, int(img.shape[0] * 1080 / img.shape[1])), img)
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
                image = cv2.imread(os.path.join(part_img_dir, fil), cv2.IMREAD_COLOR)
                # cv2.imshow('', image)
                # cv2.waitKey()
                cv2.imwrite(os.path.join(part_img_dir, fil.split('.')[0] + '.jpg'), image)
            else:
                os.rename(os.path.join(part_img_dir, fil), os.path.join(part_img_dir, fil.split('.')[0] + '.jpg'))