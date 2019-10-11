# coding=utf-8

import os
import sys
import json
import shutil

def save_json(dic, save_path):
    with open(save_path, "w") as f:
        json.dump(dic, f)


def read_json(json_path):
    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
        return load_dict
    return None

def eval_front():
    # 评估正面检测的准确率
    # part2 + part4(1853) 共3230张图  100%正确率
    result_root_dir = '/home/wurui/idcard/data/front-result/area-100000-0.7'
    label_root_dir = '/home/wurui/idcard/data/org/label'
    source_parts = ['part2', 'part3', 'part4', 'part5', 'part1']

    eval_count = 0
    wrong_count = 0

    for source_part in source_parts:
        for result_file in os.listdir(os.path.join(result_root_dir, source_part)):
            result_path = os.path.join(result_root_dir, source_part, result_file)
            label_path = os.path.join(label_root_dir,source_part, result_file)
            if not os.path.exists(label_path):
                continue

            label_dict = read_json(label_path)
            result_dict = read_json(result_path)

            if 'direction' not in label_dict.keys() or 'image_status' not in label_dict.keys():
                continue
            if label_dict['image_status'] != 'normal':
                continue


            eval_count += 1
            print('count: ', eval_count, '  wrong:', wrong_count, '  acc:', 1 - float(wrong_count) / float(eval_count))

            if int(label_dict['direction']) == 0 and int(result_dict['class'])== 0:
                continue
            elif int(label_dict['direction']) == 1 and int(result_dict['class'])== 270:
                continue
            elif int(label_dict['direction']) == 2 and int(result_dict['class'])== 180:
                continue
            elif int(label_dict['direction']) == 3 and int(result_dict['class'])== 90:
                continue
            else:
                wrong_count+=1

def test_json():
        result_dict = {}
    rects = [{'a':1, 'b':2}, {'a':4, 'b':5}  ]
    result_dict = {'rects':rects}

    save_json(result_dict,'testarr.json')

    json_dict = read_json('testarr.json')
    for rect in json_dict['rects']:
        print(rect)



if __name__ == '__main__':
    test_json()
    # eval_front()







