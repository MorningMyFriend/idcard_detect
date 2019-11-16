# coding=utf-8

import os
import random
import numpy as np


def find_lcseque(s1, s2):
    # 最长公共子序列
    # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
    m = [[0 for x in range(len(s2)+1)] for y in range(len(s1)+1)]
    # d用来记录转移方向
    d = [[None for x in range(len(s2)+1)] for y in range(len(s1)+1)]

    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:  # 字符匹配成功，则该位置的值为左上方的值加1
                m[p1+1][p2+1] = m[p1][p2]+1
                # d[p1+1][p2+1] = 'ok'
            elif m[p1+1][p2] > m[p1][p2+1]:  # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
                m[p1+1][p2+1] = m[p1+1][p2]
                # d[p1+1][p2+1] = 'left'
            else:  # 上值大于左值，则该位置的值为上值，并标记方向up
                m[p1+1][p2+1] = m[p1][p2+1]
                # d[p1+1][p2+1] = 'up'
    # (p1, p2) = (len(s1), len(s2))
    # print(np.array(d))

    # s = []
    # while m[p1][p2]:  # 不为None时
    #     c = d[p1][p2]
    #     if c == 'ok':  # 匹配成功，插入该字符，并向左上角找下一个
    #         s.append(s1[p1-1])
    #         p1 -= 1
    #         p2 -= 1
    #     if c == 'left':  # 根据标记，向左找下一个
    #         p2 -= 1
    #     if c == 'up':  # 根据标记，向上找下一个
    #         p1 -= 1
    # s.reverse()
    # return ''.join(s)

    return int(m[-1][-1])


def acc_crnn():
    # crnn 检测结果的精度统计
    label_root_dir = '/home/wurui/idcard/data/TrainCRNN-prob0.99-ratio4.0/label/part3'
    result_txt = '/home/wurui/idcard/caffe_ocr_for_linux/test_result.txt'

    LOSS1_COUNT = 0
    LOSS1_SUM_PRE = 0
    LOSS1_SUM_REC = 0

    LOSS2_COUNT = 0
    LOSS2_SUM = 0

    with open(result_txt) as result_fil:
        lines = result_fil.readlines()
        i = 0
        while i < len(lines):
            img_path = lines[i].strip('\n')
            result = lines[i+1].strip('\n')
            i += 2

            # 读取标签
            label_name = os.path.split(img_path)[-1][0:-4]+'.txt'
            label_name = os.path.join(label_root_dir, label_name)
            if not os.path.exists(label_name):
                continue

            fil = open(label_name)
            label = fil.readline().strip('\n')
            fil.close()

            # 精度评定
            LOSS1_COUNT += find_lcseque(label, result)
            LOSS1_SUM_PRE += len(result)
            LOSS1_SUM_REC += len(label)

            LOSS2_COUNT += int(label == result)
            LOSS2_SUM += 1

    precision = float(LOSS1_COUNT)/float(LOSS1_SUM_PRE)
    recall = float(LOSS1_COUNT)/float(LOSS1_SUM_REC)
    fmeasure = 0.5 * (precision+recall)/precision*recall
    print('precision: {0:.4}   recall: {1:.4}   accuracy: {2:.4}   f1-measure: {3:.4}'.format(
        precision, recall, float(LOSS2_COUNT)/float(LOSS2_SUM), fmeasure))


def train_dataset_statistic():
    # 训练数据　均衡性统计
    label_root_dir = '/home/wurui/idcard/data/TrainCRNN-prob0.99-ratio4.0/label'
    source_parts = ['part5', 'part2', 'part3', 'part4', 'part1']
    # source_parts = ['part3']
    label_dict_txt = '/home/wurui/idcard/data/TrainCRNN-prob0.99-ratio4.0/label_dict.txt'

    result_txt = '/home/wurui/idcard/data/TrainCRNN-prob0.99-ratio4.0/statistic.txt'

    count_dict = dict()
    with open(label_dict_txt) as fil:
        lines = fil.readlines()
        for line in lines:
            line = line.strip('\n')
            count_dict[line] = 0

    for source_part in source_parts:
        label_part_dir = os.path.join(label_root_dir, source_part)
        for label_file in os.listdir(label_part_dir):
            fil = open(os.path.join(label_part_dir, label_file), 'r')
            line = fil.readline().strip('\n')
            for ch in line:
                if ch not in count_dict.keys():
                    print('error!!! new key: %s  for  label dict:%s' %
                          (ch, os.path.join(label_part_dir, label_file)))
                    exit(0)
                count_dict[ch] += 1

    with open(result_txt, 'w') as fil:
        count_dict = sorted(count_dict.items(),
                            key=lambda x: x[1], reverse=True)

        for item in count_dict:
            fil.write(item[0]+' '+str(item[1])+'\n')


def main():
    # acc_crnn()
    train_dataset_statistic()


if __name__ == '__main__':
    main()
