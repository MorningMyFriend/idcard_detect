# coding=utf-8
from __future__ import print_function
from __future__ import division

import os
import sys
import cv2
import numpy as np
import utils


def match(target_img, template_img,  algorithm='FLANN', feature='SIFT', DEBUG=True):
    '''
    input: 两张图像, 输出透视转换矩阵
    @algorithom: 'BF', 'FLANN'
    @feature: 'ORB', 'SIFT', 'SURF'
    ->return: transform mat
    '''

    kp1, des1, kp2, des2 = [None]*4
    raw_matches = None

    if feature == 'SIFT':
        # 计算特征点和距离
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(target_img, None)
        kp2, des2 = sift.detectAndCompute(template_img, None)
    elif feature == 'SURF':
        surf = cv2.xfeatures2d.SURF_create()
        kp1, des1 = surf.detectAndCompute(target_img, None)
        kp2, des2 = surf.detectAndCompute(template_img, None)
    elif feature == 'ORB':
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(target_img, None)
        kp2, des2 = orb.detectAndCompute(template_img, None)

    if algorithm == 'BF':
        # 其中normType是用来指定要使用的距离测试类型。
        # 默认值为cv2.Norm_L2,适用于SIFT,SURF方法，还有一个参数为cv2.Norm_L1。
        # 如果是ORB,BRIEF,BRISK算法等，要是用cv2.NORM_HAMMING，
        # 如果ORB算法的参数设置为VTA_K==3或4，normType就应该设置为cv2.NORM_HAMMING2
        # 第二个参数是crossCheck，默认值是False。
        # 如果设置为True，匹配条件会更加严格。
        # 如果A图像中的i点和B图像中的j点距离最近，并且B中的j点到A中的i点距离也最近，相互匹配，这个匹配结果才会返回
        # bf = cv2.BFMatcher(normType=cv2.Norm_L2, crossCheck=True)
        bf = cv2.BFMatcher()
        raw_matches = bf.knnMatch(des1, des2, k=2)

    elif algorithm == 'FLANN':
        # 最邻近方法近似匹配
        # SearchParams只包含一个字段，checks，表示制定索引树要被遍历的次数。经验值推荐，5 kd—trees，50 checks 可以取得较好的匹配精度
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, tree=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        # 比率测试: KNNMatch，可设置K = 2 ，即对每个匹配返回两个最近邻描述符，仅当第一个匹配与第二个匹配之间的距离足够小时，才认为这是一个匹配
        raw_matches = flann.knnMatch(des1, des2, k=2)

    good = []

    # 距离筛选: 删除距离大于0.7可以剔除90%的误匹配, 论文:<Distinctive Image Feature from scale Scale-Invariant Keypoints>
    for m, n in raw_matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    # 高于最低匹配结果数量, 则计算变换矩阵
    MIN_MATCH_COUNT = 10
    MIN_MASK_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                     5.0)  # 第四个参数取值范围在 1 到 10 , 绝一个点对的阈值。原图像的点经过变换后点与目标图像上对应点的误差

        matchesMask = mask.ravel().tolist()

        if matchesMask.count(1) < MIN_MASK_COUNT:
            M = None
            print("Not enough matches in mask when cv2.findHomography: %d < %d" % (
                matchesMask.count(1), MIN_MASK_COUNT))

    else:
        print("Not enough matches are found - %d/%d" %
              (len(good), MIN_MATCH_COUNT))
        M, matchesMask = None, None

    if DEBUG:
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=(0, 0, 255),
                           matchesMask=matchesMask,
                           flags=2)  # draw only inliers

        vis = cv2.drawMatches(target_img, kp1, template_img,
                              kp2, good, None, **draw_params)
        vis = utils.resizeImgToScreen(vis)
        cv2.imshow("", vis)

        cv2.waitKey()
        cv2.destroyAllWindows()

    return M


def transform(img, mat, template_size, DEBUG=True):
    wrap = cv2.warpPerspective(img, mat, template_size, flags=cv2.INTER_CUBIC)

    if DEBUG:
        img = utils.resizeImgToScreen(img)
        cv2.imshow("org", img)
        wrap_show = utils.resizeImgToScreen(wrap)
        cv2.imshow("wrap", wrap_show)
        cv2.waitKey()

    return wrap


def eg():
    img_dir = '/home/wurui/Desktop/ID/img1'
    template_img_path = '/home/wurui/Desktop/ID/male_temp.jpg'

    template_img = cv2.imread(template_img_path, cv2.IMREAD_COLOR)
    template_img = cv2.resize(template_img, dsize=(int(
        template_img.shape[1]/2), int(template_img.shape[0]/2)), interpolation=cv2.INTER_CUBIC)

    for img_path in os.listdir(img_dir):
        img = cv2.imread(os.path.join(img_dir, img_path), cv2.IMREAD_COLOR)
        transform_mat = match(
            img, template_img,  algorithm='FLANN', feature='SIFT', DEBUG=True)
        if transform_mat is None:
            print('match failed: transform mat is None')
            continue
        transform(img, transform_mat, template_size=(
            template_img.shape[1], template_img.shape[0]))


if __name__ == '__main__':
    eg()
