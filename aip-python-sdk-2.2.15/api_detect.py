# coding=utf-8

import os
import sys
import json
import base64
import cv2
import shutil

# 保证兼容python2以及python3
IS_PY3 = sys.version_info.major == 3
if IS_PY3:
    from urllib.request import urlopen
    from urllib.request import Request
    from urllib.error import URLError
    from urllib.parse import urlencode
    from urllib.parse import quote_plus
else:
    import urllib2
    from urllib import quote_plus
    from urllib2 import urlopen
    from urllib2 import Request
    from urllib2 import URLError
    from urllib import urlencode

# 防止https证书校验不正确
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

API_KEY = 'i037oS7fcc2dFVy6GC2Gjxkz'
SECRET_KEY = 'GCkxU2xKsw4NeAEsjxgxoEFa4S3mT1NL'

# API_KEY = 'k6FgI8fSqPPYFt2lSGBFCbYE'
# SECRET_KEY = 'BElpKbfUwN8pv1TM3aEhofKUEljbugH4'

# API_KEY = 'KTgCUQ2ngP7XTr3inawvPzlh'
# SECRET_KEY = 'gSPGZ0OAUniG3nZHfEhGbWtEjT8CGNNG'

ACCESS = [['i037oS7fcc2dFVy6GC2Gjxkz', 'GCkxU2xKsw4NeAEsjxgxoEFa4S3mT1NL'],
          ['k6FgI8fSqPPYFt2lSGBFCbYE', 'BElpKbfUwN8pv1TM3aEhofKUEljbugH4'],
          ['KTgCUQ2ngP7XTr3inawvPzlh', 'gSPGZ0OAUniG3nZHfEhGbWtEjT8CGNNG'],
          ['Lnxl1KB4dw3m3ckGfqej3lb6', 'RTfWN9XaEN1hjv59uexEFoOQRVacwY7V'],
          ['dhZlc1Mk2dBcMExlR4F8p8xg', 'UHy6MUz0GCFG0qAC57G9QvHkrjzGnCws']
          ]

OCR_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/idcard"  # 身份证url
# OCR_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"


"""  TOKEN start """
TOKEN_URL = 'https://aip.baidubce.com/oauth/2.0/token'

"""
    获取token
"""


def fetch_token(api_key, secrect_key):
    params = {'grant_type': 'client_credentials',
              'client_id': api_key,
              'client_secret': secrect_key}
    post_data = urlencode(params)
    if (IS_PY3):
        post_data = post_data.encode('utf-8')
    req = Request(TOKEN_URL, post_data)
    result_str = None
    try:
        f = urlopen(req, timeout=5)
        result_str = f.read()
    except URLError as err:
        print(err)
    if (IS_PY3):
        result_str = result_str.decode()

    result = json.loads(result_str)

    if ('access_token' in result.keys() and 'scope' in result.keys()):
        if not 'brain_all_scope' in result['scope'].split(' '):
            print('please ensure has check the  ability')
            exit()
        return result['access_token']
    else:
        print('please overwrite the correct API_KEY and SECRET_KEY')
        exit()


"""
    读取文件
"""


def read_file(image_path):
    f = None
    try:
        f = open(image_path, 'rb')
        return f.read()
    except:
        print('read image file fail')
        return None
    finally:
        if f:
            f.close()


"""
    调用远程服务
"""


def request(url, data):
    req = Request(url, data.encode('utf-8'))
    has_error = False
    try:
        f = urlopen(req)
        result_str = f.read()
        if (IS_PY3):
            result_str = result_str.decode()
        return result_str
    except  URLError as err:
        print(err)


def demo():
    # 获取access token
    token = fetch_token()

    # 拼接通用文字识别高精度url
    image_url = OCR_URL + "?access_token=" + token

    text = ""

    # 读取书籍页面图片
    file_content = read_file('data/丁力_身份证正面照片.jpeg')

    # 调用文字识别服务
    result = request(image_url, urlencode({'image': base64.b64encode(file_content)}))

    # 解析返回结果
    result_json = json.loads(result)
    for words_result in result_json["words_result"]:
        text = text + words_result["words"]

    # 打印文字
    print(text)


def read_img(img_path):
    if os.path.exists(img_path):
        # 二进制方式打开图文件
        f = open(img_path, 'rb')
        # 参数image：图像base64编码
        img = base64.b64encode(f.read())
        return img
    else:
        print('img_path:{%s} not exist'.format(img_path))
        exit(1)


def detect_idcard(img_path, api_key, secrect_key):
    # 获取access token
    token = fetch_token(api_key, secrect_key)

    # 拼接通用文字识别高精度url
    image_url = OCR_URL + "?access_token=" + token

    # 参数image：图像base64编码
    image = read_img(img_path)
    params = {"image": image, "id_card_side": "front", "detect_direction": "true"}
    params = urlencode(params)
    request = Request(image_url, params.encode('utf-8'))
    request.add_header('Content-Type', 'application/x-www-form-urlencoded')
    response = urlopen(request)
    content = response.read()
    if (content):
        # print(content)
        return content
    else:
        return None


def idcard_detect_demo():
    # 使用百度api标注
    # source_dir = '/home/wurui/Desktop/img'
    root_img_dir = '/home/wurui/idcard/data/org/img'
    root_label_dir = '/home/wurui/idcard/data/org/label'
    source_parts = ['part5','part2','part3','part4', 'part1']
    # source_parts = ['part3']

    labeled_count = 0

    # 每一个账号能调用500次
    for api_key, secrect_key in ACCESS:

        # 标注没有label的图像
        for source_part in source_parts:
            part_img_dir = os.path.join(root_img_dir, source_part)
            part_label_dir = os.path.join(root_label_dir, source_part)

            # 已标注的文件名
            labeled_files = [x.split('.')[0] for x in os.listdir(part_label_dir)]
            labeled_count = len(labeled_files)

            for img_file in os.listdir(part_img_dir):
                if img_file.split('.')[0] not in labeled_files:

                    # 调用api
                    content = detect_idcard(os.path.join(part_img_dir, img_file), api_key, secrect_key)
                    result_str = str(content, 'utf-8')
                    result_dict = json.loads(result_str)
                    if 'error_msg' in result_dict.keys():
                        print('error_msg from baidu.api', result_dict['error_msg'])
                        break

                    save_json(result_dict, os.path.join(part_label_dir, img_file.split('.')[0] + '.json'))
                    labeled_count += 1
                    print('part: {}     labeled count: {}'.format(source_part, labeled_count))


def detect_imgdir(source_dir):
    for img in os.listdir(source_dir):
        content = detect_idcard(os.path.join(source_dir, img))
        result_str = str(content, 'utf-8')
        result_json_dict = json.loads(result_str)
        with open(os.path.join(source_dir, img.split('.')[0] + '.json'), "w") as f:
            json.dump(result_json_dict, f)
        print(result_json_dict)


def save_json(dic, save_path):
    with open(save_path, "w") as f:
        json.dump(dic, f)


def read_json(json_path):
    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
        return load_dict
    return None


def check_result_comlete(json_dict):
    items = ['姓名', '性别', '民族', '出生', '住址', '公民身份号码']
    if 'direction' not in json_dict.keys():
        return False
    if 'image_status' not in json_dict.keys():
        return False
    if 'words_result' not in json_dict.keys():
        return False

    words_result = json_dict['words_result']
    for item in items:
        if item not in words_result.keys():
            return False
    return True


def check_vision():
    # 查看原始标注label是否准确
    root_img_dir = '/home/wurui/idcard/data/org/img'
    root_label_dir = '/home/wurui/idcard/data/org/label'
    source_parts = ['part2', 'part3', 'part4', 'part5', 'part1']
    ex_names = ['.jpg', '.jpeg', '.JPG', '.JPEG']

    remove_difficult_count = 0

    items = ['姓名', '性别', '民族', '出生', '住址', '公民身份号码']

    # 检测已有label的图像
    for source_part in source_parts:
        part_img_dir = os.path.join(root_img_dir, source_part)
        part_label_dir = os.path.join(root_label_dir, source_part)

        # 已标注的文件名
        for fil in os.listdir(part_label_dir):

            img_name = [os.path.join(part_img_dir, fil.split('.')[0] + ex_name) for ex_name in ex_names if
                        os.path.exists(
                            os.path.join(part_img_dir, fil.split('.')[0] + ex_name)
                        )][0]

            label_name = os.path.join(part_label_dir, fil)
            label_dic = read_json(label_name)
            # print(label_dic)

            image = cv2.imread(img_name, cv2.IMREAD_COLOR)
            # print('\n\n', img_name)
            words_result = label_dic['words_result']
            for item in items:
                # 打印检测结果
                if item not in words_result.keys():
                    continue
                # print(item, words_result[item]['words'])
                top, w, h, left = words_result[item]['location']['top'], words_result[item]['location']['width'], \
                                  words_result[item]['location']['height'], words_result[item]['location']['left']
                cv2.rectangle(image, (left, top), (left + w, top + h), (0, 255, 0), 2)

            # 打印图像检测质量状态
            image_status = label_dic['image_status']
            # print('image_status: ', image_status)

            # 图像转正
            if 'direction' not in label_dic.keys():
                print('Error: no direction info !!!!', label_name)
                exit(1)

            image_direction = label_dic['direction']

            # print('direction: ', image_direction)
            if image_direction == '1':
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE, image)
            elif image_direction == '2':
                image = cv2.rotate(image, cv2.ROTATE_180, image)
            elif image_direction == '3':
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE, image)

            # 图像检测不过关的,移动到difficult文件夹
            if image_status != 'normal':
                print('difficult img: ', img_name)
                difficult_img_path = os.path.join(root_img_dir, 'difficult', os.path.split(img_name)[-1])
                print('move to path:', difficult_img_path)
                # shutil.move(img_name, difficult_img_path)
                print('remove label: ', label_name)
                # os.remove(label_name)
                image = resizeImgToScreen(image)
                remove_difficult_count += 1
                print('remove difficult count:', remove_difficult_count)
                # cv2.imshow('', image)
                # cv2.waitKey()


def resizeImgToScreen(img):
    # 缩放到适合屏幕显示
    if img.shape[0] > img.shape[1]:
        img = cv2.resize(img, (int(img.shape[1] * 960 / img.shape[0]), 960), img)
    else:
        img = cv2.resize(img, (1080, int(img.shape[0] * 1080 / img.shape[1])), img)
    return img


def file_convert_to_jpg():
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


def remove_error_label():
    # 删除检测返回error的label
    root_label_dir = '/home/wurui/idcard/data/org/label'
    source_parts = ['part2', 'part3', 'part4', 'part5', 'part1']

    count = 0

    for source_part in source_parts:
        for fil in os.listdir(os.path.join(root_label_dir, source_part)):
            label_path = os.path.join(root_label_dir, source_part, fil)
            label_dic = read_json(label_path)

            if 'error_msg' in label_dic.keys():
                os.remove(label_path)
                print('delete count:', count)
                count += 1


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


def rotate_location_label_dict(label_dict, items, image, flag):
    # 旋转后修改label的location
    words_result = label_dict['words_result']

    for item in items:
        if item not in words_result.keys():
            continue

        top, w, h, left = words_result[item]['location']['top'], words_result[item]['location']['width'], \
                          words_result[item]['location']['height'], words_result[item]['location']['left']

        xmin, ymin, xmax, ymax = rotate_coordinates(left, top, left + w, top + h,
                                                    image.shape[0], image.shape[1], flag)

        label_dict['words_result'][item]['location']['top'] = ymin
        label_dict['words_result'][item]['location']['left'] = xmin
        label_dict['words_result'][item]['location']['height'] = ymax - ymin
        label_dict['words_result'][item]['location']['width'] = xmax - xmin

        label_dict['direction'] = '0'

    return label_dict


def pick_normal_label_and_rotate_to_front():
    # 挑选 检测结果normal的图像, 转至正面, 修改label

    root_img_dir = '/home/wurui/idcard/data/org/img'
    root_label_dir = '/home/wurui/idcard/data/org/label'
    normal_img_dir = '/home/wurui/idcard/data/normal-front-0dree/img'
    normal_label_dir = '/home/wurui/idcard/data/normal-front-0dree/label'
    source_parts = ['part5', 'part2', 'part3', 'part4', 'part1']
    source_parts = ['part3']
    ex_names = ['.jpg', '.jpeg', '.JPG', '.JPEG']

    count = 0

    items = ['姓名', '性别', '民族', '出生', '住址', '公民身份号码']

    # 已经有label的
    for source_part in source_parts:
        part_img_dir = os.path.join(root_img_dir, source_part)
        part_label_dir = os.path.join(root_label_dir, source_part)

        # 已标注的文件名
        for fil in os.listdir(part_label_dir):
            # 图像名
            img_name = [os.path.join(part_img_dir, fil.split('.')[0] + ex_name) for ex_name in ex_names if
                        os.path.exists(
                            os.path.join(part_img_dir, fil.split('.')[0] + ex_name)
                        )][0]
            image = cv2.imread(img_name, cv2.IMREAD_COLOR)

            # 标签名
            label_name = os.path.join(part_label_dir, fil)
            label_dic = read_json(label_name)

            # 目标文件名
            normal_img_path = os.path.join(normal_img_dir, source_part, os.path.split(img_name)[-1])
            normal_label_path = os.path.join(normal_label_dir, source_part, os.path.split(label_name)[-1])

            if os.path.exists(normal_label_path) and os.path.exists(normal_label_path):
                count+=1
                continue

            # 跳过检测结果不好的图像
            if 'image_status' not in label_dic.keys() or label_dic['image_status'] != 'normal':
                continue

            # 读取旋转角度
            if 'direction' not in label_dic.keys():
                continue

            image_direction = int(label_dic['direction'])
            print('org direction:', image_direction)

            # 旋转图像 & 修改label坐标
            if image_direction == 1:
                label_dic = rotate_location_label_dict(label_dic, items, image, 'clockwise90')
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif image_direction == 2:
                label_dic = rotate_location_label_dict(label_dic, items, image, 'clockwise180')
                image = cv2.rotate(image, cv2.ROTATE_180)
            elif image_direction == 3:
                label_dic = rotate_location_label_dict(label_dic, items, image, 'clockwise270')
                cv2.imshow(' old', image)
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imshow(' rotate', image)

            # 保存新图像新label
            # print(normal_img_path, normal_label_path)
            cv2.imwrite(normal_img_path, image)
            save_json(label_dic, normal_label_path)
            count += 1
            print('count:', count)

            # 检测新label是否正确
            new_img = cv2.imread(normal_img_path)
            new_dic = read_json(normal_label_path)

            words_result = new_dic['words_result']
            for item in items:
                # 打印检测结果
                if item not in words_result.keys():
                    continue
                # print(item, words_result[item]['words'])
                top, w, h, left = words_result[item]['location']['top'], words_result[item]['location']['width'], \
                                  words_result[item]['location']['height'], words_result[item]['location']['left']
                cv2.rectangle(new_img, (left, top), (left + w, top + h), (0, 255, 0), 2)
            new_img = resizeImgToScreen(new_img)
            cv2.imshow('new', new_img)
            cv2.waitKey()


            # break
        # break


def conver_img_file_to_jpg():
    img_dir =   '/home/wurui/idcard/data/org/img/part5'
    for img in os.listdir(img_dir):
        image = cv2.imread(os.path.join(img_dir, img), cv2.IMREAD_COLOR)
        new_img = os.path.join(img_dir, img.split('.')[0]+'.jpg')
        print(new_img)
        cv2.imwrite(new_img, image)


if __name__ == '__main__':
    # conver_img_file_to_jpg()
    idcard_detect_demo()
    remove_error_label()
    # check_vision()
    pick_normal_label_and_rotate_to_front()
