# coding=utf-8
import ssl
import os
import sys
import json
import base64
import cv2
import shutil
import time

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

ssl._create_default_https_context = ssl._create_unverified_context

ACCESS = [
    ['kCdABVu26hU8Cq8Dy9YhelmP', 'M5gBftIvz8dN8Mhxh8mVaWKKsccR68Q9'],
    ['zXNGo333MUnAiVKlhYollhAI', 'caOFbuo7MbAZPtRIbSNGunGRQNjj696T'],
    ['k6FgI8fSqPPYFt2lSGBFCbYE', 'BElpKbfUwN8pv1TM3aEhofKUEljbugH4'],
    ['KTgCUQ2ngP7XTr3inawvPzlh', 'gSPGZ0OAUniG3nZHfEhGbWtEjT8CGNNG'],
    ['Lnxl1KB4dw3m3ckGfqej3lb6', 'RTfWN9XaEN1hjv59uexEFoOQRVacwY7V'],
    ['dhZlc1Mk2dBcMExlR4F8p8xg', 'UHy6MUz0GCFG0qAC57G9QvHkrjzGnCws'],
    ['Be1sqL1mQFgVl2dCtIyUNlfY', 'ovPoh950C6363vw0sy5cYSF8RByVPlYz'],
    ['gCsVEjEYYU3v5oxB5ax6hOuQ', 'P5DGwF8vvpwpdkWNcrBtjrMCiRrSZvFK'],
    ['091CGibCRl1vMLcDSCLA804k', '5hnAsOatOoZRI6BWHVGUZo4Dl7zcVeSH'],
    ['dGriMDIayDOew3Gqb6AF42zf', 'SzhHVPFjlO4HnmBLwjFhC74W4rRxWOzf'],
    ['KigIVEYDBc3Ytnd0KUtAMXvF', '3hPD5xOCFfVziKUqmxUGPIGBdDgBZCuN'],
    ['D38PFe7eLzXDDvDtDiLUYB4v', 'jReMtbEyM1x5B1jBEHMfA5GksxGjVheB'],
    ['NYmo1uIg6dmUlAGfBEl4Trfa', '9wnRGawclG6eXsMY8ecjdmDGwvTZD0a7'],
    ['3fS8EYWdDnth1FcjT8P1sFkg', '6Ix1cEcabNQKySPZjVzdlRD9ByZedfqg'],
    ['jhXYFBCrtNCa1bTxbrv27ZoH', 'rlGZNEb7GCGOwvCatqwzNNwDyxZNOGCd'],
    ['W7kKol6jCgpaDjww0BZg6NB2', 'bmZ1xAb5Yv3BiA7bfBwQRlVlCE32veAD'],
    ['srrj7Uhr7Yz2V6lm0PLS3DdU', 'vYekOoeUEy2OrejALb5lP0mPukF6rC4r'],
    ['yYas4BLER76LNDzHYy9uS63R', 'ulaNYqzNPEqzS9rFeq3ZSUGgvUiroezo'],
    ['fvi8ijsVrGB52lxvikWki392', '2G7Ok49SdTYBvePr1RQ7d4OnOAvqr3PF'],
    ['qlhqMQo14a1OSmB1Z3yR1d2x', 'HEF5USShKkFIeMMfIctz2qvAvqbgIRbj'],
    ['eDPxnqHxil6W2AvApi4jibsk', 'TagGP4pawG2GKNAezgFvvSxAGbaRQxgc'],
    ['gsnOIiQ58zttUlAiiVfebKLE', 'QYj0Bl3msFpGP411dIdXhZb7l2CAN8Mq'],
    ['Bn0LQwb8Uc2Dzgi8SHmH9Tyo', 'ZVgdOzvWOl4TNNMsi8d6zePchsYwLBdd']
]

OCR_URL = {
    # 基础版本
    'general_basic': "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic",  # 50000/day

    # 身份证url
    'idcard':  "https://aip.baidubce.com/rest/2.0/ocr/v1/idcard",  # 5000/day

    # 含位置版本
    'general': 'https://aip.baidubce.com/rest/2.0/ocr/v1/general',  # 5000/day

    # 高精度版本
    'accurate_basic': "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic",  # 500/day

    # 高精度含位置版
    'accurate': "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate"  # 50/day
}


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
        f = urlopen(req, timeout=15)
        result_str = f.read()
    except URLError as err:
        print(err)
        return None

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
    except URLError as err:
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
    result = request(image_url, urlencode(
        {'image': base64.b64encode(file_content)}))

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


def detect_idcard_json(img_path, access_que):
    '''
    @access_que: [[api_key, secrect_key, flag], [],...]
    @return: access_que, json_dict
    没 key 了就 exit(),
    其他错误 continue
    '''
    # 获取有效的 key
    if access_que is None:
        return None, None
    if len(access_que) < 2:
        print(access_que)
    if len(access_que) < 1:
        return None, None

    api_key, secrect_key, flag = access_que[0]
    content = detect_idcard(img_path, api_key, secrect_key, flag=flag)
    if content is None:
        return access_que, None

    result_str = str(content, 'utf-8')
    result_dict = json.loads(result_str)

    if 'error_msg' in result_dict.keys():
        print('error_msg from baidu.api: ',
              result_dict['error_msg'], '    error_code: ', result_dict['error_code'])

        if int(result_dict['error_code']) == 17 or int(result_dict['error_code']) == 18:
            # api daily limit
            access_que.pop(0)
            return detect_idcard_json(img_path, access_que)
        elif result_dict['error_msg'] == ' empty image' or int(result_dict['error_code']) == 216200:
            print('empty img_path: ', img_path)
        else:
            # empty image and so on
            return access_que, None

    return access_que, result_dict


def detect_idcard(img_path, api_key, secrect_key, flag='general'):
    # 获取access token
    token = fetch_token(api_key, secrect_key)

    if token is None:
        return None

    # 拼接通用文字识别高精度url
    image_url = OCR_URL[flag] + "?access_token=" + token

    # 参数image：图像base64编码
    image = read_img(img_path)
    params = {
        "image": image,
        "id_card_side": "front",
        "detect_direction": "true",
        "probability": "true"
    }
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


def idcard_detect_demo(DEBUG=True):
    # 使用百度api标注
    root_img_dir = '/home/wurui/idcard/data/org/img'

    root_label_dir = '/home/wurui/idcard/data/org/labels-git'

    source_parts = ['part5', 'part2', 'part3', 'part4', 'part1']

    labeled_count = 0

    DETECT_LIMIT = False

    # 每一个账号调用次数有限制
    api_keys_que = list()
    for api_key, secrect_key in ACCESS:
        for flag in ['accurate', 'general']:
            api_keys_que.append([api_key, secrect_key, flag])

    api_key, secrect_key, flag = api_keys_que[0]

    # 标注没有label的图像
    for source_part in source_parts:
        part_img_dir = os.path.join(root_img_dir, source_part)
        part_label_dir = os.path.join(root_label_dir, source_part)

        if not os.path.exists(part_label_dir):
            os.mkdir(part_label_dir)

        if api_keys_que is None or len(api_keys_que) < 1:
            break

        # 已标注的文件名
        labeled_files = [x.split('.')[0]
                         for x in os.listdir(part_label_dir)]
        labeled_count = len(labeled_files)

        for img_file in os.listdir(part_img_dir):
            if img_file.split('.')[0] in labeled_files:
                continue

            # 对没标注的图片  调用api
            api_keys_que, result_dict = detect_idcard_json(os.path.join(
                part_img_dir, img_file), api_keys_que)

            if result_dict is None:
                continue

            if api_keys_que is None or len(api_keys_que) < 1:
                break

            print('\ntarget file: ', os.path.join(
                part_img_dir, img_file))
            print('----> keys num: ', len(api_keys_que))
            if len(api_keys_que) > 0:
                print('---->key: ', api_keys_que[0])

            save_json(result_dict, os.path.join(
                part_label_dir, img_file.split('.')[0] + '.json'))
            labeled_count += 1
            print('part: {}     labeled count: {}'.format(
                source_part, labeled_count))


def detect_imgdir(source_dir, flag='general'):
    for img in os.listdir(source_dir):
        content = detect_idcard(os.path.join(source_dir, img), flag=flag)
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


def check_vision(flag='general'):
    '''
    @flag: idcard, normal, location
    '''

    # 查看原始标注label是否准确

    root_img_dir = '/home/wurui/idcard/data/org/img'

    root_label_dir = '/home/wurui/idcard/data/org/labels-git'
    source_parts = ['part5']
    # source_parts = ['part2', 'part3', 'part4', 'part5', 'part1']

    ex_names = ['.jpg', '.jpeg', '.JPG', '.JPEG']

    remove_difficult_count = 0

    items = ['姓名', '性别', '民族', '出生', '住址', '公民身份号码']

    # 检测已有label的图像
    for source_part in source_parts:
        part_img_dir = os.path.join(root_img_dir, source_part)
        part_label_dir = os.path.join(root_label_dir, source_part)

        if not os.path.exists(part_label_dir):
            continue

        # 已标注的文件名
        for fil in os.listdir(part_label_dir):
            # 对应的图像是否存在
            img_name = None
            try:
                img_name = [os.path.join(part_img_dir, fil.split('.')[0] + ex_name) for ex_name in ex_names if
                            os.path.exists(
                                os.path.join(
                                    part_img_dir, fil.split('.')[0] + ex_name))][0]

            except:
                pass

            if img_name is None:
                continue

            label_name = os.path.join(part_label_dir, fil)
            label_dic = read_json(label_name)
            print(label_dic, '\n\n')

            image = cv2.imread(img_name, cv2.IMREAD_COLOR)

            words_result = label_dic['words_result']

            if flag == 'idcard':
                for item in items:
                    # 打印检测结果
                    if item not in words_result.keys():
                        continue
                    # print(item, words_result[item]['words'])
                    top, w, h, left = words_result[item]['location']['top'], words_result[item]['location']['width'], \
                        words_result[item]['location']['height'], words_result[item]['location']['left']
                    cv2.rectangle(image, (left, top),
                                  (left + w, top + h), (0, 255, 0), 2)

                    # 打印图像检测质量状态
                    image_status = label_dic['image_status']
                    # 图像检测不过关的,移动到difficult文件夹
                    if image_status != 'normal':
                        print('difficult img: ', img_name)
                        difficult_img_path = os.path.join(
                            root_img_dir, 'difficult', os.path.split(img_name)[-1])
                        print('move to path:', difficult_img_path)
                        # shutil.move(img_name, difficult_img_path)
                        print('remove label: ', label_name)
                        # os.remove(label_name)
                        remove_difficult_count += 1
                        print('remove difficult count:',
                              remove_difficult_count)

            elif flag == 'general':
                for item in words_result:
                    top, w, h, left = item['location']['top'], item['location']['width'], \
                        item['location']['height'], item['location']['left']
                    cv2.rectangle(image, (left, top),
                                  (left + w, top + h), (0, 255, 0), 2)

            # 图像转正
            if 'direction' not in label_dic.keys():
                print('Error: no direction info !!!!', label_name)
                exit(1)

            image_direction = label_dic['direction']

            # print('direction: ', image_direction)
            if image_direction == '1':
                image = cv2.rotate(
                    image, cv2.ROTATE_90_COUNTERCLOCKWISE, image)
            elif image_direction == '2':
                image = cv2.rotate(image, cv2.ROTATE_180, image)
            elif image_direction == '3':
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE, image)

            image = resizeImgToScreen(image)

            cv2.imshow('', image)
            cv2.waitKey()


def resizeImgToScreen(img):
    # 缩放到适合屏幕显示
    if img.shape[0] > img.shape[1]:
        img = cv2.resize(
            img, (int(img.shape[1] * 960 / img.shape[0]), 960), img)
    else:
        img = cv2.resize(
            img, (1080, int(img.shape[0] * 1080 / img.shape[1])), img)
    return img


def file_convert_to_jpg():
    # 图片格式统一到jpeg
    root_img_dir = '/home/wurui/idcard/data/org/img'
    root_label_dir = '/home/wurui/idcard/data/org/labels-git'
    source_parts = ['part5',  'part2', 'part3', 'part4',  'part1']

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


def remove_error_label():
    # 删除检测返回error的label
    root_label_dir = '/home/wurui/idcard/data/org/labels-git'
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
    root_label_dir = '/home/wurui/idcard/data/org/labels-git'
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
            # 对应的图像是否存在
            img_name = None
            try:
                img_name = [os.path.join(part_img_dir, fil.split('.')[0] + ex_name) for ex_name in ex_names if
                            os.path.exists(
                                os.path.join(
                                    part_img_dir, fil.split('.')[0] + ex_name)
                )][0]

            except:
                pass

            if img_name is None:
                continue

            image = cv2.imread(img_name, cv2.IMREAD_COLOR)

            # 标签名
            label_name = os.path.join(part_label_dir, fil)
            label_dic = read_json(label_name)

            # 目标文件名
            normal_img_path = os.path.join(
                normal_img_dir, source_part, os.path.split(img_name)[-1])
            normal_label_path = os.path.join(
                normal_label_dir, source_part, os.path.split(label_name)[-1])

            if os.path.exists(normal_label_path) and os.path.exists(normal_label_path):
                count += 1
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
                label_dic = rotate_location_label_dict(
                    label_dic, items, image, 'clockwise90')
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif image_direction == 2:
                label_dic = rotate_location_label_dict(
                    label_dic, items, image, 'clockwise180')
                image = cv2.rotate(image, cv2.ROTATE_180)
            elif image_direction == 3:
                label_dic = rotate_location_label_dict(
                    label_dic, items, image, 'clockwise270')
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
                cv2.rectangle(new_img, (left, top),
                              (left + w, top + h), (0, 255, 0), 2)
            new_img = resizeImgToScreen(new_img)
            cv2.imshow('new', new_img)
            cv2.waitKey()

            # break
        # break


def conver_img_file_to_jpg():
    img_dir = '/home/wurui/idcard/data/org/img/part5'
    for img in os.listdir(img_dir):
        image = cv2.imread(os.path.join(img_dir, img), cv2.IMREAD_COLOR)
        new_img = os.path.join(img_dir, img.split('.')[0]+'.jpg')
        print(new_img)
        cv2.imwrite(new_img, image)


def remove_empty_img():
    # 删除检测返回error的label
    root_label_dir = '/home/wurui/idcard/data/org/img'
    # already checked: 'part2', 'part3', 'part4',  'part1'
    source_parts = ['part5']

    count = 0

    for source_part in source_parts:
        num = 0
        for fil in os.listdir(os.path.join(root_label_dir, source_part)):
            # print('part: %s   num:%d   empty count:%d' %
            #       (source_part, num, count))
            num += 1
            img_path = os.path.join(root_label_dir, source_part, fil)
            img = cv2.imread(img_path)

            if img is None or img.shape[0] < 1 or img.shape[1] < 1:
                print(img_path)
                count += 1
                os.remove(img_path)
    print('empty count: ', count)


if __name__ == '__main__':
    # 剔除空图片
    # remove_empty_img()

    # conver_img_file_to_jpg()
    # while True:
    #     print('fuck one more time')
    #     idcard_detect_demo()

    #     # 剔除 label 中含 error msg的标签
    #     remove_error_label()
    #     time.sleep(60)

    check_vision(flag='general')
    # pick_normal_label_and_rotate_to_front()
