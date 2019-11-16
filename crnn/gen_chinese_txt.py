# coding=utf-8
from gen_content import GenContent
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2
import os


class ImgGen:
    def __init__(self, content_txt):
        self.content_txt = content_txt
        self.font_source_folder = '/home/wurui/idcard/data/SythicData/fonts'
        self.bkg_source_folder = '/home/wurui/idcard/data/SythicData/bkg'
        self.noise_source_folder = '/home/wurui/idcard/data/SythicData/noise'
        self.result_img_folder = '/home/wurui/idcard/data/SythicData/img'

    def getFont(self, result_num=1, font_size=32):
        '''
        @source_folder: 字体文件夹
        @font_num: 返回多少个字体
        @ return: ImageFont.truetype   font obj
        how to use: --> gen_txt_img()
        '''
        source_folder = self.font_source_folder
        fonts = list(os.listdir(source_folder))

        result_fonts = list()

        # 随机选取 result_num 个字体
        font_ids = np.random.randint(low=0, high=len(
            fonts), size=min(result_num, len(fonts)))
        for font_id in font_ids:
            font_name = fonts[font_id]
            font_path = os.path.join(source_folder, font_name)
            result_fonts.append(ImageFont.truetype(
                font=font_path, size=font_size, encoding='utf-8'))
        return result_fonts

    def getBkgNoiseImg(self, source_folder, result_num=1, imgShape=None):

        # 随机获取 result_num 个背景图 或者 前景噪声图
        files = list(os.listdir(source_folder))

        results = list()

        indexs = np.random.randint(low=0, high=len(
            files), size=min(result_num, len(files)))
        for id in indexs:
            name = files[id]
            file_path = os.path.join(source_folder, name)
            bkg_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if bkg_img is None:
                continue
            if imgShape is not None:
                bkg_img = cv2.resize(bkg_img, (imgShape[1], imgShape[0]))

            results.append(bkg_img)

        return results

    def legal_rect_xy(self, img, xmin, xmax, ymin, ymax):
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img.shape[1]-1, xmax)
        ymax = min(img.shape[0]-1, ymax)
        return xmin, xmax, ymin, ymax

    def __shrink_txt_area__(self, img, ex_w=2, ex_h=2, DEBUG=True):
        '''
        输入白底黑字图, 切割成贴合的图
        '''
        binary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        xmin = 0
        for i in range(binary.shape[1]):
            # 每一列 黑色像素 数量>0, 则返回xmin
            if np.sum(binary[:, i] == 0) > 0:
                xmin = i
                break

        xmax = binary.shape[1]-1
        for i in range(binary.shape[1]-1, -1, -1):
            if np.sum(binary[:, i] == 0) > 0:
                xmax = i
                break

        ymin = 0
        for i in range(binary.shape[0]):
            if np.sum(binary[i, :] == 0) > 0:
                ymin = i
                break
        ymax = binary.shape[0]-1
        for i in range(binary.shape[0]-1, -1, -1):
            if np.sum(binary[i, :] == 0) > 0:
                ymax = i
                break

        xmin, xmax, ymin, ymax = self.legal_rect_xy(
            img, xmin-ex_w, xmax+ex_w, ymin-ex_h, ymax+ex_h)

        return img[ymin:ymax, xmin:xmax]

    def paste_mask_to_bkg(self, bkg, mask):
        # 1. 先把mask贴到bkg上
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        ret, mask_bi = cv2.threshold(
            mask_gray, 175, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask_bi)

        # ｍask　白色区域　保留, bkg　白色区域　清除
        mask_roi = cv2.bitwise_and(mask, mask, mask=mask_inv)
        bkg_roi = cv2.bitwise_and(bkg, bkg, mask=mask_bi)

        #　合并　mask, bkg
        merge = cv2.add(mask_roi, bkg_roi)

        # 2. 把　bkg　和带字的图片　alpha　通道融合
        alpha_merge = cv2.addWeighted(merge, 0.65, bkg, 0.35, 0)

        return alpha_merge

    def gen_txt_line(self, DEBUG=True):
        # 随机生成字体
        font = self.getFont(result_num=1, font_size=32)
        if len(font) < 1:
            print('error: font None')
            return None

        # 生产文字图, 贴合文字边缘剪裁, 白底黑字
        bkg = np.ones(shape=(100, 750, 3), dtype=np.uint8) * 255
        img_pil = Image.fromarray(bkg)
        draw = ImageDraw.Draw(img_pil)
        draw.text((5, 5), self.content_txt,
                  font=font[0], fill=(0, 0, 0, 0))
        img = np.array(img_pil)

        # 切割成贴合的图片
        txt_img = self.__shrink_txt_area__(img, ex_w=4, ex_h=4)

        # if DEBUG:
        #     cv2.imshow('txt', txt_img)
        # cv2.waitKey()

        # 背景图
        bkg = self.getBkgNoiseImg(
            source_folder=self.bkg_source_folder, result_num=1, imgShape=txt_img.shape)
        if len(bkg) < 1:
            print('error: bkg img None')
            return None
        bkg = bkg[0]

        # if DEBUG:
        #     cv2.imshow('bkg', bkg)
        # cv2.waitKey()

        # 贴图融合
        txt_img = self.paste_mask_to_bkg(bkg, txt_img)

        if DEBUG:
            cv2.imshow('txt_img', txt_img)
            cv2.waitKey()

        return txt_img

    def generate_single_line(self):
        # 生成 文字行 图
        txt_img = self.gen_txt_line(DEBUG=True)


def gen_txt_img(content, shape=(100, 750, 3)):
    bkg = np.ones(shape=shape, dtype=np.uint8)
    bkg = bkg*255

    # 字体
    fontPath = "/usr/share/fonts/opentype/noto/NotoSansCJK-DemiLight.ttc"
    font = ImageFont.truetype(font=fontPath, size=32, encoding='utf-8')

    img_pil = Image.fromarray(bkg)
    draw = ImageDraw.Draw(img_pil)
    color = (0, 0, 0, 0)
    draw.text((5, 5), content, font=font, fill=color)
    img = np.array(img_pil)

    cv2.imshow(',', img)
    cv2.waitKey()


def main():
    txt = '我爱我的祖国 14好 104动 上海市杨浦区'
    # gen_txt_img(txt)

    ImgGener = ImgGen(txt)
    ImgGener.generate_single_line()


if __name__ == '__main__':

    main()
