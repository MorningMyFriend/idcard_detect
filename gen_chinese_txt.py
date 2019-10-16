# coding=utf-8
import os
import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np


def gen_txt_img(content, shape=(320, 720, 3)):
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
    txt = '我爱我的祖国'
    gen_txt_img(txt)


if __name__ == '__main__':
    main()
