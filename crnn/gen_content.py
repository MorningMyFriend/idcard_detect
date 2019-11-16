# coding=utf-8

import os
import random
import numpy as np


class GenContent:
    def __init__(self):
        pass

    def gen_name(self):
        # 随机生成姓名
        name_txt = '姓  名  王大锤'
        return name_txt

    def gen_nation(self):
        # 性别, 民族
        nation_txt = '性 别 男   民 族 汉'
        return nation_txt

    def gen_birthday(self):
        # 随机生成出生年月
        birthday_txt = '出 生:1995年3月15日'

        return birthday_txt

    def gen_location(self):
        # 随机生成地址
        location_txt = '住址: 上海市杨浦区四平路1239号同济大学7号楼1092'
        return location_txt

    def gen_id(self):
        # 身份证号
        id_txt = '230108199415041843'
