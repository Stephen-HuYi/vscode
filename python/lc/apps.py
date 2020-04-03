'''
@Description: 
@Author: HuYi
@Date: 2020-04-02 20:51:36
@LastEditors: HuYi
@LastEditTime: 2020-04-02 21:14:12
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-04-02 19:16
# @Author  : Sunnn
# @File    : apps.py
# @Software: PyCharm
# @intro   :
import csv
import re

import openpyxl
from textrank4zh import TextRank4Keyword, TextRank4Sentence
import pkuseg


def extract_contents(data):
    """ 1. 内容提取"""
    eng_text = " ".join(re.compile("[a-z|\']+", re.I).findall(data))
    cng_text = "".join(re.findall(r'[\u4e00-\u9fa5]', data))

    """ 1. URL提取 """
    en = re.sub('\\\\\(\)（）', '', data)

    url_list = re.findall('http(.*?)tml', en)
    try:
        url_list = ';'.join(url_list)
    except:
        url_list = ''

    return (eng_text, cng_text, url_list)


def eng_participle(data):
    """

    :param data: 英语字符串
    :return: 分词列表与关键词列表
    """
    fenci_list = []
    gjc_list = []
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=data, lower=True, window=2)
    for words in tr4w.words_no_filter:
        # fc_content = '；'.join(words)   # py2中是unicode类型。py3中是str类型。
        fenci_list.append(words)

    for item in tr4w.get_keywords(20, word_min_len=1):
        gjc_list.append(item.word)
    try:
        fc_list = ";".join(fenci_list[0])
    except:
        fc_list = ''
    try:
        gjc_list = ";".join(gjc_list)
    except:
        gjc_list = ''
    return fc_list, gjc_list


def cn__participle(data):
    seg = pkuseg.pkuseg()
    text = seg.cut(data)
    try:
        text = ";".join(text)
    except:
        text = ''

    return text


if __name__ == '__main__':
    wb = openpyxl.load_workbook('.\data\博文内容-科学网.xlsx')
    ws = wb.active
    a = 1
    while True:
        a += 1
        data = ws['A{}'.format(a)].value
        f = extract_contents(data)
        url_list = f[2]
        en_list, en_gjc = eng_participle(f[0])
        cn = cn__participle(f[1])
        print(en_list)
        print(en_gjc)
        print(cn)
        print(url_list)
        dataL = [en_list, en_gjc, cn, url_list, str(data).replace(',', '')]
        with open('fenci.csv', 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(dataL)
