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
import pkuseg
from textrank4zh import TextRank4Keyword, TextRank4Sentence


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
    pattern = re.compile("[\u4e00-\u9fa5]")
    data = "".join(pattern.findall(data))
    text = seg.cut(data)

    countDict = dict()
    proportitionDict = dict()

    for i in set(text):
        countDict[i] = text.count(i)
        proportitionDict[i] = text.count(i) / len(text)

    try:
        gjc_list = []
        count = 0
        for i in sorted(proportitionDict):
            if count >= 6:
                break
            if len(i) >= 2 and i not in ["都", "全", "单", "共", "光", "尽", "净", "仅", "就", "只", "一共", "一起", "一同", "一道", "一齐",
                                         "一概", "一味", "统统", "总共", "仅仅", "惟独", "可", "倒", "一定", "必定", "必然", "却", "", "就",
                                         "幸亏", "难道", "何尝", "偏偏", "索性", "简直", "反正", "多亏", "也许", "大约", "好在", "敢情", "不",
                                         "没", "没有", "别", "刚", "恰好", "正", "将", "老（是）", "总（是）", "早就", "已经", "正在", "立刻",
                                         "马上", "起初", "原先", "一向", "永远", "从来", "偶尔", "随时", "忽然", "很", "极", "最", "太", "更",
                                         "更加", "格外", "十分", "极其", "比较", "相当", "稍微", "略微", "多么", "仿佛", "渐渐", "百般", "特地",
                                         "互相", "擅自", "几乎", "逐渐", "逐步", "猛然", "依然", "仍然", "当然", "毅然", "果然", "差点"]:
                gjc_list.append(i)
                count += 1
        gjc = ";".join(gjc_list)
    except:
        gjc = ''

    try:
        text = ";".join(text)
    except:
        text = ''

    return text, gjc


if __name__ == '__main__':
    wb = openpyxl.load_workbook('.\data\博文内容-科学网.xlsx')
    ws = wb.active
    a = 1
    while True:
        a += 1
        data = ws['A{}'.format(a)].value
        cn__participle(data)

        f = extract_contents(data)
        url_list = f[2]
        en_list, en_gjc = eng_participle(f[0])
        cn, cn_gjc = cn__participle(f[1])
        print(en_list)
        print(en_gjc)
        print(cn)
        print(url_list)
        dataL = [en_list, en_gjc, cn, cn_gjc,
                 url_list, str(data).replace(',', '')]
        with open('fenci1.csv', 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(dataL)
