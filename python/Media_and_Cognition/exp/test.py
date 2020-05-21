'''
@Description:
@Author: HuYi
@Date: 2020-05-06 11:59:42
@LastEditors: HuYi
@LastEditTime: 2020-05-18 20:33:58
'''
# Usage: python test.py --predfile pred.json --labelfile test.json

import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--predfile', type=str, default='pred.json')
parser.add_argument('--labelfile', type=str, default='test.json')
args = parser.parse_args()

pred = json.load(open(args.predfile, 'r'))
label = json.load(open(args.labelfile, 'r'))

classes = []
correct = {}
total = {}
for cls in label.values():
    if cls not in classes:
        classes.append(cls)
        correct[cls] = 0
        total[cls] = 0
classes.sort()

miss = 0
cor = 0
for imgname in label.keys():
    try:
        if(pred[imgname] == label[imgname]):
            correct[label[imgname]] += 1
        else:
            print(imgname, pred[imgname], label[imgname])
    except:
        miss += 1
    total[label[imgname]] += 1
acc_str = '%d imgs missed\n' % miss
for cls in classes:
    acc_str += 'class:%s\trecall:%f\n' % (cls, correct[cls]/total[cls])
    cor += correct[cls]
acc_str += 'Accuracy: %f' % (cor/len(label))
print(acc_str)
