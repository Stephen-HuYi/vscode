import cv2
import os
import numpy as np
import json
import random
def load_images(path,labels):
    data = {}
    for label in labels:
        tmp = []
        for filename in os.listdir(path+label+'/'):
            img = cv2.imread(path+label+'/'+filename, cv2.IMREAD_GRAYSCALE)
            tmp.append(cv2.resize(img,(28,28)))
        data[label] = tmp
    return data

def hog_compute(data,hog):
    data_hog = {}
    winStride = (8, 8)
    padding = (8, 8)
    for key in data:
        tmp = data[key]
        tmp_hog = []
        for img in tmp:
            feature = hog.compute(img,winStride,padding)
            tmp_hog.append(feature)
            #print(np.shape(feature))
        data_hog[key] = tmp_hog
    return data_hog


if __name__ == '__main__':
    train_root = '/home/ass02/Datasets/image_exp/Classification/Data/Train/'
    test_root = '/home/ass02/Datasets/image_exp/Classification/Data/Test/'
    train_annotation = json.load(open('/home/ass02/Datasets/image_exp/Classification/Data/train.json'))
    labels = []
    for img_name in train_annotation:
        if train_annotation[img_name] not in labels:
            labels.append(train_annotation[img_name])
    train_data = load_images(train_root,labels=labels)
    winSize = (28,28)
    blockSize = (14,14)
    blockStride = (7,7)
    cellSize = (7,7)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    train_hog = hog_compute(train_data,hog)
    # todo: create SVM classifier
    print('1')
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(1)
    xtrain = []
    xvalid = []
    ytrain = []
    yvalid = []
    for key in train_hog:
        tmp = train_hog[key]
        random.shuffle(tmp)
        for i,feature in enumerate(tmp):
            if i<len(tmp)*0.8:
                ytrain.append([labels.index(key)])
                xtrain.append(feature)
            else:
                yvalid.append([labels.index(key)])
                xvalid.append(feature)
    print('2')
    xtrain = np.squeeze(np.array(xtrain))
    ytrain = np.array(ytrain)
    xvalid = np.squeeze(np.array(xvalid))
    yvalid = np.array(yvalid)
    print(np.shape(xtrain))
    print(np.shape(ytrain))
    svm.train(xtrain,cv2.ml.ROW_SAMPLE, ytrain)
    svm.save('svm_data.dat')
    ret, result = svm.predict(xvalid)
    final = [yvalid.tolist(), result.tolist()]
    mask = result == yvalid
    correct = np.count_nonzero(mask)
    print(correct * 100.0 / len(result), '%')
    ttl = np.zeros((len(labels),1))
    predict = np.zeros((len(labels),1))
    yvalid = np.squeeze(yvalid)
    result = np.squeeze(result).astype(np.int32)
    for i,p in enumerate(result):
        ttl[yvalid[i]] += 1
        if yvalid[i] == p:
            predict[yvalid[i]] += 1
    prob = predict/ttl
    for i,acc in enumerate(prob):
        print('{}:{}/{}={}'.format(labels[i],predict[i],ttl[i],acc))
    json.dump(final,open('./result.txt','w',encoding='utf8'))

