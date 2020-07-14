import os
import csv
import pickle as pk
import time
import numpy as np
import sys
sys.path.append('..')
import time
import argparse
from entities.collaboration import collaboration
from data_cleaning import *

def get_index(colla_list,author_ids):
    author_index=[]
    temp_start=0
    j=0
    for i in range(len(colla_list)):
        if(colla_list[i][0]!=str(author_ids[j])):
            j=j+1
            temp_end=i-1
            author_index.append([temp_start,temp_end])
            temp_start=i
    temp_end=len(colla_list)-1
    author_index.append([temp_start, temp_end])
    return author_index




if __name__ == '__main__':

    csv_file=open('sociology_author_collaborations.csv')    #打开文件：输入标准：x id;x_aff_id;y_id;y_aff_id,year,paper_id_list,flag
    csv_reader_lines = csv.reader(csv_file)    #用csv.reader读文件
    data_PyList=[]

    for one_line in csv_reader_lines:
        data_PyList.append(one_line)

    colla_list=data_PyList[1:len(data_PyList)]
    new_colla_list=delete_one_time_collab(colla_list)     #去除只有一次合作的所有links
    colla_list=new_colla_list
    Collaboration=collaboration()    #初始化
    Collaboration.year_aff_generate(colla_list)

    #construct data to check the code
    # colla_list = [['11', '1','22','2', '2000'], ['11','1','33', '3', '2000'], ['11','1','22', '2', '2001'],['11','3','22','2','2001'], ['11','1', '33','3', '2003'], ['11','1', '22','2', '2009'],['11','1', '33','3', '2015'],
    #              ['22','2','44', '4', '2000'], ['22','2','33', '3', '2003'], ['22','2','33', '3', '2009'],['22','2','55','5','2016']]

    #data cleaning_1 , author_x!=author_y
    colla_list= [i for i in colla_list if not(i[0]==i[2])]


    # get author_ids
    temp_author_ids=[]
    for i in colla_list:
        temp_author_ids.append(int(i[0]))
    author_ids=list(set(temp_author_ids))
    author_ids.sort()

    print('first step is ok!')

    #average collaboration num
    ratio=len(colla_list)/len(author_ids)
    print(ratio)

    for i in range(len(colla_list)):
        colla_list[i][4]=int(colla_list[i][4])

    author_index = get_index(colla_list, author_ids)


    #add author imformation
    num=0
    temp_colla_list=[]
    new_colla_list=[]
    for author_id in author_ids:
        part_colla_list=colla_list[author_index[num][0]:author_index[num][1]+1]
        part_colla_list = delete_part_colla(part_colla_list)
        if len(part_colla_list)!=0:
            temp_colla_list.append(part_colla_list)
            Collaboration.add_author_collaboration(part_colla_list,str(author_id))
            for collab in part_colla_list:
                new_colla_list.append(collab)
        #else:
            #author_ids.remove(author_id)
            # print(author_id)
        num=num+1
        if num % 1000 == 0:
            print(num,  ',', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    colla_list=new_colla_list
    author_ids=list(Collaboration.authorId_b_e.keys())
    print('second step is ok!')
    #add overall imformation
    colla_list.sort(key=lambda t: t[4])    #按照时间排序
    Collaboration.add_overall_collaboration(colla_list)

    print('thrid step is ok!')
    #update into Zihao's method
    flag=1
    Collaboration.update_collaboration(flag,Collaboration.author_ids)

    print('fourth step is ok!')
    #calculate conditional prob
    Collaboration.conditional_prob(Collaboration.author_ids,flag)
    print('fifth step is ok!')
    # add information of aff_collab
    Collaboration.add_aff_collaboration()
    #plot
    Collaboration.collabration_plot(Collaboration.author_ids)


    #save
    path='C:/Users/Messilk/PycharmProjects/MAG/collaboration_entity.pkl'
    with open(path, 'wb') as f:
                pk.dump(Collaboration, f)

    #f = open(path,'rb+')
    #info = pk.load(f)