#definition of the collaboration entity in MAG data
import operator
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import  pandas as pd
import time

class collaboration:
    def __init__(self):

        self.end_year=2018
        self.author_ids=[]
        # properties related to collaborations
        self.year_indiv_links={}
        self.year_indiv_deleted_links={}
        self.year_indiv_deleted_num={}
        self.active_year=set()
        self.b_e=[]
        self.year_preprocess_list={}
        self.year_deleted_author={}   #author
        self.year_deleted_author_num={}
        self.year_coller={}
        self.year_coller_num={}
        self.year_author_num={}

        #properties related to  author collaborations
        self.authorId_year_indiv_links={}
        self.authorId_year_indiv_deleted_links={}
        self.authorId_year_indiv_deleted_num={}
        self.authorId_b_e={}    #author's first and last year
        self.authorId_active_year={}
        self.authorId_year_coller={}    # author's links
        self.authorId_year_degree={}

        #conditional proba
        self.num={}
        self.pre_next={}

        # aff
        self.year_aff_authorSet = {}
        self.year_aff_paperSet = {}
        self.year_aff_size={}
        self.year_aff_paper_num={}

        self.year_aff_deleted_internal_links = {}
        self.year_aff_deleted_internal_num = {}
        self.year_aff_deleted_external_links = {}
        self.year_aff_deleted_external_num = {}
        self.aff_size_average_internal_deleted_num={}
        self.aff_size_average_external_deleted_num = {}
       #others
        self.year_alive_author={}  #author who is still in his academic career
        self.year_active_author={}  #author who write a paper

    def add_author_collaboration(self,collab_list,author_id):    #输入的collab_list是一个人的list
        self.authorId_active_year[author_id]=set()
        for collab in collab_list:
            self.authorId_active_year[author_id].add((collab[4]))
        self.authorId_b_e[author_id]=[min(self.authorId_active_year[author_id]),max(self.authorId_active_year[author_id])]
        if (self.authorId_b_e[author_id][0]> self.end_year-5):
            del self.authorId_active_year[author_id]
            del self.authorId_b_e[author_id]
            return

        collab_list.sort(key=operator.itemgetter(4))        #按照年份排序
        life_span=5
        temp_collab_list= [i for i in collab_list if (i[4]==self.authorId_b_e[author_id][0])]

        self.authorId_year_indiv_links[author_id]={}
        self.authorId_year_indiv_links[author_id][str(self.authorId_b_e[author_id][0])]=[]
        self.authorId_year_coller[author_id]={}
        self.authorId_year_coller[author_id][str(self.authorId_b_e[author_id][0])] = []
        for collab in temp_collab_list:
            self.authorId_year_indiv_links[author_id][str(self.authorId_b_e[author_id][0])].append([collab[2],life_span])    #初始化
            self.authorId_year_coller[author_id][str(self.authorId_b_e[author_id][0])].append(collab[2])

        self.authorId_year_indiv_deleted_links[author_id] = {}

        for year in range(self.authorId_b_e[author_id][0]+1,min(self.authorId_b_e[author_id][1]+6, self.end_year+1)):
            self.authorId_year_indiv_links[author_id][str(year)]=copy.deepcopy(self.authorId_year_indiv_links[author_id][str(year-1)])
            self.authorId_year_coller[author_id][str(year)]=copy.deepcopy(self.authorId_year_coller[author_id][str(year-1)])
            for i in range(len(self.authorId_year_indiv_links[author_id][str(year)])):
                self.authorId_year_indiv_links[author_id][str(year)][i][1]=self.authorId_year_indiv_links[author_id][str(year-1)][i][1]-1
            if year not in self.authorId_active_year[author_id]:  # 如果这一年这个作者没有写论文
                deleted_links = [i for i in self.authorId_year_indiv_links[author_id][str(year)] if (i[1] == 0)]
                if len(deleted_links)!=0:
                    for deleted_link in deleted_links:
                        self.authorId_year_indiv_links[author_id][str(year)].remove(deleted_link)    #去除die的links
                    for deleted_id in deleted_links:
                        self.authorId_year_coller[author_id][str(year)].remove(deleted_id[0])
            else:
                temp_collab_list = [i for i in collab_list if (i[4] == year)]
                for collab in temp_collab_list:
                    if collab[2] in  self.authorId_year_coller[author_id][str(year-1)]:
                        index=self.authorId_year_coller[author_id][str(year-1)].index(collab[2])
                        self.authorId_year_indiv_links[author_id][str(year)][index]=[collab[2],life_span]
                    else:
                        self.authorId_year_indiv_links[author_id][str(year)].append([collab[2],life_span])
                        self.authorId_year_coller[author_id][str(year)].append(collab[2])

                deleted_links = [i for i in self.authorId_year_indiv_links[author_id][str(year)] if (i[1] == 0)]
                if len(deleted_links)!=0:
                    for deleted_link in deleted_links:
                        self.authorId_year_indiv_links[author_id][str(year)].remove(deleted_link)    #去除die的links
                    for deleted_id in deleted_links:
                        self.authorId_year_coller[author_id][str(year)].remove(deleted_id[0])
            self.authorId_year_indiv_deleted_links[author_id][str(year)]=[]
            for link in deleted_links:
                self.authorId_year_indiv_deleted_links[author_id][str(year)].append(link[0])


            # num of deleted_link
        self.authorId_year_indiv_deleted_num[author_id] = {}
        for year ,link in self.authorId_year_indiv_deleted_links[author_id].items():
            self.authorId_year_indiv_deleted_num[author_id][year]=len(link)

            #degree of authors per year
        self.authorId_year_degree[author_id]={}
        for year in range(self.authorId_b_e[author_id][0],min(self.authorId_b_e[author_id][1]+6, self.end_year+1)):
            self.authorId_year_degree[author_id][str(year)]=len(self.authorId_year_coller[author_id][str(year)])

        #self.year_alive_author ,self.year_active_author
        for year in self.authorId_active_year[author_id]:
            if str(year) not in self.year_active_author.keys():
                self.year_active_author[str(year)]=set()
                self.year_active_author[str(year)].add(author_id)
            else:
                self.year_active_author[str(year)].add(author_id)

        for year in range(self.authorId_b_e[author_id][0],self.authorId_b_e[author_id][1]+1):
            if str(year) not in self.year_alive_author.keys():
                self.year_alive_author[str(year)] = set()
                self.year_alive_author[str(year)].add(author_id)
            else:
                self.year_alive_author[str(year)].add(author_id)

        self.author_ids.append(author_id)

    def add_overall_collaboration(self,collab_list):   #输入为原始csv文件按时间排序的结果
        self.b_e=[collab_list[0][4],collab_list[-1][4]]
        for collab in collab_list:
            self.active_year.add((collab[4]))

        temp_year_list=[]
        for collab in collab_list :      #先生成非累积合作表
           if collab[4] not in  temp_year_list:
               temp_year_list.append(collab[4])
               self.year_preprocess_list[str(collab[4])]=[]
               self.year_preprocess_list[str(collab[4])].append([collab[0],collab[2]])

           else:
                self.year_preprocess_list[str(collab[4])].append([collab[0],collab[2]])

        life_span=5
        self.year_indiv_links[str(self.b_e[0])]={}
        for collab in self.year_preprocess_list[str(self.b_e[0])]:
            self.year_indiv_links[str(self.b_e[0])][str(collab[0])+'/'+str(collab[1])]=life_span

        self.year_coller[str(self.b_e[0])]=list(self.year_indiv_links[str(self.b_e[0])].keys())

        for year in range(self.b_e[0]+1,self.b_e[1]+1):
            self.year_indiv_links[str(year)] = copy.deepcopy( self.year_indiv_links[str(year - 1)])
            for key in self.year_indiv_links[str(year)].keys():
                self.year_indiv_links[str(year)][key] = self.year_indiv_links[str(year)][key] - 1
            if year not in self.active_year:       #本年度没人写文章（基本不可能）
                deleted_links = [i for i in self.year_indiv_links[str(year)] if (self.year_indiv_links[str(year)][i]==0)]
                if len(deleted_links)!=0:
                    for deleted_link in deleted_links:
                        del self.year_indiv_links[str(year)][deleted_link]    #去除die的links
                self.year_coller[str(year)] = list(self.year_indiv_links[str(year)].keys())
            else:
                for collab in self.year_preprocess_list[str(year)]:
                    self.year_indiv_links[str(year)].update({str(collab[0])+'/'+str(collab[1]):life_span})   #使用update函数更新

                deleted_links = [i for i in self.year_indiv_links[str(year)] if(self.year_indiv_links[str(year)][i] == 0)]

                if len(deleted_links)!=0:
                    for deleted_link in deleted_links:
                        del self.year_indiv_links[str(year)][deleted_link]    #去除die的links
                self.year_coller[str(year)] = list(self.year_indiv_links[str(year)].keys())
            self.year_indiv_deleted_links[str(year)]=[]
            for link in deleted_links:
                self.year_indiv_deleted_links[str(year)].append(link)

        #self.year_coller_num
        for year in range(self.b_e[0] , self.b_e[1] + 1):
            self.year_coller_num[str(year)]=len(self.year_coller[str(year)])/2



        #self.year_indiv_deleted_num
        for year in range(self.b_e[0]+1,self.b_e[1]+1):
            self.year_indiv_deleted_num[str(year)]=len(self.year_indiv_deleted_links[str(year)])/2

        #self.year_deleted_author

        for year in range(self.b_e[0] + 1, self.b_e[1] + 1):
            self.year_deleted_author[str(year)]=set()
            for string in self.year_indiv_deleted_links[str(year)]:
                temp_author=string[0:string.find('/')]
                self.year_deleted_author[str(year)].add(temp_author)

        # self.year_deleted_author_num
        for year in range(self.b_e[0] + 1, self.b_e[1] + 1):
            self.year_deleted_author_num[str(year)]=len(self.year_deleted_author[str(year)])


   #  update      life_span不用更新，无关紧要
    def update_collaboration(self,flag,author_ids):
        if flag==0:
           return
        else:            #更新authorid_year_coller,authorid_year_degree,authorid_year_indiv_deleted_links and num

            for author_id in author_ids:  #authorid_year_coller
                author_id=str(author_id)
                for year in range(self.authorId_b_e[author_id][0]+1,min(self.end_year+1,self.authorId_b_e[author_id][1]+6)):
                    if len(self.authorId_year_indiv_deleted_links[author_id][str(year)])!=0:
                        for link in self.authorId_year_indiv_deleted_links[author_id][str(year)]:
                            self.authorId_year_coller[author_id][str(year-1)].remove(link)
                            self.authorId_year_coller[author_id][str(year - 2)].remove(link)
                            self.authorId_year_coller[author_id][str(year - 3)].remove(link)
                            self.authorId_year_coller[author_id][str(year - 4)].remove(link)
                for year in range(min(self.end_year-4,self.authorId_b_e[author_id][1]+1),min(self.end_year+1,self.authorId_b_e[author_id][1]+6)):
                    del self.authorId_year_coller[author_id][str(year)]

                #authorid_year_indiv_deleted_links
                for year in range(self.authorId_b_e[author_id][0]+1,min(self.end_year-3,self.authorId_b_e[author_id][1]+2)):
                    self.authorId_year_indiv_deleted_links[author_id][str(year)] = copy.deepcopy(self.authorId_year_indiv_deleted_links[author_id][str(year+4)])

                for year in range(min(self.end_year-3,self.authorId_b_e[author_id][1]+2),min(self.end_year+1,self.authorId_b_e[author_id][1]+6)):
                    del self.authorId_year_indiv_deleted_links[author_id][str(year)]

                    # num of deleted_link
                self.authorId_year_indiv_deleted_num[author_id] = {}
                for year, link in self.authorId_year_indiv_deleted_links[author_id].items():
                    self.authorId_year_indiv_deleted_num[author_id][year] = len(link)

                    # degree of authors per year
                self.authorId_year_degree[author_id] = {}
                for year in range(self.authorId_b_e[author_id][0], min(self.end_year-4,self.authorId_b_e[author_id][1]+1)):
                    self.authorId_year_degree[author_id][str(year)] = len(
                        self.authorId_year_coller[author_id][str(year)])

            # self.year_coller


            for year in range(self.b_e[0] + 1, self.b_e[1]):
                if len(self.year_indiv_deleted_links[str(year)]) != 0:
                    for link in self.year_indiv_deleted_links[str(year)]:
                        self.year_coller[str(year - 1)].remove(link)
                        self.year_coller[str(year - 2)].remove(link)
                        self.year_coller[str(year - 3)].remove(link)
                        self.year_coller[str(year - 4)].remove(link)
                print('update', year, ',', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


            for year in range(self.b_e[1] -4, self.b_e[1] + 1):
                del self.year_coller[str(year)]
                #del self.year_coller_num[str(year)]

            # self.year_coller_num
            self.year_coller_num={}
            for year in range(self.b_e[0], self.b_e[1] -4):
                self.year_coller_num[str(year)] = len(self.year_coller[str(year)])/2


            #self.year_indiv_deleted_links
            for year in range(self.b_e[0]+1,self.b_e[1]-3):
                self.year_indiv_deleted_links[str(year)] = copy.deepcopy(
                    self.year_indiv_deleted_links[str(year + 4)])
            for year in range(self.b_e[1]-3,self.b_e[1]+1):
                del self.year_indiv_deleted_links[str(year)]

            self.year_indiv_deleted_num={}
            for year in range(self.b_e[0] + 1, self.b_e[1] -3):
                self.year_indiv_deleted_num[str(year)] = len(self.year_indiv_deleted_links[str(year)])/2



            #self.year_deleted_author
            self.year_deleted_author={}
            self.year_deleted_author_num={}
            for year in range(self.b_e[0] + 1, self.b_e[1] -3):
                self.year_deleted_author[str(year)]=set()
                for string in self.year_indiv_deleted_links[str(year)]:
                    temp_author=string[0:string.find('/')]
                    self.year_deleted_author[str(year)].add(temp_author)

            # self.year_deleted_author_num
            for year in range(self.b_e[0] + 1, self.b_e[1] -3):
                self.year_deleted_author_num[str(year)]=len(self.year_deleted_author[str(year)])







    def conditional_prob(self,author_ids,flag):
        self.num['0']=[];self.num['1']=[];self.num['2']=[];self.num['3']=[];self.num['4']=[];self.num['5']=[];
        self.num['6'] = [];self.num['7']=[];self.num['8']=[];self.num['9']=[];self.num['10']=[];

        for author_id in (author_ids):
            author_id=str(author_id)
            if flag==0:
                for year in range(self.authorId_b_e[author_id][0]+1, min(self.end_year+1,self.authorId_b_e[author_id][1] + 5)):
                    if self.authorId_year_indiv_deleted_num[author_id][str(year)]==0:
                        self.num['0'].append(self.authorId_year_indiv_deleted_num[author_id][str(year+1)])
                    elif self.authorId_year_indiv_deleted_num[author_id][str(year)]==1:
                        self.num['1'].append(self.authorId_year_indiv_deleted_num[author_id][str(year+1)])
                    elif self.authorId_year_indiv_deleted_num[author_id][str(year)]==2:
                        self.num['2'].append(self.authorId_year_indiv_deleted_num[author_id][str(year+1)])
                    elif self.authorId_year_indiv_deleted_num[author_id][str(year)]==3:
                        self.num['3'].append(self.authorId_year_indiv_deleted_num[author_id][str(year+1)])
                    elif self.authorId_year_indiv_deleted_num[author_id][str(year)]==4:
                        self.num['4'].append(self.authorId_year_indiv_deleted_num[author_id][str(year+1)])
                    elif self.authorId_year_indiv_deleted_num[author_id][str(year)]==5:
                        self.num['5'].append(self.authorId_year_indiv_deleted_num[author_id][str(year+1)])
                    elif self.authorId_year_indiv_deleted_num[author_id][str(year)]==6:
                        self.num['6'].append(self.authorId_year_indiv_deleted_num[author_id][str(year+1)])
                    elif self.authorId_year_indiv_deleted_num[author_id][str(year)]==7:
                        self.num['7'].append(self.authorId_year_indiv_deleted_num[author_id][str(year+1)])
                    elif self.authorId_year_indiv_deleted_num[author_id][str(year)]==8:
                        self.num['8'].append(self.authorId_year_indiv_deleted_num[author_id][str(year+1)])
                    elif self.authorId_year_indiv_deleted_num[author_id][str(year)]==9:
                        self.num['9'].append(self.authorId_year_indiv_deleted_num[author_id][str(year+1)])
                    elif self.authorId_year_indiv_deleted_num[author_id][str(year)]==10:
                        self.num['10'].append(self.authorId_year_indiv_deleted_num[author_id][str(year+1)])
            else:
                for year in range(self.authorId_b_e[author_id][0] + 1, min(self.end_year-4,self.authorId_b_e[author_id][1] + 1)):
                    if self.authorId_year_indiv_deleted_num[author_id][str(year)] == 0:
                        self.num['0'].append(self.authorId_year_indiv_deleted_num[author_id][str(year + 1)])
                    elif self.authorId_year_indiv_deleted_num[author_id][str(year)] == 1:
                        self.num['1'].append(self.authorId_year_indiv_deleted_num[author_id][str(year + 1)])
                    elif self.authorId_year_indiv_deleted_num[author_id][str(year)] == 2:
                        self.num['2'].append(self.authorId_year_indiv_deleted_num[author_id][str(year + 1)])
                    elif self.authorId_year_indiv_deleted_num[author_id][str(year)] == 3:
                        self.num['3'].append(self.authorId_year_indiv_deleted_num[author_id][str(year + 1)])
                    elif self.authorId_year_indiv_deleted_num[author_id][str(year)] == 4:
                        self.num['4'].append(self.authorId_year_indiv_deleted_num[author_id][str(year + 1)])
                    elif self.authorId_year_indiv_deleted_num[author_id][str(year)] == 5:
                        self.num['5'].append(self.authorId_year_indiv_deleted_num[author_id][str(year + 1)])
                    elif self.authorId_year_indiv_deleted_num[author_id][str(year)] == 6:
                        self.num['6'].append(self.authorId_year_indiv_deleted_num[author_id][str(year + 1)])
                    elif self.authorId_year_indiv_deleted_num[author_id][str(year)] == 7:
                        self.num['7'].append(self.authorId_year_indiv_deleted_num[author_id][str(year + 1)])
                    elif self.authorId_year_indiv_deleted_num[author_id][str(year)] == 8:
                        self.num['8'].append(self.authorId_year_indiv_deleted_num[author_id][str(year + 1)])
                    elif self.authorId_year_indiv_deleted_num[author_id][str(year)] == 9:
                        self.num['9'].append(self.authorId_year_indiv_deleted_num[author_id][str(year + 1)])
                    elif self.authorId_year_indiv_deleted_num[author_id][str(year)] == 10:
                        self.num['10'].append(self.authorId_year_indiv_deleted_num[author_id][str(year + 1)])

        # 数据统计
        for num in self.num.keys():
            self.pre_next[num]={}
            if len(self.num[num]) != 0:
                lens = len(self.num[num])
                temp_counter = []
                for i in self.num[num]:
                    if i not in temp_counter:
                        temp_counter.append(i)
                        self.pre_next[num][i] = 1
                    else:
                        self.pre_next[num][i] = self.pre_next[num][i] + 1
            #self.pre_next[num] = sorted(self.pre_next[num].items(), key=lambda x: x[0])

    def year_aff_generate(self, collab_list):
        for row in collab_list:
            year = str(row[4])
            if year not in self.year_aff_authorSet:
                self.year_aff_authorSet[year] = {}
            if year not in self.year_aff_paperSet:
                self.year_aff_paperSet[year] = {}
            for aff in [row[1], row[3]]:
                if aff not in self.year_aff_authorSet[year]:
                    self.year_aff_authorSet[year][aff] = set()
                if aff not in self.year_aff_paperSet[year]:
                    self.year_aff_paperSet[year][aff] = set()

               # paperList = row[5].replace('[', '').replace(']', '').replace(' ', '').split(',')
               #for paper in paperList:
                    #self.year_aff_paperSet[year][aff].add(paper)
                if aff==row[1]:
                    self.year_aff_authorSet[year][aff].add(row[0])
                else:
                    self.year_aff_authorSet[year][aff].add(row[2])

        for key, value in self.year_aff_authorSet.items():
            self.year_aff_size[key]={}
            for key_2,value_2 in value.items():
                self.year_aff_size[key][key_2]=len(value_2)



    def add_aff_collaboration(self):
        for year in range(self.b_e[0],self.b_e[1]-4):
            if str(year) in self.year_aff_authorSet:
                temp_list_x = []
                temp_list_y = []
                self.year_aff_deleted_internal_links[str(year+1)]={}
                self.year_aff_deleted_external_links[str(year+1)] = {}

                for string in self.year_indiv_deleted_links[str(year+1)]:
                    temp_list_x.append(string[0:string.find('/')])
                    temp_list_y.append(string[string.find('/') + 1:len(string) + 1])
                for key,value in self.year_aff_authorSet[str(year)].items():
                    self.year_aff_deleted_internal_links[str(year+1)][key] = []
                    self.year_aff_deleted_external_links[str(year+1)][key]= []
                    for i in value:
                        if i in temp_list_x:
                            index = [j for j, x in enumerate(temp_list_x) if x == i]
                            for j in index:
                                if temp_list_y[j] in value:   #internal
                                    self.year_aff_deleted_internal_links[str(year+1)][key].append(i+'/'+temp_list_y[j])
                                else:                         #external
                                    self.year_aff_deleted_external_links[str(year+1)][key].append(
                                        i + '/' + temp_list_y[j])
            print('aff_collab ', year, ',', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


        for year in  range(self.b_e[0]+1,self.b_e[1]-3):
            if str(year) in self.year_aff_deleted_internal_links:
                self.year_aff_deleted_internal_num[str(year)]={}
                for key,value in self.year_aff_deleted_internal_links[str(year)].items():
                    self.year_aff_deleted_internal_num[str(year)][key]=len(value)/2

        for year in  range(self.b_e[0]+1,self.b_e[1]-3):
            if str(year) in self.year_aff_deleted_external_links:
                self.year_aff_deleted_external_num[str(year)]={}
                for key,value in self.year_aff_deleted_external_links[str(year)].items():
                    self.year_aff_deleted_external_num[str(year)][key]=len(value)







    def collabration_plot(self,author_ids):


        # 绘制条件概率分布图
        i=1
        for num in self.num.keys():
            if len(self.num[num]) != 0:
                plt.subplot(5,3,i)
                #plt.axes(yscale="log")
                #plt.xlabel('Conditional Prob on ' + num + ' times the previous year')
                #plt.ylabel('the num of del links the next year')
                n, bins, patches = plt.hist(self.num[num],20,(0,20),density = False)
                i=i+1
                plt.show()


        # 绘制总的deleted links随年份变化数
        year_list=[]
        year_num=[]
        for key,value in self.year_indiv_deleted_num.items():
            year_list.append(int(key))
            year_num.append(value)

        plt.figure()
        plt.plot(year_list,year_num)
        plt.title('the number of deleted links every year')


        #绘制average 随年份变化数

        year_list = []
        year_num = []

        for key,value in self.year_indiv_deleted_num.items():
            year_list.append(int(key))
            year_num.append(value/max(1,self.year_deleted_author_num[key]))

        plt.figure()
        plt.plot(year_list,year_num)
        plt.title('the average number of deleted links every year')

        #绘制每年的合作者总数
        year_list = []
        year_num = []

        for key, value in self.year_coller_num.items():
            year_list.append(int(key))
            year_num.append(value)

        plt.figure()
        plt.plot(year_list, year_num)
        plt.title('the number of links every year')


        #deleted links the next year vs degree
        temp_degree_deleted_num={}
        for authorid in author_ids:
            authorid=str(authorid)
            for year in range(self.authorId_b_e[authorid][0],min(self.end_year-4,self.authorId_b_e[authorid][1]+1)):
                if self.authorId_year_degree[authorid][str(year)] not in temp_degree_deleted_num.keys():
                    temp_degree_deleted_num[ self.authorId_year_degree[authorid][str(year)]]=[]
                    temp_degree_deleted_num[self.authorId_year_degree[authorid][str(year)]].append(
                        self.authorId_year_indiv_deleted_num[authorid][str(year+1)])
                else:
                    temp_degree_deleted_num[self.authorId_year_degree[authorid][str(year)]].append(
                        self.authorId_year_indiv_deleted_num[authorid][str(year+1)])

        degree_list=[]
        ave_deleted_num_list=[]
        for key,value in temp_degree_deleted_num.items():
            degree_list.append(key)
            ave_deleted_num_list.append(np.mean(value))

        plt.figure()
        plt.scatter(degree_list, ave_deleted_num_list)
        plt.xlabel('degree in this year')
        plt.ylabel('average deleted links the next year')



        # heat map(deleted links previous year ,degree this year vs. deleted links the next year)
        temp_dict={}
        for authorid in author_ids:
            authorid=str(authorid)
            for year in range(self.authorId_b_e[authorid][0]+1,min(self.end_year-4,self.authorId_b_e[authorid][1]+1)):
                temp_key_1=str(self.authorId_year_degree[authorid][str(year)])
                temp_key_2=str(self.authorId_year_indiv_deleted_num[authorid][str(year)])
                if temp_key_1 not in temp_dict.keys():
                    temp_dict[temp_key_1]={}
                    temp_dict[temp_key_1][temp_key_2]=[]
                    temp_dict[temp_key_1][temp_key_2].append(self.authorId_year_indiv_deleted_num[authorid][str(year + 1)])
                else:
                    if temp_key_2 not in temp_dict[temp_key_1].keys():
                        temp_dict[temp_key_1][temp_key_2] = []
                        temp_dict[temp_key_1][temp_key_2].append(
                            self.authorId_year_indiv_deleted_num[authorid][str(year + 1)])
                    else:
                        temp_dict[temp_key_1][temp_key_2].append(
                            self.authorId_year_indiv_deleted_num[authorid][str(year + 1)])


        real_key_1=[]
        real_key_2=set()

        for key_1,_ in temp_dict.items():
            real_key_1.append(int(key_1))
            for key_2,value in temp_dict[key_1].items():
                real_key_2.add(int(key_2))
                temp_dict[key_1][key_2]=np.mean(temp_dict[key_1][key_2])

        real_key_2=list(real_key_2)
        real_key_2.sort(reverse=True)
        real_key_1.sort()

        #construct dataframe
        real_dict={}
        for key_1 in real_key_1:
            key_1=str(key_1)
            real_dict[key_1]=[]
            for key_2 in real_key_2:
                key_2=str(key_2)
                if key_2 not in temp_dict[key_1].keys():
                    real_dict[key_1].append(None)
                else:
                    real_dict[key_1].append(temp_dict[key_1][key_2])

        real_dataframe=pd.DataFrame(real_dict,index=real_key_2)
        plt.figure()
        sns.heatmap(real_dataframe,annot=True,cmap="RdBu_r",linewidths=0.2)
        plt.xlabel('degree this year')
        plt.ylabel('deleted links this year')
        plt.title('deleted links the next year')


        # aff_size vs the next year's num of deleted links
        self.aff_size_average_internal_deleted_num={}
        self.aff_size_average_external_deleted_num={}

        for year in range(self.b_e[0],self.b_e[1]-4):
            if str(year) in self.year_aff_size.keys():
                for aff ,size in self.year_aff_size[str(year)].items():
                    if str(size) not in self.aff_size_average_internal_deleted_num.keys():
                        self.aff_size_average_internal_deleted_num[str(size)]=[]
                        self.aff_size_average_internal_deleted_num[str(size)].append(self.year_aff_deleted_internal_num[str(year+1)][aff])
                    else:
                        self.aff_size_average_internal_deleted_num[str(size)].append(
                            self.year_aff_deleted_internal_num[str(year + 1)][aff])



        for year in range(self.b_e[0],self.b_e[1]-4):
            if str(year) in self.year_aff_size.keys():
                for aff ,size in self.year_aff_size[str(year)].items():
                    if str(size) not in self.aff_size_average_external_deleted_num.keys():
                        self.aff_size_average_external_deleted_num[str(size)]=[]
                        self.aff_size_average_external_deleted_num[str(size)].append(self.year_aff_deleted_external_num[str(year+1)][aff])
                    else:
                        self.aff_size_average_external_deleted_num[str(size)].append(
                            self.year_aff_deleted_external_num[str(year + 1)][aff])


        for key,value in self.aff_size_average_internal_deleted_num.items():
            self.aff_size_average_internal_deleted_num[key]=np.mean(value)

        for key,value in self.aff_size_average_external_deleted_num.items():
            self.aff_size_average_external_deleted_num[key]=np.mean(value)


        internal_size_list=[];internal_deleted_num_list=[]
        external_size_list = []; external_deleted_num_list = []
        for key , value in self.aff_size_average_internal_deleted_num.items():
            internal_size_list.append(int(key))
            internal_deleted_num_list.append(value)


        for key , value in self.aff_size_average_external_deleted_num.items():
            external_size_list.append(int(key))
            external_deleted_num_list.append(value)


        plt.figure()
        plt.scatter(internal_size_list, internal_deleted_num_list)
        plt.xlabel('size in this year')
        plt.ylabel('average internal deleted links the next year')

        plt.figure()
        plt.scatter(external_size_list, external_deleted_num_list)
        plt.xlabel('size in this year')
        plt.ylabel('average external deleted links the next year')




