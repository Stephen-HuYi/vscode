


#delete all of the one_time_collab
def delete_one_time_collab(colla_list):
    temp_dict={}
    new_colla_list=[]
    i=0  #column index
    num=1
    for collab in colla_list:
        temp_key=collab[0]+'/'+collab[2]
        if temp_key not in temp_dict.keys():
            temp_dict[temp_key]=[]
            temp_dict[temp_key].append([i,collab[5].count(',')+1])
        else:
            temp_dict[temp_key].append([i, collab[5].count(',')+1])
        i=i+1

    for key,value in temp_dict.items():
        if (len(value)!=1 or value[0][1]!=1):
            for index in value:
                new_colla_list.append(colla_list[index[0]])
    return new_colla_list









#delete the situation :A & B collaborate with each other several times in one year and one of them change their institution

def delete_part_colla(part_colla_list):
    remove_index=set()

    for i in range(0,len(part_colla_list)-1):
        for j in range(i+1,len(part_colla_list)):
            if (part_colla_list[i][0]==part_colla_list[j][0] and part_colla_list[i][2]==part_colla_list[j][2] and part_colla_list[i][4]==part_colla_list[j][4]):
                remove_index.add(j)
                #remove_index.add(i)

    remove_index=list(remove_index)
    remove_list=[]
    for i in remove_index:
        remove_list.append(part_colla_list[i])
    for i in remove_list:
        part_colla_list.remove(i)
    return part_colla_list

