from Model import *
from DataWeight_load import *
from preprocessing import *
import numpy as np
from random import shuffle

def Validation(model, data_ans_raw, data_ans_emb, data_embedd, data_predicted):
    arr_inx = list(range(len(data_predicted)))
    shuffle(arr_inx)
    print(arr_inx)

    total_count = 0
    cor_count = 0
    meanless_count = 0
    valid_set =[]
    print(data_embedd)
    print(data_ans_emb)
    print(data_ans_raw)
    print("=====================")
    for i, encoded_piece in enumerate(data_predicted):
        i = arr_inx[i]
        min_distance = 1000
        min_index = 0
        count = 0

        for temp in data_embedd:
            if temp == data_embedd[i]:
                count = count +1

        if count == 1:
            meanless_count = meanless_count +1

        for j, others in enumerate(data_predicted):
            #if i != j:
            distance = np.linalg.norm(encoded_piece - others)
            if distance < min_distance:
                min_distance = distance
                min_index = j


        if data_embedd[min_index] == data_embedd[i]:
            #print(data_embedd[i])
            if data_ans_emb[data_embedd[i]] == data_ans_raw[i]:
                valid_set.append(data_embedd[i])
                cor_count = cor_count + 1
            #print(cor_count)

        total_count = total_count + 1

    print("==================")
    print(meanless_count)
    print(cor_count)
    print(total_count)
    print( str(int(cor_count/total_count*100)) )
    print( str(int((cor_count+meanless_count)/(total_count+meanless_count)*100)))
    print("==================")
    sum = 0

    for i in range(len(valid_set)):
        print(valid_set[i])
        print("---")
        print(data_embedd[i])
        if valid_set[i] != data_embedd[i]:
            sum = sum + 1

    print("===")
    print(sum)
    print(str(int(((total_count-cor_count)/total_count)*100)))
    print(arr_inx)
    return 1