from Model import *
from DataWeight_load import *
from preprocessing import *
import numpy as np
from random import shuffle

def Validation(model, data_ans_raw, data_ans_emb, data_embedd, data_predicted):
    '''
    Validation(model, y_test, y_test_ans, y_test_embed, predict_test)
    predict_test => target vector
    y_test_embed => class_embedd ( 0 to number of classes)
    y_test_ans => class_name
    data_ans_raw => serial list of class
    '''
    arr_inx = list(range(len(data_predicted)))
    shuffle(arr_inx)
    total_count = 0
    cor_count = 0
    extra = 0
    same_count = 0
    wrong_count = 0
    valid_set =[]
    valid_idx = []
    min_idx = []
    #print(arr_inx)
    print(data_embedd)
    print(data_ans_emb)
    print(data_ans_raw)
    print("=====================")

    for i, encoded_piece in enumerate(data_predicted):
        min_distance = 1000
        min_index = 0

        simular_count = 0
        for k in data_ans_raw:
            if data_ans_raw[i] == k:
                simular_count = simular_count + 1

        if simular_count == 1:
            same_count = same_count + 1
            valid_set.append(data_embedd[i])
            valid_idx.append(i)
            min_idx.append(i)
        else:
            for j, others in enumerate(data_predicted):
                if i != j:
                    distance = np.linalg.norm(encoded_piece - others)
                    if distance < min_distance:
                        min_distance = distance
                        min_index = j

            if data_ans_raw[min_index] == data_ans_raw[i]:
                valid_set.append(data_embedd[i])
                valid_idx.append(i)
                min_idx.append(min_idx)
                cor_count = cor_count + 1
            else:
                for j, others in enumerate(data_predicted):
                    distance = np.linalg.norm(encoded_piece - others)
                    if distance < min_distance:
                        min_distance = distance
                        min_index = j

                if data_ans_raw[min_index] == data_ans_raw[i]:
                    extra = extra + 1
                else:
                    wrong_count = wrong_count + 1

        total_count = total_count + 1

    print("==================")
    print(same_count)
    print(cor_count)
    print(extra)
    print(wrong_count)
    print(total_count)
    print(str(int(((same_count+cor_count)/total_count)*100)))
    print(str(int(((same_count + cor_count + extra) / total_count )* 100)))
    print("==================")
    '''
    sum = 0
    
    for i in range(len(valid_set)):
        print("=======")
        print(valid_set[i])
        print(valid_idx[i])
        print(data_ans_emb[valid_set[i]])
        print("-----------")
        print(data_embedd[i])
        print(i)
        print(data_ans_emb[data_embedd[i]])
        print("=======")
        if valid_idx[i] != i:
            sum = sum + 1

    print("===")
    print(sum)
    print(str(int(((total_count-cor_count)/total_count)*100)))
    '''
    return 1