from Model import *
from DataWeight_load import *
from preprocessing import *
import numpy as np


def Validation(model, data_raw, data_ans, data_embedd, data_predicted):
    total_count = 0
    cor_count = 0
    meanless_count = 0

    for i, encoded_piece in enumerate(data_predicted):
        distance = 0
        min_distance = 1000
        min_index = -1
        count = 0

        for temp in data_predicted:
            if np.linalg.norm(encoded_piece - temp) == 0:
                count = count + 1

        if count == 1:
            meanless_count = meanless_count + 1
            continue
        else:
            pass

        for j, others in enumerate(data_predicted):
            if i != j:
                distance = np.linalg.norm(encoded_piece - others)

                if distance < min_distance:
                    min_distance = distance
                    min_index = j

        if data_embedd[min_index] == data_embedd[i]:
            cor_count = cor_count + 1
            # print(cor_count)

        total_count = total_count + 1

    print("==================")
    print(meanless_count)
    print(cor_count)
    print(total_count)

    print(str(int(cor_count / total_count * 100)))
    print(str(int((cor_count + meanless_count) / (total_count + meanless_count) * 100)))

    return 1