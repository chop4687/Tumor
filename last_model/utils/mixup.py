import torch
import random
from torch import nn
def mixup(feature, label):
    #################### 한개짜리
    # #print(feature.shape)
    # add = 10
    # mixs = 0
    # sels = 0
    # rand = random.randint(0,len(feature)-1)
    # first_label = feature[rand,...]
    # # print(label[0])
    # for i,lab in enumerate(label):
    #     if label[rand] != lab:
    #         second_label = feature[i,...]
    #         def_label = label[i]
    #         # print(label[i])
    #         break
    # #label에서 label[rand]인 class들만 모아다가 평균내기
    # first_features = feature[label == label[rand]]
    # anchor_feature = first_features[random.randint(0,len(first_features)-1),...]
    # mean_feature = first_features.mean(0)
    # #print(label == label[0])
    # # 그담에 first랑 second 섞기(0.5비율로) = mix feature가 됨
    # mix_feature = first_label * 0.5 + second_label * 0.5
    # # far feature는 label[0]인 놈들 중 mean feature랑 서로빼고 제곱했을때 제일 값이 큰feature
    # far_feature = (feature[label == label[rand]] - mean_feature)**2
    #
    # far_feature = feature[far_feature.sum((1,2,3)).argmax(),...]
    #
    # # if ((far_feature - mean_feature)**2).sum() < ((mix_feature - mean_feature)**2).sum():
    # #     return 0
    # # else:
    # #     return 1
    # sel = ((anchor_feature - far_feature)**2).sum()/(256*11*22)
    # mix = ((mix_feature - anchor_feature)**2).sum()/(256*11*22)
    #
    # result = max(sel - sel, sel + 0.001 - mix)
    ################################################
    rand = random.randint(0,len(feature)-1)

    first_features = feature[label == label[rand]]
    first_rand = random.randint(0,len(first_features)-1)
    first_feature = first_features[first_rand]

    for i,lab in enumerate(label):
        if lab != label[rand]:
            diff_cls = label[i]
            break
    second_features = feature[label == diff_cls]
    mix_feature = first_features[random.randint(0,len(first_features)-1)] * 0.5 + second_features[random.randint(0,len(second_features)-1)] * 0.5

    temp = (first_feature - first_features)**2 # vector (21,256,11,22)
    temp = temp.sum((1,2,3))/(256*11*22) #vector (21,1)

    mix_temp = (mix_feature - first_features)**2
    mix_temp = mix_temp.sum((1,2,3))/(256*11*22) #vector (21,1)

    result = temp + 1 - mix_temp
    result = result.mean()
    result = max(result-result, result)
    mix_label = [label[rand], diff_cls]
    return result, mix_feature, mix_label
