import torch
import torch.nn as nn
import torch.nn.functional as F


class ISDALoss(nn.Module):
    def __init__(self, feature_num, class_num):
        super(ISDALoss, self).__init__()
        self.class_num = class_num
        self.cross_entropy = nn.CrossEntropyLoss()

    def isda_aug(self, linear_layer, features, labels, cv_matrix, ratio):
        
        N = features.size(0)  # batch size
        C = self.class_num    # number of class (200 for CUB-200-2011)
        A = features.size(1)  # feature dimension (2048 for ResNet50)
        #print('features',features.shape)
        weight_m = list(linear_layer.parameters())[0]  # weight of Linear, shape = [200, 2048] = [C, A]
        #print('weight_m',weight_m.shape)
        NxW_ij = weight_m.expand(N, C, A)  # shape=[8, 200, 2048] = [N, C, A] (copy weight_m for N times)
        NxW_kj = torch.gather(
            NxW_ij,
            1,
            labels.view(N, 1, 1).expand(N, C, A)
        ) 
        #print('labels',labels.shape)
        #print('NxW_kj',NxW_kj.shape) 

        #lin_layer1 = nn.Linear(cv_matrix.view(N,-1).size(1), A)
        #CV_temp = lin_layer1(cv_matrix.view(N,-1))
        CV_temp = cv_matrix
        #print('CV_temp',CV_temp.shape)
        sigma2 = ratio \
               * torch.mul(
                   (weight_m - NxW_kj).pow(2),
                   CV_temp.view(N, 1, A).expand(N, C, A),
               ).sum(2)
        #print('sigma2',sigma2.shape)
        #print('linear_layer.weight',linear_layer.weight.shape)
        #print('linear_layer.bias',linear_layer.bias.shape)
        logits = torch.nn.functional.linear(features, weight=linear_layer.weight, bias=linear_layer.bias)
        #print('logits',logits.shape)
        #logits = logits.view(N,-1)
        #print('logits',logits.shape)
        #lin_layer2 = nn.Linear(logits.view(N,-1).size(1), C)
        #logits =   lin_layer2(logits.view(N,-1))
        #print('logits',logits.shape)
        
        aug_logits = logits + 0.5 * sigma2

        return logits, aug_logits

    def forward(self, linear_layer, features, labels, ratio, cv_matrix):

        logits, aug_logits = self.isda_aug(linear_layer, features, labels, cv_matrix, ratio)

        loss = self.cross_entropy(aug_logits, labels)
        
        return loss, logits