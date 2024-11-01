from dis import dis
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F 
import gc
def _entr(dist):
    dist = dist + 1e-7
    en_z_M = torch.mul(
            -1*dist, torch.log(dist)
        ) 
    en_z = torch.sum(
            torch.sum(en_z_M, dim=-1),
            dim=-1)/en_z_M.size(-2)
    return en_z

class RIBLoss(nn.Module):
    def __init__(self, lam):
        super().__init__()
        self.lam = lam 
        self.cross = nn.CrossEntropyLoss()
        
    def forward(self, dist, outputs, targets):
        loss_cross = self.cross(outputs, targets)
        loss_entr = _entr(dist)
        loss = loss_cross - self.lam*loss_entr
        return loss, loss_cross, loss_entr
    
class VAELoss(nn.Module):
    def __init__(self, lam):
        super().__init__()
        self.lam = lam 
        self.recon = nn.MSELoss()
        
    def forward(self, dist, outputs, targets):
        loss_recon = self.recon(outputs, targets)
        loss_entr = _entr(dist)
        loss = loss_recon - self.lam*loss_entr
        return loss, loss_recon, loss_entr
def MI(outputs_target):
    batch_size = outputs_target.size(0)
    softmax_outs_t = nn.Softmax(dim=1)(outputs_target)
    avg_softmax_outs_t = torch.sum(softmax_outs_t, dim=0) / float(batch_size)
    log_avg_softmax_outs_t = torch.log(avg_softmax_outs_t)
    item1 = -torch.sum(avg_softmax_outs_t * log_avg_softmax_outs_t)
    item2 = -torch.sum(softmax_outs_t * torch.log(softmax_outs_t)) / float(batch_size)
    return item1 - item2

def CalculateMean(features, labels, class_num):
    N = features.size(0)
    C = class_num
    A = features.size(1)

    avg_CxA = torch.zeros(C, A)
    NxCxFeatures = features.view(N, 1, A).expand(N, C, A)

    onehot = torch.zeros(N, C)
    onehot.scatter_(1, labels.view(-1, 1), 1)
    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

    Amount_CxA = NxCxA_onehot.sum(0)
    Amount_CxA[Amount_CxA == 0] = 1.0

    del onehot
    gc.collect()
    for c in range(class_num):
        c_temp = NxCxFeatures[:, c, :].mul(NxCxA_onehot[:, c, :])
        c_temp = torch.sum(c_temp, dim=0)
        avg_CxA[c] = c_temp / Amount_CxA[c]
    return avg_CxA.detach()

def Calculate_CV(features, labels, ave_CxA, class_num):
    N = features.size(0)
    C = class_num
    A = features.size(1)

    var_temp = torch.zeros(C, A, A)
    NxCxFeatures = features.view(N, 1, A).expand(N, C, A)

    onehot = torch.zeros(N, C)
    onehot.scatter_(1, labels.view(-1, 1), 1)
    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

    Amount_CxA = NxCxA_onehot.sum(0)
    Amount_CxA[Amount_CxA == 0] = 1
    Amount_CxAxA = Amount_CxA.view(C, A, 1).expand(C, A, A)
    del Amount_CxA, onehot
    gc.collect()

    avg_NxCxA = ave_CxA.expand(N, C, A)
    for c in range(C):
        features_by_sort_c = NxCxFeatures[:, c, :].mul(NxCxA_onehot[:, c, :])
        avg_by_sort_c = avg_NxCxA[:, c, :].mul(NxCxA_onehot[:, c, :])
        var_temp_c = features_by_sort_c - avg_by_sort_c
        var_temp[c] = torch.mm(var_temp_c.permute(1,0), var_temp_c).div(Amount_CxAxA[c])
    return var_temp.detach()

class Cls_Loss(nn.Module):
    def __init__(self, class_num):
        super(Cls_Loss, self).__init__()
        self.class_num = class_num
        self.cross_entropy = nn.CrossEntropyLoss()

    def aug(self, s_mean_matrix, t_mean_matrix, fc, features, y_s, labels_s, t_cv_matrix, Lambda):
        N = features.size(0)
        A = features.size(1)

        weight_m = list(fc.parameters())[0]

    # Print the current shape of weight_m for debugging
        print(f"Shape of weight_m: {weight_m.shape}, Total elements: {weight_m.numel()}")
       
        NxW_ij = weight_m
        NxW_kj = torch.gather(NxW_ij, 1, labels_s.view(N, -1, -1))

        t_CV_temp = t_cv_matrix[labels_s]

        sigma2 = Lambda * torch.bmm(torch.bmm(NxW_ij - NxW_kj, t_CV_temp), (NxW_ij - NxW_kj).permute(0, 2, 1))
        sigma2 = sigma2.mul.sum(2)

        sourceMean_NxA = s_mean_matrix[labels_s]
        targetMean_NxA = t_mean_matrix[labels_s]
        dataMean_NxA = (targetMean_NxA - sourceMean_NxA)
        dataMean_NxAx1 = dataMean_NxA.expand(1, N, A).permute(1, 2, 0)

        del t_CV_temp, sourceMean_NxA, targetMean_NxA, dataMean_NxA
        gc.collect()

        dataW_NxA = NxW_ij - NxW_kj
        dataW_x_detaMean_Nx1 = torch.bmm(dataW_NxA, dataMean_NxAx1)
        datW_x_detaMean_N = dataW_x_detaMean_Nx1

        aug_result = y_s + 0.5 * sigma2 + Lambda * datW_x_detaMean_N
        return aug_result

    def forward(self, fc, features_source: torch.Tensor, y_s, labels_source, Lambda, mean_source, mean_target, covariance_target):
        aug_y = self.aug(mean_source, mean_target, fc, features_source, y_s, labels_source, covariance_target, Lambda)
        loss = self.cross_entropy(aug_y, labels_source)
        return loss
        
if __name__ == '__main__':
    targets = torch.randint(0,5, size=(3,))
    targets_ = F.one_hot(targets, num_classes=5).type(torch.float)*1000
    outputs = torch.randn((3,5))
    model = nn.CrossEntropyLoss()
    out = model(targets_, targets)
    out_2 = model(outputs, targets)
    print(targets_)
    print(targets)
    print(out)
    print(out_2)
    
    # print(targets)
    # print(targets_)
    
    # outputs = torch.randn((3,5))
    
    # dist = torch.randn((3, 1000))
    # dist = F.softmax(dist, dim=-1)
    
    # criterion = SemmLoss(0.1)
    # loss, loss_cross, loss_entr = criterion(dist, targets_, targets)
    # print(loss)
    # print(loss_cross)
    # print(loss_entr)
    
    