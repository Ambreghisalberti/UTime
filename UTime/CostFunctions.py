import torch
import torch.nn.functional as F
import numpy as np

def size_tensor(tensor):
    size=1
    for dim in tensor.size():
        size *= dim
    return size


class WeightedLoss(torch.nn.Module):
    def __init__(self, dl):
        super(WeightedLoss, self).__init__()
        self.dl = dl

        count_BL = 0
        count_all = 0
        for i, X, y in self.dl:
            count_BL += int(y.sum().item())
            count_all += size_tensor(y)
        if count_BL == 0:
            raise Exception("There is no BL point in the whole data loader!")

        self.weight_BL = count_all / count_BL
        self.weight_not_BL = count_all / (count_all - count_BL)


class WeightedBCE(WeightedLoss):

    def forward(self, input, target):
        weights = target * (self.weight_BL - self.weight_not_BL) + self.weight_not_BL
        # gives an array of same dimension as target, with value weight_not_BL for not BL points,
        # and weight_BL for BL points
        loss = F.cross_entropy(input, target, weight=weights)
        # loss = torch.mean(weights*(input-target)**2)
        return loss


class WeightedMSE(WeightedLoss):

    def forward(self, input, target):
        weights = target * (self.weight_BL - self.weight_not_BL) + self.weight_not_BL
        # gives an array of same dimension as target, with value weight_not_BL for not BL points,
        # and weight_BL for BL points
        # loss = F.cross_entropy(input, target, weight = torch.tensor([self.weight_not_BL,self.weight_BL]))
        loss = torch.mean(weights * (input - target) ** 2)
        return loss


class DiceLoss(torch.nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target, epsilon=0.000000001):
        return (1 - (input*target + epsilon).sum()/(input + target + epsilon).sum() -
                ((1-input)*(1-target) + epsilon).sum())/((1-input) + (1-target) + epsilon).sum()


class IntersectionOverUnion(torch.nn.Module):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        super(IntersectionOverUnion, self).__init__()

    def forward(self, input, target):
        return - (input*target).sum() / ((input*(1-target)).sum()+target.sum())

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=3, alpha=1):
        self.alpha = alpha
        self.gamma = gamma
        super(FocalLoss, self).__init__()

    def forward(self, input, target):
        p_t = input*target + (1-input)*(1-target)
        return - (self.alpha * (1-p_t)**self.gamma * torch.log(p_t)).sum()


class WeightedByDistanceMP(torch.nn.Module):
    def __init__(self):
        super(WeightedByDistanceMP, self).__init__()

    def forward(self, input, target, delta=0.1):
        # Need to change all the rest to pass on the distance to MP with the inputs, or the i (window id), to make it general.
        # For now, this cost function can only work if relative distance to MP is given as the last feature of windows,
        # for input of dimension 4.
        if isinstance(input, list):
            moments, sepctro = input
        assert len(input.shape)==4, ("The input should be a 4D tensor, otherwise our implementation of ponderation "
                                     "by distance to MP might not give the wanted results.")
        distances_to_MP = input[:,-1]
        FN_ponderation = np.exp(-(distances_to_MP/delta)**2)*0.9 + 0.1
        FP_ponderation = 1 - np.exp(-(distances_to_MP/delta)**2)*0.5

        FN = input*(1-target)
        FP = (1-input)*target

        return (FN*FN_ponderation + FP*FP_ponderation).mean()
