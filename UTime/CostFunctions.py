import torch
import torch.nn.functional as F

class WeightedLoss():
    def __init__(self, dl):
        self.dl = dl

        count_BL = 0
        count_all = 0
        for i, X, y in self.dl:
            count_BL += int(y.sum().item())
            count_all += y.shape[-1] * y.shape[-2]
        if count_BL == 0:
            raise Exception("There is no BL point in the whole data loader!")

        self.weight_BL = count_all / count_BL
        self.weight_not_BL = count_all / (count_all - count_BL)


class WeightedBCE(torch.nn.Module, WeightedLoss):
    def __init__(self, dl):
        super().__init__(dl)

    def forward(self, input, target):
        weights = target * (self.weight_BL - self.weight_not_BL) + self.weight_not_BL
        # gives an array of same dimension as target, with value weight_not_BL for not BL points,
        # and weight_BL for BL points
        loss = F.cross_entropy(input, target, weight=weights)
        # loss = torch.mean(weights*(input-target)**2)
        return loss


class WeightedMSE(torch.nn.Module, WeightedLoss):
    def __init__(self, dl):
        super().__init__(dl)

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
