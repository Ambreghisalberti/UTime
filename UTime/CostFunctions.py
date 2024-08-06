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
        input, target = torch.flatten(input), torch.flatten(target)
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
        input, target = torch.flatten(input), torch.flatten(target)
        return (1 - (input*target + epsilon).sum()/(input + target + epsilon).sum() -
                ((1-input)*(1-target) + epsilon).sum())/((1-input) + (1-target) + epsilon).sum()


class IntersectionOverUnion(torch.nn.Module):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        super(IntersectionOverUnion, self).__init__()

    def forward(self, input, target):
        input, target = torch.flatten(input), torch.flatten(target)
        return - (input*target).sum() / ((input*(1-target)).sum()+target.sum())

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0.1, alpha=0.5):
        self.alpha = alpha
        self.gamma = gamma
        super(FocalLoss, self).__init__()

    def forward(self, input, target):
        input, target = torch.flatten(input), torch.flatten(target)
        p_t = input*target + (1-input)*(1-target)
        alpha_t = self.alpha*(1-target) + (1-self.alpha)*target
        return - (alpha_t * (1-p_t)**self.gamma * torch.log(p_t)).sum()


class WeightedByDistanceMP(torch.nn.Module):
    def __init__(self):
        super(WeightedByDistanceMP, self).__init__()

    def forward(self, input, target, delta=0.1):
        input, target = torch.flatten(input), torch.flatten(target)
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

# Need to check that it's convex and derivable (= useful for optimization)
class maxF1(torch.nn.Module):
    def __init__(self):
        super(maxF1, self).__init__()

    def forward_one_class(self, input, target):
        # target = target.numpy().flatten()
        # input = input.detach().numpy().flatten()

        F1s = []
        thresholds = np.linspace(0, 1, 1000)
        for threshold in thresholds:
            pred_class = input > threshold

            TP, FP, FN = 0, 0, 0
            TP += (target * pred_class).sum()  # target = 1 and pred = 1
            FP += ((1 - target) * pred_class).sum()  # target = 0 and pred_class = 1
            FN += (target * (1 - pred_class)).sum()  # target = 1 and pred_class = 0
            #TP, FP, FN = TP.item(), FP.item(), FN.item()

            if (TP + (FN + FP) / 2) == 0:
                F1 = 0
            else:
                F1 = TP / (TP + (FN + FP) / 2)

            F1s.append(F1)

        F1s = np.array(F1s)

        return torch.Tensor([1 - np.max(F1s)])

    def forward(self,input,target):
        _, nb_classes, _, _ = input.shape
        loss = 0

        for i in range(nb_classes):
            ins = input[:, i, :, :]
            targ = target[:, i, :, :]
            loss += self.forward_one_class(torch.flatten(ins), torch.flatten(targ)).double()

        loss = loss / nb_classes
        return loss
    '''
    def roc_star_loss(self, input, target, gamma, _epoch_true, epoch_pred):
        """
        Nearly direct loss function for AUC.
        See article,
        C. Reiss, "Roc-star : An objective function for ROC-AUC that actually works."
        https://github.com/iridiumblue/articles/blob/master/roc_star.md
            target: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
            input: `Tensor` . Predictions.
            gamma  : `Float` Gamma, as derived from last epoch.
            _epoch_true: `Tensor`.  Targets (labels) from last epoch.
            epoch_pred : `Tensor`.  Predicions from last epoch.
        """
        # convert labels to boolean
        target = (target >= 0.50)
        epoch_true = (_epoch_true >= 0.50)

        # if batch is either all true or false return small random stub value.
        if torch.sum(target) == 0 or torch.sum(target) == target.shape[0]:
            return torch.sum(input) * 1e-8

        pos = input[target]
        neg = input[~target]

        epoch_pos = epoch_pred[epoch_true]
        epoch_neg = epoch_pred[~epoch_true]

        # Take random subsamples of the training set, both positive and negative.
        max_pos = 1000  # Max number of positive training samples
        max_neg = 1000  # Max number of positive training samples
        cap_pos = epoch_pos.shape[0]
        cap_neg = epoch_neg.shape[0]
        epoch_pos = epoch_pos[torch.rand_like(epoch_pos) < max_pos / cap_pos]
        epoch_neg = epoch_neg[torch.rand_like(epoch_neg) < max_neg / cap_pos]

        ln_pos = pos.shape[0]
        ln_neg = neg.shape[0]

        # sum positive batch elements against (subsampled) negative elements
        if ln_pos > 0:
            pos_expand = pos.view(-1, 1).expand(-1, epoch_neg.shape[0]).reshape(-1)
            neg_expand = epoch_neg.repeat(ln_pos)

            diff2 = neg_expand - pos_expand + gamma
            l2 = diff2[diff2 > 0]
            m2 = l2 * l2
            len2 = l2.shape[0]
        else:
            m2 = torch.tensor([0], dtype=torch.float).cuda()
            len2 = 0

        # Similarly, compare negative batch elements against (subsampled) positive elements
        if ln_neg > 0:
            pos_expand = epoch_pos.view(-1, 1).expand(-1, ln_neg).reshape(-1)
            neg_expand = neg.repeat(epoch_pos.shape[0])

            diff3 = neg_expand - pos_expand + gamma
            l3 = diff3[diff3 > 0]
            m3 = l3 * l3
            len3 = l3.shape[0]
        else:
            m3 = torch.tensor([0], dtype=torch.float).cuda()
            len3 = 0

        if (torch.sum(m2) + torch.sum(m3)) != 0:
            res2 = torch.sum(m2) / max_pos + torch.sum(m3) / max_neg
            # code.interact(local=dict(globals(), **locals()))
        else:
            res2 = torch.sum(m2) + torch.sum(m3)

        res2 = torch.where(torch.isnan(res2), torch.zeros_like(res2), res2)

        return res2
    '''

class roc_star(torch.nn.Module):
    def __init__(self, p=2):
        super(roc_star, self).__init__()
        self.p = p
        self.gamma = 0

    def forward(self, input, target):
        target = target > 0.5
        pos = torch.flatten(input[target])
        neg = torch.flatten(input[~target])

        size_pos = len(pos)
        size_neg = len(neg)
        pos = pos.repeat(len(neg))
        neg = neg.repeat(size_pos).reshape((size_pos, size_neg)).transpose(0, 1).flatten()
        loss = ((neg + self.gamma - pos) ** self.p).sum()

        self.gamma = max(1,torch.mean(pos-neg).item()*1.1)
        return loss

    '''
    def forward(self,input,target):
        _, nb_classes, _, _ = input.shape
        loss = 0

        for i in range(nb_classes):
            ins = input[:, i, :, :]
            targ = target[:, i, :, :]
            loss += self.forward_one_class(torch.flatten(ins), torch.flatten(targ)).double()

        loss = loss / nb_classes
        return loss
    '''


