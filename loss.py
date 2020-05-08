import torch
from torch.nn import functional as F

def focal_loss(y_pred, y_true, alpha = 0.25, gamma = 2, reduction = 'mean'):
    '''
    focal_loss = -alpha * (1 - pt) ^ gamma * log(pt)

    y_pred/y_true: [batch, H, W]

    Adapted from FacebookResearch focal_loss.py
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    '''

    y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)

    ce_loss = F.binary_cross_entropy(y_pred, y_true, reduction="none")
    p_t = 1 * y_pred * y_true + (1 - y_pred) * (1 - y_true)
    focal_loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
        focal_loss = alpha_t * focal_loss

    if reduction == "mean":
        focal_loss = focal_loss.mean()
    elif reduction == "sum":
        focal_loss = focal_loss.sum()
    else:
        raise Error('Wrong reduction defined for focal loss')

    return focal_loss

    # y_true_part, y_pred_part = y_true, y_pred

    # if torch.cuda.is_available():
    #     alpha_factor = torch.ones(y_true_part.shape).cuda() * alpha
    # else:
    #     alpha_factor = torch.ones(y_true_part.shape) * alpha

    # alpha_factor = torch.where(torch.eq(y_true_part, 1.), alpha_factor, 1. - alpha_factor)
    # focal_weight = torch.where(torch.eq(y_true_part, 1.), 1. - y_pred_part, y_pred_part)
    # focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

    # bce = -(y_true_part * torch.log(y_pred_part) + (1.0 - y_true_part) * torch.log(1.0 - y_pred_part))
    # cls_loss = focal_weight * bce

    # if torch.cuda.is_available():
    #     cls_loss = torch.where(torch.ne(y_true_part, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
    # else:
    #     cls_loss = torch.where(torch.ne(y_true_part, -1.0), cls_loss, torch.zeros(cls_loss.shape))

    # if reduction == 'sum':
    #     focal_loss = cls_loss.sum()
    # elif reduction == 'mean':
    #     focal_loss = cls_loss.mean()

    # y_pred = y_pred.reshape(y_pred.shape[0], -1)
    # y_true = y_true.reshape(y_true.shape[0], -1)

    # pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)

    # if reduction == 'sum':
    #     focal_loss = torch.sum(-alpha * (1 - pt) ** gamma * torch.log(pt))
    # elif reduction == 'mean':
    #     focal_loss = torch.mean(-alpha * (1 - pt) ** gamma * torch.log(pt))
    # else:
    #     raise Error('Wrong reduction defined for focal loss')

def contrasiveloss(original_features, path_features):
    pass

if __name__ == "__main__":

    y_pred = torch.rand((1,400,538))
    y_true = torch.ones((1,400,538)) 

    fl = focal_loss(y_pred, y_true)
    print(fl)
