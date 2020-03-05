import torch
import torch.nn as nn


def ohem_single(score, gt_score, training_mask):
    '''
    ohem for 1 sample. pos/neg = 1/3
    Args:
        score: model output(cls), shape=(h, w)
        gt_score: 0-1 gt, shape=(h, w)
        training_mask: 1 if valid area else 0, shape=(h, w)
    Return:
        ohem_mask: 1 if selected else 0, shape=(1, 1, h, w)
    '''
    pos_num = torch.sum(gt_score > 0.5) - torch.sum((gt_score > 0.5) & (training_mask <= 0.5))
    neg_num = torch.sum(gt_score < 0.5)
    if (pos_num == 0):
        # only neg pixel
        # 取前50%的负样本作为难样本
        neg_num = int(0.5 * neg_num.item())
        neg_score = score.reshape(-1)
        neg_score_sorted, _ = torch.sort(-neg_score) # 从大到小排序
        threshold = -neg_score_sorted[neg_num - 1] # score >= threshold的样本是hard样本
        ohem_mask = (score >= threshold)
        ohem_mask = ohem_mask.reshape(1, 1, ohem_mask.shape[0], ohem_mask.shape[1]).float()
        return ohem_mask

    neg_num = min(pos_num * 3, neg_num)
    if (neg_num == 0):
        # only pos pixel
        ohem_mask = training_mask
        ohem_mask = ohem_mask.reshape(1, 1, ohem_mask.shape[0], ohem_mask.shape[1]).float()
        return ohem_mask

    # pos and neg pixels in one sample
    neg_score = score[gt_score < 0.5] # 负样本的模型输出score
    neg_score_sorted, _ = torch.sort(-neg_score) # 从大到小排序
    threshold = -neg_score_sorted[neg_num - 1] # score >= threshold的样本是hard样本

    ohem_mask = ((score >= threshold) | (gt_score > 0.5)) & (training_mask > 0.5) # shape=(h, w)
    ohem_mask = ohem_mask.reshape(1, 1, ohem_mask.shape[0], ohem_mask.shape[1]).float()

    return ohem_mask


def ohem_batch(y_true_cls, y_pred_cls, training_mask):
    '''
    ohem for 1 batch. pos/neg = 1/3
    Args:
        y_true_cls: shape=(b, 1, h, w)
        y_pred_cls: shape=(b, 1, h, w)
        training_mask: shape=(b, 1, h, w)
    Return:
        ohem_masks: shape=(b, 1, h, w)
    '''
    ohem_masks = []
    for i in range(y_pred_cls.shape[0]):
        ohem_mask = ohem_single(y_pred_cls[i, 0, :, :],
                                y_true_cls[i, 0, :, :],
                                training_mask[i, 0, :, :])
        ohem_masks.append(ohem_mask)
    ohem_masks = torch.cat(ohem_masks, dim=0).float()

    return ohem_masks


# bs * channels * W * H
def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = torch.sum(y_true_cls * y_pred_cls * training_mask)
    union = torch.sum(y_true_cls * training_mask) + torch.sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)

    return loss


class LossFunc(nn.Module):
    def __init__(self):
        super(LossFunc, self).__init__()
        self.bceloss = nn.BCELoss(reduction='mean')
        return

    def forward(self, y_true_cls, y_pred_cls,
                y_true_geo, y_pred_geo,
                training_mask):
        ohem_masks = ohem_batch(y_true_cls, y_pred_cls, training_mask)
        classification_loss = dice_coefficient(y_true_cls, y_pred_cls, ohem_masks)
        # scale classification loss to match the iou loss part
        classification_loss *= 0.01

        # bce_loss = self.bceloss(y_pred_cls * ohem_masks, y_true_cls * ohem_masks) * 0.01

        # d1 -> top, d2->right, d3->bottom, d4->left
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(y_true_geo, 1, 1)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = torch.split(y_pred_geo, 1, 1)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = torch.min(d2_gt, d2_pred) + torch.min(d4_gt, d4_pred)
        h_union = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        L_AABB = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
        L_theta = 1 - torch.cos(theta_pred - theta_gt)
        L_g = L_AABB + 20 * L_theta

        return torch.mean(L_g * y_true_cls * training_mask) + classification_loss
