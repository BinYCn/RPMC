import pdb
from monai.transforms import RandAdjustContrast
import numpy as np
import torch
import torch.nn.functional as F
import random


def augmentation(volume, aug_factor):
    return volume + aug_factor * torch.clip(torch.randn(*volume.shape).cuda() * 0.1, -0.2, 0.2)


def context_mask(img, mask_ratio, mask_pseudo):
    mask_pseudo = mask_pseudo.unsqueeze(0)
    channel, img_x, img_y, img_z = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(1, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x * mask_ratio), int(img_y * mask_ratio), int(
        img_z * mask_ratio)
    w = np.random.randint(0, img_x - patch_pixel_x)
    h = np.random.randint(0, img_y - patch_pixel_y)
    z = np.random.randint(0, img_z - patch_pixel_z)
    mask[w:w + patch_pixel_x, h:h + patch_pixel_y, z:z + patch_pixel_z] = 0
    loss_mask[:, w:w + patch_pixel_x, h:h + patch_pixel_y, z:z + patch_pixel_z] = \
        mask_pseudo[:, w:w + patch_pixel_x, h:h + patch_pixel_y, z:z + patch_pixel_z]
    return mask.long(), loss_mask.long()


def pseudo_labeling_from_most_confident_prediction(pred_1_1, pred_1_2, pred_2_1, pred_2_2,
                                                          max_1_1, max_1_2, max_2_1, max_2_2):
    prob_all_ex_3 = torch.stack([pred_1_1, pred_1_2, pred_2_1, pred_2_2], dim=2)  # bs, n_c, n_branch - 1, h, w, d
    max_all_ex_3 = torch.stack([max_1_1, max_1_2, max_2_1, max_2_2], dim=1)  # bs, n_branch - 1, h, w, d
    max_conf_each_branch_ex_3, _ = torch.max(prob_all_ex_3, dim=1)  # bs, n_branch - 1, h, w, d
    max_conf_ex_3, branch_id_max_conf_ex_3 = torch.max(max_conf_each_branch_ex_3, dim=1, keepdim=True)  # bs, h, w, d
    pseudo_12 = torch.gather(max_all_ex_3, dim=1, index=branch_id_max_conf_ex_3)[:, 0]
    max_pred, _ = torch.max(prob_all_ex_3, dim=2)

    return pseudo_12, max_pred


def generate_loss_mask(pseudo, pseudo1, pseudo_pre, pseudo1_pre, beta=0.5, unc=0.4):
    pseudo_pre, _ = torch.max(pseudo_pre, dim=1)
    pseudo1_pre, _ = torch.max(pseudo1_pre, dim=1)

    intersection = (pseudo == pseudo1).float()
    different = (((pseudo == 1) & (pseudo1 == 0) & (pseudo_pre >= (1 - unc))).float() +
                 ((pseudo == 0) & (pseudo1 == 1) & (pseudo1_pre >= (1 - unc))).float()) * beta
    loss_mask = intersection + different

    pseudo_union = ((pseudo == 1) | (pseudo1 == 1)).float()
    return loss_mask.long(), pseudo_union


def RPM(X, U, eval_net, K, T, alpha, mixup_mode, aug_factor, beta=0.5, unc=0.4):
    X_b = len(X)
    X_cap = [(augmentation(x[0], aug_factor), x[1]) for x in X]

    U_b = len(U)
    U_cap = U.repeat(K, 1, 1, 1, 1)
    U_cap += torch.clamp(torch.randn_like(U_cap) * 0.1, -0.2, 0.2)
    con = RandAdjustContrast(prob=1.0, gamma=(0.5, 1.5))

    U_cap[:K, :, :, :, :] = con(U_cap[:K, :, :, :, :])

    with torch.no_grad():
        pred1, pred2 = eval_net(U_cap)

        max_pred_1_1 = torch.argmax(pred1, dim=1)[:2]
        max_pred_1_2 = torch.argmax(pred1, dim=1)[2:]

        max_pred_2_1 = torch.argmax(pred2, dim=1)[:2]
        max_pred_2_2 = torch.argmax(pred2, dim=1)[2:]

        pred_sup_1_1 = F.softmax(pred1, dim=1)[:2]
        pred_sup_1_2 = F.softmax(pred1, dim=1)[2:]

        pred_sup_2_1 = F.softmax(pred2, dim=1)[:2]
        pred_sup_2_2 = F.softmax(pred2, dim=1)[2:]

        pseudo, max_pred = pseudo_labeling_from_most_confident_prediction(pred_sup_1_1, pred_sup_1_2,
                                                                                 pred_sup_2_1,
                                                                                 pred_sup_2_2,
                                                                                 max_pred_1_1, max_pred_1_2,
                                                                                 max_pred_2_1,
                                                                                 max_pred_2_2)

        pseudo1_pre = F.softmax((pred1[:2] + pred1[2:] + pred2[:2] + pred2[2:]) / 4, dim=1)
        pseudo1 = torch.argmax(pseudo1_pre, dim=1)

    loss_mask, pseudo_union = generate_loss_mask(pseudo, pseudo1, max_pred, pseudo1_pre, beta, unc)
    pseudo_label = pseudo
    guessed = pseudo_union.repeat(K, 1, 1, 1)
    loss_mask = loss_mask.repeat(K, 1, 1, 1)
    U_cap = list(zip(U_cap, guessed))

    x_mixup_mode, u_mixup_mode = mixup_mode[0], mixup_mode[1]

    if x_mixup_mode == '_':
        X_prime = X_cap
    else:
        raise ValueError('wrong mixup_mode')

    if u_mixup_mode == 'x':
        idxs = np.random.permutation(range(U_b * K)) % X_b
        U_prime = [cutmix_mask(U_cap[i], X_cap[idxs[i]], alpha, loss_mask[i]) for i in range(U_b * K)]
    elif u_mixup_mode == '_':
        U_prime = U_cap
    else:
        raise ValueError('wrong mixup_mode')

    return X_prime, U_prime, pseudo_label, pseudo_union


def cutmix_mask(s1, s2, alpha, mask):
    l = torch.distributions.Beta(alpha, alpha).sample()

    x1, p1 = s1
    x2, p2 = s2

    img_mask, loss_mask = context_mask(x1, l, mask)

    x = img_mask * x1 + (1 - img_mask) * x2
    p = img_mask * p1 + (1 - img_mask) * p2

    return (x, p, loss_mask.long())
