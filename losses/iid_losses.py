import torch
import torch.nn as nn
import torch.sparse
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

#losses taken from NIID-Net
class L1ImageGradientLoss(nn.Module):
    def __init__(self, step=2):
        super(L1ImageGradientLoss, self).__init__()
        self.step = step

    def forward(self, pred, target, mask):
        step = self.step

        N = torch.sum(mask)
        diff = pred - target
        diff = torch.mul(diff, mask)

        v_gradient = torch.abs(diff[:, :, 0:-step, :] - diff[:, :, step:, :])
        v_mask = torch.mul(mask[:, :, 0:-step, :], mask[:, :, step:, :])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(diff[:, :, :, 0:-step] - diff[:, :, :, step:])
        h_mask = torch.mul(mask[:, :, :, 0:-step], mask[:, :, :, step:])
        h_gradient = torch.mul(h_gradient, h_mask)

        gradient_loss = (torch.sum(h_gradient) + torch.sum(v_gradient)) / 2.0
        gradient_loss = gradient_loss / (N + 1e-6)

        return gradient_loss

class MultiScaleGradientLoss(nn.Module):
    def __init__(self, scale_step=2):
        super(MultiScaleGradientLoss, self).__init__()
        self.gradient_loss = L1ImageGradientLoss(step=1)
        self.step = scale_step

    def forward(self, pred, target, mask):
        step = self.step

        prediction_1 = pred[:,:,::step,::step]
        prediction_2 = prediction_1[:,:,::step,::step]
        prediction_3 = prediction_2[:,:,::step,::step]

        mask_1 = mask[:,:,::step,::step]
        mask_2 = mask_1[:,:,::step,::step]
        mask_3 = mask_2[:,:,::step,::step]

        gt_1 = target[:,:,::step,::step]
        gt_2 = gt_1[:,:,::step,::step]
        gt_3 = gt_2[:,:,::step,::step]

        final_loss = self.gradient_loss(pred, target, mask)
        final_loss += self.gradient_loss(prediction_1, gt_1, mask_1)
        final_loss += self.gradient_loss(prediction_2, gt_2, mask_2)
        final_loss += self.gradient_loss(prediction_3, gt_3, mask_3)
        return final_loss

class ReflectConsistentLoss(nn.Module):
    def __init__(self, sample_num_per_area=1, split_areas=(3, 3)):
        super(ReflectConsistentLoss, self).__init__()
        self.sample_num = sample_num_per_area
        self.split_x, self.split_y = split_areas

        self.cos_similar = nn.CosineSimilarity(dim=1, eps=0)

    def random_relative_loss(self, random_pixel, pred_R, gt_R, target_rgb, mask):
        x, y = random_pixel

        samples_gt_R = gt_R[:, :, x:x+1, y:y+1]
        samples_rgb = target_rgb[:, :, x:x+1, y:y+1]
        samples_pred_R = pred_R[:, :, x:x+1, y:y+1]
        samples_mask = mask[:, :, x:x+1, y:y+1]

        rel_gt_R = gt_R - samples_gt_R
        # rel_rgb = target_rgb - samples_rgb
        rel_pred_R = pred_R - samples_pred_R

        # Compute similarity
        mask_rel = mask * samples_mask
        mean_rel_gt_R = torch.mean(torch.abs(rel_gt_R), dim=1, keepdim=True).repeat(1, 3, 1, 1) * mask_rel
        mask_rel[mean_rel_gt_R >= 0.2] = 0

        diff = (rel_pred_R - rel_gt_R) * mask_rel
        loss = torch.sum(torch.abs(diff)) / (torch.sum(mask_rel)+1e-6)
        return loss

    def forward(self, pred_R, gt_R, target_rgb, mask):
        device_id = pred_R.get_device()
        total_loss = Variable(torch.zeros(1).type(torch.FloatTensor)).cuda(device_id)

        x_spaces = np.linspace(0, gt_R.size(2), self.split_x+1, endpoint=True, dtype=np.int)
        y_spaces = np.linspace(0, gt_R.size(3), self.split_y+1, endpoint=True, dtype=np.int)
        for idx in range(self.sample_num):
            for idx_x in range(x_spaces.size - 1):
                for idx_y in range(y_spaces.size - 1):
                    x = np.random.randint(x_spaces[idx_x], x_spaces[idx_x+1], 1)[0]
                    y = np.random.randint(y_spaces[idx_y], y_spaces[idx_y+1], 1)[0]
                    total_loss += self.random_relative_loss((x, y), pred_R, gt_R, target_rgb, mask)
        return total_loss / (self.sample_num * self.split_x * self.split_y)