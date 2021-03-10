import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


def get_gaussian_delta(m_i, cov_i, m_j, cov_j, x_dim):
    # m_i, m_j : (x_dim,)
    # cov_i, cov_j : (x_dim, x_dim)

    cov_inv_i = torch.inverse(cov_i)
    cov_inv_j = torch.inverse(cov_j)

    cov_inv_ij = cov_inv_i + cov_inv_j
    cov_ij = torch.inverse(cov_inv_ij)
    m_ij = torch.mm(
        cov_ij,
        (torch.mm(cov_inv_i, m_i.unsqueeze(1)) + torch.mm(cov_inv_j, m_j.unsqueeze(1))),
    )  # (x_dim, 1)

    delta = 0.5 * (
        torch.mm(m_ij.t(), torch.mm(cov_inv_ij, m_ij))
        - (
            torch.mm(m_i.unsqueeze(0), torch.mm(cov_inv_i, m_i.unsqueeze(1)))
            + torch.mm(m_j.unsqueeze(0), torch.mm(cov_inv_j, m_j.unsqueeze(1)))
        )
        - torch.log(
            torch.det(cov_inv_ij) / (torch.det(cov_inv_i) * torch.det(cov_inv_j))
        )
        - x_dim * torch.log(torch.Tensor([2 * math.pi]).to(m_i))
    )
    return delta


def log_gaussian_mixture_cs_divergence(
    p_mean, p_cov, p_alpha, q_mean, q_cov, q_alpha, x_dim
):
    Kp = p_mean.size(0)
    Kq = q_mean.size(0)
    pq = 0
    pq_weight = torch.einsum("pi,qi->pq", p_alpha, q_alpha)
    for i in range(Kp):
        for j in range(Kq):
            pq += pq_weight[i, j] * torch.exp(
                get_gaussian_delta(p_mean[i], p_cov[i], q_mean[j], q_cov[j], x_dim)
            )

    pp = 0
    pp_weight = torch.einsum("pi,qi->pq", p_alpha, p_alpha)
    for i in range(Kp):
        for j in range(Kp):
            pp += pp_weight[i, j] * torch.exp(
                get_gaussian_delta(p_mean[i], p_cov[i], p_mean[j], p_cov[j], x_dim)
            )

    qq = 0
    qq_weight = torch.einsum("pi,qi->pq", q_alpha, q_alpha)
    for i in range(Kq):
        for j in range(Kq):
            qq += qq_weight[i, j] * torch.exp(
                get_gaussian_delta(q_mean[i], q_cov[i], q_mean[j], q_cov[j], x_dim)
            )

    return 2 * torch.log(pq) - torch.log(pp) - torch.log(qq)


@LOSSES.register_module()
class CSDivergenceLoss(nn.Module):
    """CS Divergence Loss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, reduction="mean", loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred_bboxes,
        pred_labels,
        gt_bboxes_list,
        gt_labels_list,
        weight=None,
        avg_factor=None,
    ):
        """Forward function of loss.

        Args:
            pred_bboxes (torch.Tensor): The box predicton. shape [bs, kp, 4]
            pred_labels (torch.Tensor): The label prediction. shape [bs, kp, num_classes + 1]
            gt_bboxes_list (list[Tensor]): The gt bboxes of each image. each Tensor shape [kg, 4]
            gt_labels_list (list[Tensor]): The gt labels of each image. each Tensor shape [kg,]
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        bs = pred_bboxes.size(0)
        num_pred = pred_bboxes.size(1)
        num_classes = pred_labels.size(2) - 1

        # Split class prob and confidence and compute alpha
        pred_conf_logits = pred_labels[:, :, -1]
        pred_class_logits = pred_labels[:, :, :-1]
        pred_alphas = torch.sigmoid(pred_conf_logits).unsqueeze(2) * F.softmax(
            pred_class_logits, dim=2
        )
        pred_alpha_list = list(pred_alphas)

        gt_alpha_list = []
        total_num_gt = 0
        for gt_labels in gt_labels_list:
            num_gt = gt_labels.size(0)
            total_num_gt += num_gt
            gt_alpha = torch.zeros(num_gt, num_classes).to(pred_alphas)
            gt_alpha[range(num_gt), gt_labels] = 1
            gt_alpha_list.append(gt_alpha)

        # change bboxes cxcywh to mean and covariance(diagonal) matrix
        pred_mean_list = list(pred_bboxes[:, :, :2])
        pred_covs = (torch.pow(pred_bboxes[:, :, 2:] / 2, 2)).unsqueeze(-1).repeat(
            1, 1, 1, 2
        ) * (torch.eye(2).unsqueeze(0).unsqueeze(0).repeat(bs, num_pred, 1, 1)).to(
            pred_bboxes
        )
        pred_cov_list = list(pred_covs)

        gt_mean_list = []
        gt_cov_list = []
        for gt_bboxes in gt_bboxes_list:
            num_gt = gt_bboxes.size(0)
            gt_mean_list.append(gt_bboxes[:, :2])
            gt_cov = (torch.pow(gt_bboxes[:, 2:] / 2, 2)).unsqueeze(-1).repeat(
                1, 1, 2
            ) * (torch.eye(2).unsqueeze(0).repeat(num_gt, 1, 1)).to(gt_bboxes)
            gt_cov_list.append(gt_cov)

        loss = 0
        for i in range(bs):
            loss -= log_gaussian_mixture_cs_divergence(
                gt_mean_list[i],
                gt_cov_list[i],
                gt_alpha_list[i],
                pred_mean_list[i],
                pred_cov_list[i],
                pred_alpha_list[i],
                x_dim=2,
            )

        if avg_factor is not None:
            loss = loss / avg_factor

        return loss
