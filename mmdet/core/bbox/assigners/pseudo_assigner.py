import torch

from ..builder import BBOX_ASSIGNERS
from ..match_costs import build_match_cost
from ..transforms import bbox_cxcywh_to_xyxy
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@BBOX_ASSIGNERS.register_module()
class PseudoAssigner(BaseAssigner):
    """
    This class does not assign anything.
    Just get bboxes and gt_boxxes then return as assigned result form.
    All sample are regraded as positive sample

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:

    """

    def __init__(self):
        pass

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Nothing
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Nothing
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Nothing
            eps (int | float, optional): Nothing

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              1,
                                              dtype=torch.long)

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps=None, labels=None)
5