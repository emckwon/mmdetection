import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.runner import force_fp32

from mmdet.core import (
    bbox_cxcywh_to_xyxy,
    bbox_xyxy_to_cxcywh,
    build_assigner,
    build_sampler,
    multi_apply,
    reduce_mean,
)
from mmdet.models.utils import FFN, build_positional_encoding, build_transformer
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead


@HEADS.register_module()
class CSDivTransformerHead(AnchorFreeHead):
    """Implements the DETR transformer head.

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (dict, optional): Config for transformer.
        positional_encoding (dict, optional): Config for position encoding.
        loss_cls (dict, optional): Config of the classification loss.
            Default `CrossEntropyLoss`.
        loss_bbox (dict, optional): Config of the regression loss.
            Default `L1Loss`.
        loss_iou (dict, optional): Config of the regression iou loss.
            Default `GIoULoss`.
        tran_cfg (dict, optional): Training config of transformer head.
        test_cfg (dict, optional): Testing config of transformer head.

    Example:
        >>> import torch
        >>> self = TransformerHead(80, 2048)
        >>> x = torch.rand(1, 2048, 32, 32)
        >>> mask = torch.ones(1, 32, 32).to(x.dtype)
        >>> mask[:, :16, :15] = 0
        >>> all_cls_scores, all_bbox_preds = self(x, mask)
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        num_fcs=2,
        transformer=dict(
            type="Transformer",
            embed_dims=256,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            feedforward_channels=2048,
            dropout=0.1,
            act_cfg=dict(type="ReLU", inplace=True),
            norm_cfg=dict(type="LN"),
            num_fcs=2,
            pre_norm=False,
            return_intermediate_dec=True,
        ),
        positional_encoding=dict(
            type="SinePositionalEncoding", num_feats=128, normalize=True
        ),
        loss_cs=dict(
            type="CSDivergenceLoss",
        ),
        train_cfg=dict(),
        test_cfg=dict(max_per_img=100),
        **kwargs,
    ):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(AnchorFreeHead, self).__init__()
        assert "embed_dims" in transformer and "num_feats" in positional_encoding
        num_feats = positional_encoding["num_feats"]
        embed_dims = transformer["embed_dims"]
        assert num_feats * 2 == embed_dims, (
            "embed_dims should"
            f" be exactly 2 times of num_feats. Found {embed_dims}"
            f" and {num_feats}."
        )
        assert test_cfg is not None and "max_per_img" in test_cfg

        if train_cfg:
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type="PseudoSampler")
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes + 1
        self.in_channels = in_channels
        self.num_fcs = num_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.embed_dims = embed_dims
        self.num_query = test_cfg["max_per_img"]
        self.fp16_enabled = False
        self.loss_cs = build_loss(loss_cs)
        self.act_cfg = transformer.get("act_cfg", dict(type="ReLU", inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.transformer = build_transformer(transformer)
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        self.input_proj = Conv2d(self.in_channels, self.embed_dims, kernel_size=1)
        self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        self.reg_ffn = FFN(
            self.embed_dims,
            self.embed_dims,
            self.num_fcs,
            self.act_cfg,
            dropout=0.0,
            add_residual=False,
        )
        self.fc_reg = Linear(self.embed_dims, 4)
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

    def init_weights(self, distribution="uniform"):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.
        super(AnchorFreeHead, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, feats, img_metas):
        """Forward function.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Outputs for all scale levels.

                - all_cls_scores_list (list[Tensor]): Classification scores \
                    for each scale level. Each is a 4D-tensor with shape \
                    [nb_dec, bs, num_query, cls_out_channels]. Note \
                    `cls_out_channels` should includes background.
                - all_bbox_preds_list (list[Tensor]): Sigmoid regression \
                    outputs for each scale level. Each is a 4D-tensor with \
                    normalized coordinate format (cx, cy, w, h) and shape \
                    [nb_dec, bs, num_query, 4].
        """
        num_levels = len(feats)
        img_metas_list = [img_metas for _ in range(num_levels)]
        return multi_apply(self.forward_single, feats, img_metas_list)

    def forward_single(self, x, img_metas):
        """ "Forward function for a single feature level.

        Args:
            x (Tensor): Input feature from backbone's single stage, shape
                [bs, c, h, w].
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, 4].
        """
        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        batch_size = x.size(0)
        input_img_h, input_img_w = img_metas[0]["batch_input_shape"]
        masks = x.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]["img_shape"]
            masks[img_id, :img_h, :img_w] = 0

        x = self.input_proj(x)
        # interpolate masks to have the same spatial shape with x
        masks = (
            F.interpolate(masks.unsqueeze(1), size=x.shape[-2:])
            .to(torch.bool)
            .squeeze(1)
        )
        # position encoding
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
        # outs_dec: [nb_dec, bs, num_query, embed_dim]
        outs_dec, _ = self.transformer(x, masks, self.query_embedding.weight, pos_embed)

        all_cls_scores = self.fc_cls(outs_dec)
        all_bbox_preds = self.fc_reg(self.activate(self.reg_ffn(outs_dec))).sigmoid()
        return all_cls_scores, all_bbox_preds

    @force_fp32(apply_to=("all_cls_scores_list", "all_bbox_preds_list"))
    def loss(
        self,
        all_cls_scores_list,
        all_bbox_preds_list,
        gt_bboxes_list,
        gt_labels_list,
        img_metas,
        gt_bboxes_ignore=None,
    ):
        """ "Loss function.

        Only outputs from the last feature level are used for computing
        losses by default.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # NOTE defaultly only the outputs from the last feature scale is used.
        all_cls_scores = all_cls_scores_list[-1]
        all_bbox_preds = all_bbox_preds_list[-1]
        assert (
            gt_bboxes_ignore is None
        ), "Only supports for gt_bboxes_ignore setting to None."

        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        # all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        # losses_cs = multi_apply(
        #     self.loss_single,
        #     all_cls_scores,
        #     all_bbox_preds,
        #     all_gt_bboxes_list,
        #     all_gt_labels_list,
        #     img_metas_list,
        # )
        losses_cs = self.loss_single(
            all_cls_scores[-1],
            all_bbox_preds[-1],
            gt_bboxes_list,
            gt_labels_list,
            img_metas,
        )

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict["loss_cs"] = losses_cs
        # loss from other decoder layers
        # num_dec_layer = 0
        # for loss_cs_i in losses_cs[:-1]:
        #     loss_dict[f"d{num_dec_layer}.loss_cs"] = loss_cs_i
        #     num_dec_layer += 1
        return loss_dict

    def loss_single(
        self,
        cls_scores,
        bbox_preds,
        gt_bboxes_list,
        gt_labels_list,
        img_metas,
    ):
        """ "Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, list(bbox_preds)):
            img_h, img_w, _ = img_meta["img_shape"]
            factor = (
                bbox_pred.new_tensor([img_w, img_h, img_w, img_h])
                .unsqueeze(0)
                .repeat(bbox_pred.size(0), 1)
            )
            factors.append(factor)
        factors = torch.stack(factors, 0)
        bbox_preds = bbox_preds / factors

        bbox_preds = bbox_xyxy_to_cxcywh(bbox_preds.view(-1, 4)).view(num_imgs, -1, 4)

        gt_bboxes_list_xywh = []
        for img_meta, gt_bboxes in zip(img_metas, gt_bboxes_list):
            img_h, img_w, _ = img_meta["img_shape"]
            factor = (
                gt_bboxes.new_tensor([img_w, img_h, img_w, img_h])
                .unsqueeze(0)
                .repeat(gt_bboxes.size(0), 1)
            )
            gt_bboxes_list_xywh.append(bbox_xyxy_to_cxcywh(gt_bboxes / factor))

        # cls_scores [bs, Kp, num_classes + 1]
        # bbox_preds [bs, Kp, 4]
        # gt_bboxes_list [list([Kg, 4])]
        # gt_labels_list [list([Kg])]
        # tic = time.time()
        # print(f"tic: {tic}")
        loss_cs = self.loss_cs(
            bbox_preds,
            cls_scores,
            gt_bboxes_list_xywh,
            gt_labels_list,
            avg_factor=num_imgs,
        )
        # toc = time.time()
        # print(f"toc: {toc}, diff: {toc -tic}")

        return loss_cs

    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(
        self,
        x,
        img_metas,
        gt_bboxes,
        gt_labels=None,
        gt_bboxes_ignore=None,
        proposal_cfg=None,
        **kwargs,
    ):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(x, img_metas)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    @force_fp32(apply_to=("all_cls_scores_list", "all_bbox_preds_list"))
    def get_bboxes(
        self, all_cls_scores_list, all_bbox_preds_list, img_metas, rescale=False
    ):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Defalut False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        # NOTE defaultly only using outputs from the last feature level,
        # and only the ouputs from the last decoder layer is used.
        cls_scores = all_cls_scores_list[-1][-1]
        bbox_preds = all_bbox_preds_list[-1][-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]["img_shape"]
            scale_factor = img_metas[img_id]["scale_factor"]
            proposals = self._get_bboxes_single(
                cls_score, bbox_pred, img_shape, scale_factor, rescale
            )
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(
        self, cls_score, bbox_pred, img_shape, scale_factor, rescale=False
    ):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_query, 4].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5], \
                    where the first 4 columns are bounding box positions \
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores \
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with \
                    shape [num_query].
        """
        assert len(cls_score) == len(bbox_pred)
        # exclude background
        scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)
        return det_bboxes, det_labels

    def get_targets(
        self,
        cls_scores_list,
        bbox_preds_list,
        gt_bboxes_list,
        gt_labels_list,
        img_metas,
        gt_bboxes_ignore_list=None,
    ):
        """
        Nothing
        """
        return None

    def _get_target_single(
        self,
        cls_score,
        bbox_pred,
        gt_bboxes,
        gt_labels,
        img_meta,
        gt_bboxes_ignore=None,
    ):
        """
        Do nothing
        """

        return None