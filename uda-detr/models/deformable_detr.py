# ------------------------------------------------------------------------
# Modified by Wei-Jie Huang
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import itertools
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
from .utils import GradientReversal
import copy
from torch.nn.functional import binary_cross_entropy_with_logits, l1_loss, mse_loss
from .CBAM import CBAM,CoordAtt,KAN
from .DA_utils import decompose_features,grad_reverse,FCDiscriminator_img,get_prototype_class_wise

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False,
                 backbone_align=False, space_align=False, channel_align=False, instance_align=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.uda = backbone_align or space_align or channel_align or instance_align
        self.backbone_align = backbone_align
        self.space_align = space_align
        self.channel_align = channel_align
        self.instance_align = instance_align

        # self.cbam = CBAM(3)
        # self.ca = CoordAtt(256,256)


        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        if backbone_align:
            self.grl = GradientReversal()
            self.backbone_D = MLP(hidden_dim, hidden_dim, 1, 3)
            for layer in self.backbone_D.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)
        if space_align:
            self.space_D = MLP(hidden_dim, hidden_dim, 1, 3)
            for layer in self.space_D.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)
        if channel_align:
            self.channel_D = MLP(hidden_dim, hidden_dim, 1, 3)
            for layer in self.channel_D.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)
        if instance_align:
            self.instance_D = MLP(hidden_dim, hidden_dim, 1, 3)
            for layer in self.instance_D.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)

    ###########################################################
        if self.training:
            #空域
            self.D_img = FCDiscriminator_img(256) # Need to know the channel
            #原型
            self.global_proto = torch.zeros(self.num_classes, 256).cuda(non_blocking=True)
            self.Amount = torch.zeros(self.num_classes).cuda(non_blocking=True)
            self.Proto_D = MLP(hidden_dim, hidden_dim, 1, 3)

    @staticmethod
    def get_mask_list(mask_list, mask_ratio):
        """
              生成掩码列表。
              """
        mae_mask_list = copy.deepcopy(mask_list)
        for i in range(len(mask_list)):
            mae_mask_list[i] = torch.rand(mask_list[i].shape).to(mask_list[i].device) < mask_ratio
        return mae_mask_list


    def forward(self, samples: NestedTensor,enable_mae=False, mask_ratio=0.8):
        # samples = self.cbam(samples)
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)


        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            src = self.input_proj[l](src)
#######################################################
            # src = self.cbam(src)
            # src = self.ca(src)



####################################################
            srcs.append(src)
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, da_output = \
            self.transformer(srcs, masks, pos, query_embeds,enable_mae=False)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)


        if self.training and self.uda:
            B = outputs_class.shape[1]
            outputs_class = outputs_class[:, :B//2]
            outputs_coord = outputs_coord[:, :B//2]
            if self.two_stage:
                enc_outputs_class = enc_outputs_class[:B//2]
                enc_outputs_coord_unact = enc_outputs_coord_unact[:B//2]
            if self.backbone_align:
                da_output['backbone'] = torch.cat([self.backbone_D(self.grl(src.flatten(2).transpose(1, 2))) for src in srcs], dim=1)
            if self.space_align:
                da_output['space_query'] = self.space_D(da_output['space_query'])
            if self.channel_align:
                da_output['channel_query'] = self.channel_D(da_output['channel_query'])
            if self.instance_align:
                da_output['instance_query'] = self.instance_D(da_output['instance_query'])

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        # MAE branch
        if enable_mae:
            assert self.transformer.mae_decoder is not None
            mae_mask_list = self.get_mask_list(masks, mask_ratio)
            mae_src_list = srcs
            mae_output = self.transformer(
                mae_src_list,
                mae_mask_list,
                pos,
                query_embeds,
                enable_mae=enable_mae,
            )
            out['mae_output'] = mae_output
            out['features'] = [features[mae_idx].tensors.detach() for mae_idx in [2]]


        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        if self.training and self.uda:
            out['da_output'] = da_output

        ##############################################################################缝合怪############
        # ----DA Process
        if self.training:
            # da_output = {}
            #First.==============backbone 特征对齐======================
            D_img_out = []
            for src in srcs:  #不同层级
                src = grad_reverse(src)  #GRL
                D_img_out_sub = self.D_img(src)  #Dis
                D_img_out.append(D_img_out_sub)
            da_output['backbone_DA'] = torch.cat([out.flatten(2).transpose(1, 2) for out in D_img_out],dim=1)
            #======================================================


            # Second.==============objcet query Prototypical Contrast Adaptation======================
            # ---- For Source Domain
            # 1.--------提取原型特征-----------------
            # (1)得到最后一层object query 结果
            # output_source = hs[-1][:, dn_meta['pad_size']:, :]  # source_query_feature :[bs,N,256]
            output_source = hs[-1][:B // 2, :, :]  # source_query_feature :[bs,N,256]
            # (2)得到预测分类结果
            outputs_class_source = out['pred_logits']
            # (3)提取各类别prototype特征
            class_prototypes_source, vaild_class_map_source, global_proto, global_amount, query_mask_source = get_prototype_class_wise(
                output_source, outputs_class_source, self.num_classes,
                global_proto=self.global_proto.detach(), global_amount=self.Amount)

            # class_prototypes_source, vaild_class_map_source, global_proto, global_amount, query_mask_source = get_prototype_class_wise(
            #     output_source, outputs_class_source, self.num_classes)

            # (4)更新全局prototype
            self.global_proto = global_proto
            self.Amount = global_amount

            # ---- For Target Domain
            # 1.--------对目标域数据进行推理-----------------

            # 2.--------提取原型特征-----------------
            # (1)得到最后一层object query 结果
            outputs_target = hs[-1][B // 2:, :, :]
            # (2)得到预测分类结果
            outputs_class_target = self.class_embed[-1](outputs_target)
            # (3)提取各类别prototype特征
            class_prototypes_target, vaild_class_map_target, global_proto, global_amount, query_mask_target = get_prototype_class_wise(
                outputs_target, outputs_class_target, self.num_classes,
                global_proto=self.global_proto.detach(), global_amount=self.Amount)
            # class_prototypes_target, vaild_class_map_target, global_proto, global_amount, query_mask_target = get_prototype_class_wise(
            #     outputs_target, outputs_class_target, self.num_classes)
            # (4)更新全局prototype
            self.global_proto = global_proto
            self.Amount = global_amount

            class_prototypes = torch.cat([class_prototypes_source, class_prototypes_target], dim=0)  # [14,1024]
            class_prototypes_da = self.Proto_D(grad_reverse(class_prototypes))  # [14,1]

            # 3.--------记录 prototype 原型结果-----------------
            da_output['proto_DA'] = {'da_protos': class_prototypes_da,  # concat的原型
                                     'class_map_source': vaild_class_map_source,  # 用于记录源域图像中的存在类别
                                     'class_map_target': vaild_class_map_target,  ##用于记录目标域图像中的存在类别
                                     }

            da_output['global_proto_DA'] = {'output_source': class_prototypes_source,
                                            'outputs_target': class_prototypes_target,
                                            'query_mask_source': vaild_class_map_source,
                                            'query_mask_target': vaild_class_map_target,
                                            'global_proto': self.global_proto,
                                            }

            # 保存所有跟域适应有关的结果
            out['da_output'] = da_output



        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, da_gamma=2):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.da_gamma = da_gamma

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_da(self, outputs, use_focal=False):
        B = outputs.shape[0]
        assert B % 2 == 0

        targets = torch.empty_like(outputs)
        targets[:B//2] = 0
        targets[B//2:] = 1

        loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')

        if use_focal:
            prob = outputs.sigmoid()
            p_t = prob * targets + (1 - prob) * (1 - targets)
            loss = loss * ((1 - p_t) ** self.da_gamma)

        return loss.mean()

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    @staticmethod
    def loss_mae(out):
        num_layers = len(out['mae_output'])
        mae_loss = 0.0
        for layer_idx in range(num_layers):
            if out['mae_output'][layer_idx].shape[1] > 0:
                mae_loss += mse_loss(out['mae_output'][layer_idx], out['features'][layer_idx])
        mae_loss /= num_layers
        return mae_loss

    #####################
    # ==============增添域适应 Loss
    def loss_proto_da(self, outputs):  # ----proto

        output_protos = outputs['da_protos']
        assert output_protos.shape[0] % 2 == 0
        class_map_source = outputs['class_map_source']  # [9,1]
        class_map_target = outputs['class_map_target']  # [9,1]
        class_num = class_map_source.shape[0]
        targets = torch.empty_like(output_protos)
        targets[:class_num] = 0
        targets[class_num:] = 1
        # 计算损失
        loss = F.binary_cross_entropy_with_logits(output_protos, targets, reduction='none')  # [18]
        # 得到class_map
        class_map = torch.cat([class_map_source, class_map_target], dim=0).unsqueeze(1)
        loss = loss * class_map
        return loss.mean()

        # ----全局原型损失-----

    def loss_contrast_da(self, outputs):

        # """ -----------验证结果，考虑调研对比学习损失
        # Args:
        # da_output['global_proto_DA'] = {'output_source': output_source,
        #                                 'outputs_target': outputs_target,
        #                                 'query_mask_source': query_mask_source,
        #                                 'query_mask_target': query_mask_target,
        #                                 'global_proto': global_proto,
        #                                 }
        # Returns:
        #
        # """

        # ---object query
        query_source = outputs['output_source']  # [class_num,256]
        query_target = outputs['outputs_target']  # [class_num,256]

        # ---对应query 掩模图
        query_mask_source = outputs['query_mask_source']  # [class_num]
        query_mask_target = outputs['query_mask_target']  # [class_num]
        # ---global_proto
        global_proto = outputs['global_proto']  # [class_num,256]

        # ---判断反向传播
        assert not global_proto.requires_grad
        assert not query_mask_source.requires_grad
        assert not query_mask_target.requires_grad
        assert query_source.requires_grad
        assert query_target.requires_grad

        # (0)获取维度
        class_num, C = query_source.shape
        # (1)归一化
        query_source = F.normalize(query_source, dim=1)
        query_target = F.normalize(query_target, dim=1)
        global_proto = F.normalize(global_proto, dim=1)

        # (2)计算相似性
        logits_source = query_source.mm(
            global_proto.permute(1, 0).contiguous())  # (class_num,C) * (C,class_num) = （class_num,class_num）
        logits_target = query_target.mm(
            global_proto.permute(1, 0).contiguous())  # (class_num,C) * (C,class_num) = （class_num,class_num）

        # (3)制备标签
        source_label = torch.eye(class_num).to(logits_source.device)
        source_label = source_label * query_mask_source

        target_label = torch.eye(class_num).to(logits_target.device)
        target_label = target_label * query_mask_target

        # (4)计算损失
        ce_criterion = nn.CrossEntropyLoss()
        loss_source = ce_criterion(logits_source, source_label)
        loss_target = ce_criterion(logits_target, target_label)
        loss = loss_source + loss_target

        return loss


    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = {}
        if 'mae_output' in outputs:
            loss_mae = self.loss_mae(outputs)
            losses["loss_mae"] = loss_mae
            return losses

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses


        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.


        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if 'da_output' in outputs:
            for k, v in itertools.islice(outputs['da_output'].items(), 1):
                losses[f'loss_{k}'] = self.loss_da(v, use_focal='query' in k)


            # loss += self.coef_mae * loss_mae




        #===============DA LOSS================
        if 'da_output' in outputs:
            da_output = outputs['da_output']
            #1.backbone_DA
            losses['loss_backbone_DA'] = self.loss_da(da_output['backbone_DA'])

            #2.Proto_DA
            losses['loss_proto_DA'] = self.loss_proto_da(da_output['proto_DA'])

            # 3.global contrast_DA
            losses['loss_global_proto_DA'] = self.loss_contrast_da(da_output['global_proto_DA'])
        #======================================



        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(cfg):
    device = torch.device(cfg.DEVICE)

    backbone = build_backbone(cfg)

    transformer = build_deforamble_transformer(cfg)
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=cfg.DATASET.NUM_CLASSES,
        num_queries=cfg.MODEL.NUM_QUERIES,
        num_feature_levels=cfg.MODEL.NUM_FEATURE_LEVELS,
        aux_loss=cfg.LOSS.AUX_LOSS,
        with_box_refine=cfg.MODEL.WITH_BOX_REFINE,
        two_stage=cfg.MODEL.TWO_STAGE,
        backbone_align=cfg.MODEL.BACKBONE_ALIGN,
        space_align=cfg.MODEL.SPACE_ALIGN,
        channel_align=cfg.MODEL.CHANNEL_ALIGN,
        instance_align=cfg.MODEL.INSTANCE_ALIGN
    )
    if cfg.MODEL.MASKS:
        model = DETRsegm(model, freeze_detr=(cfg.MODEL.FROZEN_WEIGHTS is not None))
    matcher = build_matcher(cfg)
    weight_dict = {'loss_ce': cfg.LOSS.CLS_LOSS_COEF, 'loss_bbox': cfg.LOSS.BBOX_LOSS_COEF}
    weight_dict['loss_mae'] = cfg.coef_mae
    weight_dict['loss_giou'] = cfg.LOSS.GIOU_LOSS_COEF
    if cfg.MODEL.MASKS:
        weight_dict["loss_mask"] = cfg.LOSS.MASK_LOSS_COEF
        weight_dict["loss_dice"] = cfg.LOSS.DICE_LOSS_COEF
    # TODO this is a hack
    if cfg.LOSS.AUX_LOSS:
        aux_weight_dict = {}
        for i in range(cfg.MODEL.DEC_LAYERS - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    weight_dict['loss_backbone'] = cfg.LOSS.BACKBONE_LOSS_COEF
    weight_dict['loss_space_query'] = cfg.LOSS.SPACE_QUERY_LOSS_COEF
    weight_dict['loss_channel_query'] = cfg.LOSS.CHANNEL_QUERY_LOSS_COEF
    weight_dict['loss_instance_query'] = cfg.LOSS.INSTANCE_QUERY_LOSS_COEF

    ######################缝合损失############
    # for DA training:
    weight_dict['loss_backbone_DA'] = cfg.da_backbone_loss_coef
    weight_dict['loss_proto_DA'] = cfg.da_proto_loss_coef
    weight_dict['loss_global_proto_DA'] = cfg.da_global_proto_coef
    print('************')
    print('loss_backbone_DA:', weight_dict['loss_backbone_DA'])
    print('loss_proto_DA:', weight_dict['loss_proto_DA'])
    print('loss_global_proto_DA:', weight_dict['loss_global_proto_DA'])

################################################


    losses = ['labels', 'boxes', 'cardinality']
    if cfg.MODEL.MASKS:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(cfg.DATASET.NUM_CLASSES, matcher, weight_dict, losses, focal_alpha=cfg.LOSS.FOCAL_ALPHA, da_gamma=cfg.LOSS.DA_GAMMA)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if cfg.MODEL.MASKS:
        postprocessors['segm'] = PostProcessSegm()
        if cfg.DATASET.DATASET_FILE == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
