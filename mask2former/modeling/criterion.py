# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list

from torchvision import transforms

from .transformer_decoder.gcn.manifolds.lorentz import Lorentz


# Hyperbolic dice loss
def dice_loss(inputs: torch.Tensor,
                         targets: torch.Tensor,
                         num_masks: float,
                         manifold,
                         c=1.0):
    """
    Example "hyperbolic" Dice. We do the following:
      1) Convert logits -> probability using sigmoid.
      2) Flatten into shape [batch, dim].
      3) Map each vector into hyperbolic space using manifold.expmap0(...).
      4) Approximate "intersection" via a function of hyperbolic distance, e.g. exp(-dist).
      5) Build a ratio akin to dice, then sum over batch.

    Args:
        inputs: shape [batch_size, ...], raw logits.
        targets: same shape, binary in {0, 1}, or possibly real in [0,1].
        num_masks: normalizing constant for final average.
        manifold: e.g. geoopt.manifolds.PoincareBall() or similar.
        c: curvature parameter (positive if using Poincaré c>0).

    Returns:
        A scalar loss.
    """
    # 1) Convert to probability + flatten
    inputs = inputs.sigmoid()                  # shape: [B, ...]
    B = inputs.size(0)
    inputs_flat = inputs.flatten(1)            # e.g. [B, D]
    targets_flat = targets.flatten(1)          # [B, D]

    # 2) Map to hyperbolic space
    x_hyp = manifold.expmap0(inputs_flat)     # shape [B, D, *], depends on manifold
    y_hyp = manifold.expmap0(targets_flat)

    # 3) Define "intersection" + "union" using a distance-based approach.
    #    This is not standard. For demonstration, we treat intersection ~ e^{-dist}.
    dist = manifold.dist(x_hyp, y_hyp)    # shape [B], distance per sample
    intersection = torch.exp(-dist)           # ~ "closer => bigger intersection"
    # Let's say "union" ~ 2 + dist for example
    union = 2.0 + dist

    # 4) Dice coefficient: 1 - (2 * intersection / union)
    dice_per_sample = 1 - (2 * intersection / union)
    loss = dice_per_sample.mean()  # average across batch

    # 5) Scale by num_masks for consistency with your code (optional)
    return loss / num_masks


# def dice_loss(
#         inputs: torch.Tensor,
#         targets: torch.Tensor,
#         num_masks: float,
# ):
#     """
#     Compute the DICE loss, similar to generalized IOU for masks
#     Args:
#         inputs: A float tensor of arbitrary shape.
#                 The predictions for each example.
#         targets: A float tensor with the same shape as inputs. Stores the binary
#                  classification label for each element in inputs
#                 (0 for the negative class and 1 for the positive class).
#     """
#     inputs = inputs.sigmoid()
#     inputs = inputs.flatten(1)
#     numerator = 2 * (inputs * targets).sum(-1)
#     denominator = inputs.sum(-1) + targets.sum(-1)
#     loss = 1 - (numerator + 1) / (denominator + 1)
#     return loss.sum() / num_masks


# dice_loss_jit = torch.jit.script(
#     dice_loss
# )  # type: torch.jit.ScriptModule

# Hyperbolic
dice_loss_jit = dice_loss

# Hyperbolic
def sigmoid_ce_loss(inputs: torch.Tensor,
                        targets: torch.Tensor,
                        num_masks: float,
                        manifold,
                        c=1.0,
                        margin=1.0):
    """
    Example hyperbolic 'BCE' that tries to separate matched vs. unmatched points
    in hyperbolic space by a margin.

    Args:
        inputs: shape [B, ...], raw logits.
        targets: shape [B, ...], 0 or 1 typically.
        num_masks: scalar for normalization, as in your original code.
        manifold, c: for hyperbolic geometry.
        margin: margin for negative pairs.

    Returns:
        A scalar loss.
    """
    # Flatten
    B = inputs.shape[0]
    inputs_flat = inputs.sigmoid().flatten(1)  # map logits to [0,1]
    targets_flat = targets.flatten(1)

    # Map to hyperbolic
    x_hyp = manifold.expmap0(inputs_flat)
    y_hyp = manifold.expmap0(targets_flat)

    # Distances in hyperbolic space
    dist = manifold.dist(x_hyp, y_hyp)  # shape [B], one dist per sample if flattened properly

    # For 'positive' pairs (when target=1): we want dist to be small
    # For 'negative' pairs (when target=0): we want dist > margin
    # We'll define a simple hinge approach:
    # if label=1 => loss = dist
    # if label=0 => loss = F.relu(margin - dist)

    # But we have a vector of 0/1 in targets_flat, so let's separate them:
    mask_pos = (targets_flat.sum(dim=1) > 0).float()  # hack: if sum>0 => it's "pos"
    # Or do per-pixel if you'd like. This part is not trivial.

    pos_loss = dist * mask_pos
    neg_loss = F.relu(margin - dist) * (1 - mask_pos)
    total_loss = (pos_loss + neg_loss).mean()

    return total_loss / num_masks

# def sigmoid_ce_loss(
#         inputs: torch.Tensor,
#         targets: torch.Tensor,
#         num_masks: float,
# ):
#     """
#     Args:
#         inputs: A float tensor of arbitrary shape.
#                 The predictions for each example.
#         targets: A float tensor with the same shape as inputs. Stores the binary
#                  classification label for each element in inputs
#                 (0 for the negative class and 1 for the positive class).
#     Returns:
#         Loss tensor
#     """
#     loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
#
#     return loss.mean(1).sum() / num_masks


# sigmoid_ce_loss_jit = torch.jit.script(
#     sigmoid_ce_loss
# )  # type: torch.jit.ScriptModule
sigmoid_ce_loss_jit = sigmoid_ce_loss

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))

def get_batch_avg(logits, label_ood):
    N, _, H, W = logits.shape
    m = logits.mean(1).mean()
    ma = - m.view(1,1,1).repeat(N,H,W) * label_ood
    return ma

class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, ood_loss, margin, lovasz_loss,
                 pebal_reward, ood_reg, densehybrid_beta,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        manifold = Lorentz()  # Puoi configurare la curvatura negativa qui

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks, manifold),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks, manifold),
            # "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            # "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_batch_avg(logits, label_ood):
        N, _, H, W = logits.shape
        m = logits.mean(1).mean()
        ma = - m.view(1, 1, 1).repeat(N, H, W) * label_ood
        return ma
    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)