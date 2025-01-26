# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .hyperbolic_models import HyperbolicKAN, HyperbolicMLP, HyperbolicLKAN
from .position_encoding import PositionEmbeddingSine
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY

from .gcn.manifolds.lorentz import Lorentz
from .gcn.manifolds.lmath import expmap, logmap

## Adding a second decoder to refine Masks

class HyperbolicRefineDecoder(nn.Module):
    """
    An optional second-stage decoder that refines mask predictions in hyperbolic space.
    """
    def __init__(self, hidden_dim, nheads=4, num_refine_layers=1, curvature=-1.0):
        super().__init__()
        self.curvature = curvature
        self.self_attn_layers = nn.ModuleList()
        self.cross_attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        for _ in range(num_refine_layers):
            self.self_attn_layers.append(
                SelfAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0)
            )
            self.cross_attn_layers.append(
                HyperbolicCrossAttentionLayer(d_model=hidden_dim, nhead=nheads,
                                              dropout=0.0, curvature=curvature)
            )
            self.ffn_layers.append(
                FFNLayer(d_model=hidden_dim, dim_feedforward=1024, dropout=0.0)
            )

    def forward(self, query, memory, mask_features, pos_embed):
        """
        Refine 'query' using memory, or even the previous 'pred_masks'.
        """
        for cross_attn, self_attn, ffn in zip(self.cross_attn_layers, self.self_attn_layers, self.ffn_layers):
            query = cross_attn(query, memory, pos=pos_embed)
            query = self_attn(query)
            query = ffn(query)
        return query


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)

class HyperbolicCrossAttentionLayer(nn.Module):
    """
    A version of CrossAttentionLayer that transforms Q and K to hyperbolic space
    before the attention operation.
    """
    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False, curvature=-1.0):
        super().__init__()
        self.curvature = curvature  # Curvature for hyperbolic operations
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # Transform queries & keys in hyperbolic space
        # Transform queries & keys in hyperbolic space
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(memory, pos)
        q_hyp = expmap(q, q / (q.norm(dim=-1, keepdim=True) + 1e-7), k=self.curvature)
        k_hyp = expmap(k, k / (k.norm(dim=-1, keepdim=True) + 1e-7), k=self.curvature)

        # Compute the Gromov product: shape is [T, B]
        # gromov_qk = 0.5 * (q_hyp.norm(dim=-1, keepdim=True) ** 2 +
        #                    k_hyp.norm(dim=-1, keepdim=True) ** 2 -
        #                    (q_hyp - k_hyp).norm(dim=-1, keepdim=True) ** 2)
        # gromov_qk = gromov_qk.squeeze(-1)  # shape => [T, B]

        # Fix: Expand Gromov product to [T, B, embed_dim]
        # d_model = q_hyp.shape[-1]  # e.g. if q_hyp is [T, B, embed_dim]
        # gromov_qk = gromov_qk.unsqueeze(-1)  # => [T, B, 1]
        # gromov_qk = gromov_qk.expand(-1, -1, d_model)  # => [T, B, embed_dim]

        q_euc = logmap(torch.zeros_like(q_hyp), q_hyp, k=self.curvature)
        k_euc = logmap(torch.zeros_like(k_hyp), k_hyp, k=self.curvature)

        # Gromov product => [T, B, d_model]
        # gromov_qk = gromov_qk.expand(-1, -1, memory.shape[-1])  # already done

        # memory => [1, B, d_model] originally
        # Expand memory to the same T dimension (e.g., T=10)
        # num_queries = gromov_qk.size(0)
        # if memory.size(0) == 1 and memory.size(0) != num_queries:
        #     memory = memory.expand(num_queries, -1, -1)  # => [T, B, d_model]

        # tgt2 = self.multihead_attn(
        #     query=gromov_qk,  # [T, B, d_model]
        #     key=gromov_qk,  # [T, B, d_model]
        #     value=memory,  # [T, B, d_model] after expansion
        #     attn_mask=memory_mask,
        #     key_padding_mask=memory_key_padding_mask
        # )[0]

        tgt2 = self.multihead_attn(q_euc, k_euc, value=memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = self.with_pos_embed(tgt2, query_pos)
        k = self.with_pos_embed(memory, pos)

        # Hyperbolic transform
        q_hyp = expmap(q, q / (q.norm(dim=-1, keepdim=True) + 1e-7), k=self.curvature)
        k_hyp = expmap(k, k / (k.norm(dim=-1, keepdim=True) + 1e-7), k=self.curvature)

        # Compute the Gromov product for queries and keys in hyperbolic space
        gromov_qk = 0.5 * (q_hyp.norm(dim=-1, keepdim=True) ** 2 +
                           k_hyp.norm(dim=-1, keepdim=True) ** 2 -
                           (q_hyp - k_hyp).norm(dim=-1, keepdim=True) ** 2)
        gromov_qk = gromov_qk.squeeze(-1)  # Adjust dimensions for attention computation

        # Use the Gromov product directly for attention computation
        tgt2 = self.multihead_attn(gromov_qk, gromov_qk, value=memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]


        # q_euc = logmap(torch.zeros_like(q_hyp), q_hyp, k=self.curvature)
        # k_euc = logmap(torch.zeros_like(k_hyp), k_hyp, k=self.curvature)
        #
        # tgt2 = self.multihead_attn(q_euc, k_euc, value=memory,
        #                            attn_mask=memory_mask,
        #                            key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


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


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoder(nn.Module):
    _version = 2

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
            self,
            in_channels,
            mask_classification=True,
            *,
            num_classes: int,
            hidden_dim: int,
            num_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            pre_norm: bool,
            mask_dim: int,
            enforce_input_project: bool,
            hyperbolic_curvature: float = -1.0,  # New: curvature for hyperbolic layers
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                HyperbolicCrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    curvature=hyperbolic_curvature
                )
            )

            # old euclidean space
            # self.transformer_cross_attention_layers.append(
            #     CrossAttentionLayer(
            #         d_model=hidden_dim,
            #         nhead=nheads,
            #         dropout=0.0,
            #         normalize_before=pre_norm,
            #     )
            # )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.use_refine = True  # Enable refinement decoder
        if self.use_refine:
            self.refine_decoder = HyperbolicRefineDecoder(
                hidden_dim=hidden_dim,
                nheads=nheads,
                num_refine_layers=1,
                curvature=hyperbolic_curvature
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # self.mask_embed = HyperbolicMLP(hidden_dim, hidden_dim, mask_dim, 3, manifold, curvature=-1.0)
        #
        # # Assuming 'hidden_dim' and 'mask_dim' are already defined in your model
        self.mask_embed = HyperbolicKAN(
            layers_hidden=[hidden_dim, hidden_dim, mask_dim],  # Define layer sizes
            grid_size=5,  # Optional: Adjust grid size
            spline_order=3,  # Optional: Adjust spline order
            scale_noise=0.1,  # Optional: Noise scaling
            scale_base=1.0,  # Optional: Base scaling
            scale_spline=1.0,  # Optional: Spline scaling
            base_activation=torch.nn.SiLU,  # Activation function
            grid_eps=0.02,  # Optional: Grid perturbation
            grid_range=[-1, 1],  # Optional: Range of grid
            curvature=-1
        )

        # self.mask_embed = HyperbolicLKAN(
        #     layers_hidden=[hidden_dim, hidden_dim, mask_dim],  # Dimensioni strati
        #     curvature=-1.0,  # Curvatura del piano iperbolico
        #     dropout_rate=0.5  # Fattore di dropout
        # )

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        return ret

    def forward(self, x, mask_features, mask=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        # print(query_embed.shape)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        # print(f'{output.shape=}, {mask_features.shape=}')

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
                                                                               attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
                                                                                   attn_mask_target_size=size_list[(
                                                                                                                           i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }

        #HYPERBOLIC APPROACH: 2 STAGE PART using refine decoder
        refined_query = self.refine_decoder(
            output,
            src[level_index],  # memory from last cross-attn
            mask_features,
            pos[level_index]
        )
        refined_class, refined_mask, _ = self.forward_prediction_heads(refined_query, mask_features, size_list[-1])
        out['pred_logits_refined'] = refined_class
        out['pred_masks_refined'] = refined_mask
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        # mask_embed = self.mask_embed(decoder_output)
        # curvature = 1.0
        # Explicitly pass curvature to the mask_embed forward method
        curvature = -1.0  # Or whatever curvature value you intend to use
        mask_embed = self.mask_embed(decoder_output, curvature=curvature)
        # mask_embed = self.mask_embed(decoder_output, curvature=curvature)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                         1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]