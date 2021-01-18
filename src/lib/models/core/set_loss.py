import torch
import torch.nn as nn
import torch.nn.functional as F

from models.losses import FocalLoss, RegL1Loss, RegLoss
from models.losses import sigmoid_focal_loss_jit
from utils import clamped_sigmoid


class MinCostMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, opt):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = 1
        self.cost_offset = 0.1
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2
        if self.cost_class == 0 and self.cost_offset == 0:
            raise ValueError('The weight of matcher must not be all zero')

    @torch.no_grad()
    def forward(self, output, batch):
        """ Performs the matching

        Params:
            output: This is a dict that contains at least these entries:
                 "hm_rel": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            batch: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, k, h, w = output['hm_rel'].shape

        # We flatten to compute the cost matrices in a batch

        batch_out_prob = output['hm_rel'].permute(0, 2, 3, 1).reshape(
            bs, h * w, k).sigmoid()  # [batch_size, num_queries, num_classes]
        batch_out_offset = torch.cat(
            [output['sub_offset'], output['obj_offset']],
            1).permute(0, 2, 3, 1).reshape(bs, h * w, 4)  # [batch_size, num_queries, 4]
        batch_tgt_offset = batch['offset'].permute(0, 3, 4, 1, 2).reshape(bs, -1, h * w, 4)

        indices = []

        for i in range(bs):
            index = batch['offset_mask'][i].nonzero().squeeze()

            tgt_ids = batch['hm_rel'][i].index_select(0, index)
            if tgt_ids.shape[0] == 0:
                indices.append(([], []))
                continue

            out_prob = batch_out_prob[i]
            out_offset = batch_out_offset[i]
            tgt_offset = batch_tgt_offset[i].index_select(0, index)

            # Compute the classification cost.
            alpha = self.focal_loss_alpha
            gamma = self.focal_loss_gamma
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (
                -(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * (
                    (1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:,
                                                      tgt_ids]

            # Compute the L1 cost between boxes
            cost_offset = torch.abs(tgt_offset - out_offset).sum(-1).permute(1, 0)

            # Final cost matrix
            C = self.cost_offset * cost_offset + self.cost_class * cost_class

            _, src_ind = torch.min(C, dim=0)
            tgt_ind = torch.arange(len(tgt_ids)).to(src_ind)
            indices.append((src_ind, tgt_ind))

        return [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class SetLoss(nn.Module):
    def __init__(self, opt, Matcher=MinCostMatcher):
        super(SetLoss, self).__init__()
        self.opt = opt
        self.matcher = Matcher(opt)
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = self.crit_reg
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2
        self.states = [
            'loss', 'hm_loss', 'wh_loss', 'off_loss', 'hm_rel_loss',
            'offset_loss'
        ]

    def forward(self, outputs, batch):
        opt = self.opt
        # hm_loss, wh_loss, off_loss, hm_rel_loss, obj_offset_loss = 0, 0, 0, 0, 0
        losses = [0, 0, 0, 0, 0]

        for s in range(opt.num_stacks):
            output = outputs[s]

            indices = self.matcher(output, batch)
            losses_ = self.loss(output, batch, indices)
            losses = [i + x for i, x in zip(losses, losses_)]

        hm_loss, wh_loss, off_loss, hm_rel_loss, offset_loss = losses

        loss = opt.hm_weight * (hm_loss + hm_rel_loss) + opt.wh_weight * (
                wh_loss + offset_loss) + opt.off_weight * off_loss
        loss_states = {
            'loss': loss,
            'hm_loss': hm_loss,
            'wh_loss': wh_loss,
            'off_loss': off_loss,
            'hm_rel_loss': hm_rel_loss,
            'offset_loss': offset_loss
        }
        return loss, loss_states

    def loss(self, output, batch, indices):
        hm_loss, hm_rel_loss = self.loss_cls(output, batch, indices)
        wh_loss, offset_loss, off_loss = self.loss_reg(output, batch, indices)
        return hm_loss, wh_loss, off_loss, hm_rel_loss, offset_loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_cls(self, output, batch, indices):
        opt = self.opt
        src_logits = output['hm_rel']
        tgt_logits = batch['hm_rel']
        bs, k, h, w = src_logits.shape
        src_logits = src_logits.permute(0, 2, 3, 1).reshape(bs, h * w, k)

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(tgt_logits, indices)])
        target_classes = torch.full(src_logits.shape[:2],
                                    117,
                                    dtype=torch.int64,
                                    device=src_logits.device)
        target_classes[idx] = target_classes_o
        src_logits = src_logits.flatten(0, 1)
        target_classes = target_classes.flatten(0, 1)
        pos_inds = torch.nonzero(target_classes != 117, as_tuple=True)[0]
        labels = torch.zeros_like(src_logits)
        labels[pos_inds, target_classes[pos_inds]] = 1

        output['hm'] = clamped_sigmoid(output['hm'])
        output['hm_rel'] = sigmoid_focal_loss_jit(
            src_logits,
            labels,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )
        hm_loss = self.crit(output['hm'], batch['hm']) / self.opt.num_stacks
        hm_rel_loss = output['hm_rel'] / self.opt.num_stacks
        # output['hm_rel'] = clamped_sigmoid(output['hm_rel'])
        # hm_rel_loss = self.crit(output['hm_rel'],
        #                        batch['hm_rel']) / self.opt.num_stacks
        return hm_loss, hm_rel_loss

    def loss_reg(self, output, batch, indices):
        wh_loss, offset_loss, off_loss = 0, 0, 0

        tgt_offset = batch['offset']
        bs, k, n, h, w = tgt_offset.shape

        idx = self._get_src_permutation_idx(indices)
        src_offset = torch.cat(
            [output['sub_offset'], output['obj_offset']],
            1).permute(0, 2, 3, 1).reshape(bs, h * w, 4)
        src_offset = src_offset[idx]
        tgt_offset = torch.cat([t.flatten(-2, -1)[i, :, j] for t, (j, i) in zip(tgt_offset, indices)], dim=0)
        offset_loss = F.l1_loss(src_offset, tgt_offset, reduction='sum')
        offset_loss = offset_loss / max(float(src_offset.shape[0]), 1.0)
        offset_loss = offset_loss / self.opt.num_stacks
        if self.opt.wh_weight > 0:
            wh_loss = self.crit_reg(output['wh'], batch['reg_mask'],
                                    batch['ind'],
                                    batch['wh']) / self.opt.num_stacks
        if self.opt.reg_offset and self.opt.off_weight > 0:
            off_loss = self.crit_reg(output['reg'], batch['reg_mask'],
                                     batch['ind'],
                                     batch['reg']) / self.opt.num_stacks
        return wh_loss, offset_loss, off_loss
