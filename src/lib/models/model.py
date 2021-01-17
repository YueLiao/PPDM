from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import models.networks as networks

from .losses import FocalLoss, RegL1Loss, RegLoss, GIouLoss, generalized_box_iou, sigmoid_focal_loss_jit
from utils import clamped_sigmoid


def create_model(arch, heads, head_conv):
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = getattr(networks, arch)
    model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
    return model


def load_model(model,
               model_path,
               optimizer=None,
               resume=False,
               lr=None,
               lr_step=None):
    start_epoch = 0
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    optimizer_grouped_parameters = []
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}.'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def save_model(path, epoch, model, optimizer=None):
    # TODO: Save DDP Model
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch, 'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)


class HoidetLoss(nn.Module):
    def __init__(self, opt):
        super(HoidetLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = self.crit_reg
        self.opt = opt
        self.states = [
            'loss', 'hm_loss', 'wh_loss', 'off_loss', 'hm_rel_loss',
            'offset_loss'
        ]

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, hm_rel_loss, offset_loss = 0, 0, 0, 0, 0, 0

        for s in range(opt.num_stacks):
            output = outputs[s]
            output['hm'] = clamped_sigmoid(output['hm'])
            output['hm_rel'] = clamped_sigmoid(output['hm_rel'])
            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            hm_rel_loss += self.crit(output['hm_rel'],
                                     batch['hm_rel']) / opt.num_stacks

            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(output['wh'], batch['reg_mask'],
                                         batch['ind'],
                                         batch['wh']) / opt.num_stacks
                sub_offset_loss += self.crit_reg(output['sub_offset'],
                                                 batch['offset_mask'],
                                                 batch['rel_ind'],
                                                 batch['sub_offset'])
                obj_offset_loss += self.crit_reg(output['obj_offset'],
                                                 batch['offset_mask'],
                                                 batch['rel_ind'],
                                                 batch['obj_offset'])
            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'],
                                          batch['reg']) / opt.num_stacks

        loss = opt.hm_weight * (hm_loss + hm_rel_loss) + opt.wh_weight * (
                wh_loss + sub_offset_loss + obj_offset_loss) + \
               opt.off_weight * off_loss
        loss_states = {
            'loss': loss,
            'hm_loss': hm_loss,
            'wh_loss': wh_loss,
            'off_loss': off_loss,
            'hm_rel_loss': hm_rel_loss,
            'sub_offset_loss': sub_offset_loss,
            'obj_offset_loss': obj_offset_loss
        }
        return loss, loss_states


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
        self.cost_offset = 1
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
            neg_cost_class = (1 - alpha) * (out_prob**gamma) * (
                -(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * (
                (1 - out_prob)**gamma) * (-(out_prob + 1e-8).log())
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
        #output['hm_rel'] = clamped_sigmoid(output['hm_rel'])
        #hm_rel_loss = self.crit(output['hm_rel'],
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
        dt=tgt_offset[0]
        dj, di = indices[0]
        ti = dt[di]
        tgt_offset = torch.cat([t.flatten(-2, -1)[i, :, j] for t, (j, i) in zip(tgt_offset, indices)], dim=0)

        output['offset_loss'] = F.l1_loss(src_offset, tgt_offset, reduction='none')
        offset_loss = output['offset_loss'] / self.opt.num_stacks

        if self.opt.wh_weight > 0:
            wh_loss = self.crit_reg(output['wh'], batch['reg_mask'],
                                    batch['ind'],
                                    batch['wh']) / self.opt.num_stacks
        if self.opt.reg_offset and self.opt.off_weight > 0:
            off_loss = self.crit_reg(output['reg'], batch['reg_mask'],
                                     batch['ind'],
                                     batch['reg']) / self.opt.num_stacks
        return wh_loss, offset_loss, off_loss


class SetCriterion(nn.Module):
    """ This class computes the loss for OneNet.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, cfg, num_classes, matcher, weight_dict, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_loss_alpha = cfg.MODEL.OneNet.ALPHA
        self.focal_loss_gamma = cfg.MODEL.OneNet.GAMMA
        self.giou = GIouLoss()
        self.l1 = F.l1_loss

    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'hm_rel' in outputs
        src_logits = outputs['hm_rel']
        bs, k, h, w = src_logits.shape
        src_logits = src_logits.permute(0, 2, 3, 1).reshape(bs, h * w, k)

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2],
                                    self.num_classes,
                                    dtype=torch.int64,
                                    device=src_logits.device)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.flatten(0, 1)
        # prepare one_hot target.
        target_classes = target_classes.flatten(0, 1)
        pos_inds = torch.nonzero(target_classes != self.num_classes,
                                 as_tuple=True)[0]
        labels = torch.zeros_like(src_logits)
        labels[pos_inds, target_classes[pos_inds]] = 1
        # comp focal loss.
        class_loss = 0
        losses = {'loss_ce': class_loss}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 0
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes']
        bs, k, h, w = src_boxes.shape
        src_boxes = src_boxes.permute(0, 2, 3, 1).reshape(bs, h * w, k)

        src_boxes = src_boxes[idx]
        target_boxes = torch.cat(
            [t['boxes_xyxy'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        loss_giou = self.giou(src_boxes, target_boxes)
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        image_size = torch.cat([v["image_size_xyxy_tgt"] for v in targets])
        src_boxes_ = src_boxes / image_size
        target_boxes_ = target_boxes / image_size

        loss_bbox = self.l1(src_boxes_, target_boxes_, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
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
        outputs_without_aux = {
            k: v
            for k, v in outputs.items() if k != 'aux_outputs'
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes],
                                    dtype=torch.float,
                                    device=next(iter(outputs.values())).device)
        if dist_ok():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_boxes))

        return losses


def dist_ok():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if dist_ok() else 1


def get_rank():
    return dist.get_rank() if dist_ok() else 0
