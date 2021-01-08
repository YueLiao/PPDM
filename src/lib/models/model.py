from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import models.networks as networks

from .losses import FocalLoss, RegL1Loss, RegLoss
from utils import clamped_sigmoid


def create_model(arch, heads, head_conv):
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = getattr(networks, arch)
    model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
    return model


def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
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
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)


class HoidetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(HoidetLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = self.crit_reg
        self.opt = opt
        self.states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'hm_rel_loss',
                       'sub_offset_loss', 'obj_offset_loss']

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, hm_rel_loss, sub_offset_loss, obj_offset_loss = 0, 0, 0, 0, 0, 0

        for s in range(opt.num_stacks):
            output = outputs[s]
            output['hm'] = clamped_sigmoid(output['hm'])
            output['hm_rel'] = clamped_sigmoid(output['hm_rel'])
            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            hm_rel_loss += self.crit(output['hm_rel'], batch['hm_rel']) / opt.num_stacks

            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks
                sub_offset_loss += self.crit_reg(
                    output['sub_offset'], batch['offset_mask'],
                    batch['rel_ind'], batch['sub_offset']
                )
                obj_offset_loss += self.crit_reg(
                    output['obj_offset'], batch['offset_mask'],
                    batch['rel_ind'], batch['obj_offset']
                )
            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

        loss = opt.hm_weight * (hm_loss + hm_rel_loss) + opt.wh_weight * (
                wh_loss + sub_offset_loss + obj_offset_loss) + \
               opt.off_weight * off_loss
        loss_states = {'loss': loss, 'hm_loss': hm_loss,
                       'wh_loss': wh_loss, 'off_loss': off_loss, 'hm_rel_loss': hm_rel_loss,
                       'sub_offset_loss': sub_offset_loss, 'obj_offset_loss': obj_offset_loss}
        return loss, loss_states
