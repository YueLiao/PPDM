from models.losses import FocalLoss, RegL1Loss, RegLoss
from utils import clamped_sigmoid

import torch.nn as nn


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
        hm_loss, wh_loss, off_loss, hm_rel_loss, sub_offset_loss, obj_offset_loss = 0, 0, 0, 0, 0, 0

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
