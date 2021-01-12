from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os

import torch

from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter
from models.model import HoidetLoss, SetLoss


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        outputs = self.model(batch['input'])
        loss, loss_states = self.loss(outputs, batch)
        return outputs[-1], loss, loss_states


class Hoidet(object):
    def __init__(self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        loss = SetLoss(opt)
        self.loss_states = loss.states
        self.model_with_loss = ModelWithLoss(model, loss)

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus,
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def set_dist(self, local_rank):
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        self.model_with_loss.cuda()

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=self.opt.device, non_blocking=True)

        if self.opt.apex:
            try:
                from apex.parallel import DistributedDataParallel as DDP
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
                )

            if self.opt.fp16:
                self.model_with_loss, self.optimizer = amp.initialize(self.model_with_loss, self.optimizer,
                                                                      opt_level='O2', keep_batchnorm_fp32=True,
                                                                      loss_scale='dynamic')
            else:
                self.model_with_loss, self.optimizer = amp.initialize(self.model_with_loss, self.optimizer,
                                                                      opt_level='O0')
            self.model_with_loss = DDP(self.model_with_loss)

            if self.opt.sync_bn:
                from apex.parallel import convert_syncbn_model
                self.model_with_loss = convert_syncbn_model(self.model_with_loss)
        else:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model_with_loss = DDP(self.model_with_loss, device_ids=[local_rank], find_unused_parameters=True)

    def run_epoch(self, model_with_loss, epoch, data_loader, phase='train'):
        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_states = {l: AverageMeter() for l in self.loss_states}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)
            output, loss, loss_states = model_with_loss(batch)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()

                if self.opt.apex:
                    from apex import amp
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_states:
                avg_loss_states[l].update(
                    loss_states[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_states[l].avg)
            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
            else:
                bar.next()

            del output, loss, loss_states

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_states.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results

    def train(self, epoch, data_loader):
        model_with_loss = self.model_with_loss
        model_with_loss.train()
        ret, results = self.run_epoch(model_with_loss, epoch, data_loader)
        return ret, results

    def val(self, epoch, data_loader):
        model_with_loss = self.model_with_loss
        model_with_loss.eval()
        torch.cuda.empty_cache()
        with torch.no_grad:
            ret, results = self.run_epoch(model_with_loss, epoch, data_loader, phase='val')
        return ret, results
