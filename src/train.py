from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data

import trainers
from opts import opts
from models.model import create_model, load_model, save_model
from logger import Logger
from datasets import get_dataset

import torch.distributed as dist
from utils.sampler import DistributedSampler


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    trainer = getattr(trainers, opt.task)(opt, model, optimizer)
    train_dataset = Dataset(opt, 'train')
    if opt.dist:
        if opt.slurm:
            local_rank = int(os.environ.get('LOCAL_RANK') or 0)
            trainer.set_dist(local_rank)
        else:
            num_gpus = torch.cuda.device_count()
            local_rank = opt.rank % num_gpus
            trainer.set_dist(local_rank)

        num_replicas = dist.get_world_size()
        opt.batch_size = opt.batch_size // num_replicas
        opt.num_workers = opt.num_workers // num_replicas
        train_sampler = DistributedSampler(train_dataset,
                                           rank=opt.rank,
                                           num_replicas=num_replicas)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            pin_memory=True)
    else:
        trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

        train_loader = torch.utils.data.DataLoader(
            Dataset(opt, 'train'),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True
        )
    if opt.rank == 0:
        logger = Logger(opt)
    print('Starting training...')
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        if opt.dist:
            train_sampler.set_epoch(epoch - 1)

        log_dict_train, _ = trainer.train(epoch, train_loader)
        if opt.rank == 0:
            logger.write('epoch: {} |'.format(epoch))
            for k, v in log_dict_train.items():
                logger.scalar_summary('train_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            if epoch > 110:
                save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                           epoch, model, optimizer)
            else:
                save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                           epoch, model, optimizer)
            logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    if opt.rank == 0:
        logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
