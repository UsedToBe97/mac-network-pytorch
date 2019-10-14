import sys
import os
import pickle
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from dataset import CLEVR, collate_data, transform
from model import MACNetwork
from config import cfg


device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
print("using {}".format(device))


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def train(epoch):
    clevr = CLEVR(cfg.DATALOADER.FEATURES_PATH, transform=transform)
    train_set = DataLoader(
        clevr, batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_data, drop_last=True
    )

    dataset = iter(train_set)
    pbar = tqdm(dataset)
    moving_loss = 0

    net.train(True)
    for image, question, q_len, answer, _, _ in pbar:
        image, question, answer = (
            image.to(device),
            question.to(device),
            answer.to(device),
        )

        net.zero_grad()
        output = net(image, question, q_len)
        loss = criterion(output, answer)
        loss.backward()
        if cfg.SOLVER.GRAD_CLIP:
            nn.utils.clip_grad_norm_(net.parameters(), cfg.SOLVER.GRAD_CLIP)
        optimizer.step()
        correct = output.detach().argmax(1) == answer
        accuracy = correct.float().mean().item()

        if moving_loss == 0:
            moving_loss = accuracy
        else:
            moving_loss = moving_loss * 0.99 + accuracy * 0.01

        pbar.set_description(
            'Epoch: {}; Loss: {:.5f}; Acc: {:.5f}'.format(
                epoch, loss.item(), moving_loss
            )
        )
        accumulate(net_running, net)

    clevr.close()


def valid(epoch):
    clevr = CLEVR(cfg.DATALOADER.FEATURES_PATH, 'val', transform=None)
    valid_set = DataLoader(
        clevr, batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=collate_data, drop_last=True
    )
    dataset = iter(valid_set)

    net_running.train(False)
    with torch.no_grad():
        all_corrects = 0

        for image, question, q_len, answer, _, _ in tqdm(dataset):
            image, question = image.to(device), question.to(device)

            output = net_running(image, question, q_len)
            correct = output.detach().argmax(1) == answer.to(device)
            
            all_corrects += correct.float().mean().item()
        
        if scheduler:
            scheduler.step(all_corrects / len(dataset))

        print('Avg Acc: {:.5f}'.format(all_corrects / len(dataset)))

    clevr.close()


def update_cfg(cfg):
    parser = argparse.ArgumentParser(description="Train for CACT-Mac")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    return cfg  


if __name__ == '__main__':
    cfg = update_cfg(cfg)

    net = MACNetwork(cfg).to(device)
    net_running = MACNetwork(cfg).to(device)

    if cfg.LOAD:
        with open(cfg.LOAD_PATH, 'rb') as f:
            state = torch.load(f, map_location=device)
        net.load_state_dict(state)
    accumulate(net_running, net, 0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=cfg.SOLVER.LR)
    
    # LR scheduler
    scheduler = None
    if cfg.SOLVER.USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=0, threshold=0.001, threshold_mode='rel')

    for epoch in range(1, cfg.SOLVER.EPOCHS + 1):
        train(epoch)
        valid(epoch)

        with open(
            'checkpoint/checkpoint_{}_{}.model'.format(
                cfg.SAVE_PATH,
                str(epoch).zfill(2)),
            'wb') as f:
            torch.save(net_running.state_dict(), f)
