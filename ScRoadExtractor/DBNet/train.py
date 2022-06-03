# import torch_xla
# import torch_xla.core.xla_model as xm
from tqdm import tqdm 

import os
# import hickle as hkl
import pickle
import numpy as np
from time import time
import random
import torch
import torch.utils.data as data
from networks.dinknet import ResNet34_EdgeNet
from framework import MyFrame
from loss import Regularized_Loss
from data import ImageFolder
import gc


SHAPE = (512, 512)
# os.chdir("/content/drive/MyDrive/Road-Network-Inference/ScRoadExtractor")
sat_dir = './data/train/sat/'
lab_dir = './data/train/proposal_mask/'
hed_dir = './data/train/rough_edge/'
# sat_dir = './data/train/s2/'
# lab_dir = './data/train/pm2/'
# hed_dir = './data/train/re2/'
print(torch.cuda.device_count())

NAME = 'DBNet_10Cities_zoomed_2'
BATCHSIZE_PER_CARD = 12
batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
solver = MyFrame(ResNet34_EdgeNet, Regularized_Loss, 2e-4)


imagelist = (os.listdir(lab_dir))
# print(len(imagelist))
# random.shuffle(imagelist)
imagelist = imagelist[:9996]
trainlist = map(lambda x: x[:-9], imagelist)
print("Pre Loading Data")
dataset = ImageFolder(trainlist, sat_dir, lab_dir, hed_dir)


# Load data
  # print("Saving dataset pickle")
  # with open('dataset.pkl', 'wb') as fid:
  #    pickle.dump(dataset, fid)
print("Loading Complete")

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=False,
    num_workers=0,
    pin_memory=True)



mylog = open('./logs/' + NAME + '.log', 'w')
print('==============================================================================', file=mylog)
print('==============================================================================', file=mylog)
print('====================================NEW SESSION===============================', file=mylog)
print('==============================================================================', file=mylog)
print('==============================================================================', file=mylog)

tic = time()
no_optim = 0
total_epoch = 300
train_epoch_best_loss = 100.

solver.load('./weights/' + NAME + '.th')
for epoch in range(1, total_epoch + 1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    for img, mask, hed in tqdm(data_loader_iter):
        solver.set_input(img, mask, hed)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss

    train_epoch_loss /= len(data_loader_iter)
    print('********', file=mylog)
    print('epoch:', epoch, '    time:', int(time() - tic), file=mylog)
    print('train_loss:', train_epoch_loss, file=mylog)
    print('SHAPE:', SHAPE, file=mylog)
    print('********')
    print('epoch:', epoch, '    time:', int(time() - tic))
    print('train_loss:', train_epoch_loss)
    print('SHAPE:', SHAPE)

    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save('weights/' + NAME + '.th')
    if no_optim > 6:
        print('early stop at %d epoch' % epoch, file=mylog)
        print('early stop at %d epoch' % epoch)
        break
    if no_optim > 3:
        if solver.old_lr < 5e-7:
            break
        solver.load('./weights/' + NAME + '.th')
        solver.update_lr(5.0, factor=True, mylog=mylog)
    mylog.flush()
    # del data_loader_iter
    # gc.collect()
print('Finish!', file=mylog)
print('Finish!')
mylog.close()
