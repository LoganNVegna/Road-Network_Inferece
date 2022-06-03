import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
from tqdm import tqdm 
from google.colab.patches import cv2_imshow
from skimage.draw import line
import cv2
import os
import math
import numpy as np
import pandas as pd

from time import time
# from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_size256, DinkNet34_size256_EdgeNet
from networks.dinknet import ResNet34_EdgeNet#, ResNet34_BRN, ResNet34_
# from networks.unet import Unet
# from networks.resUnet import resUnet

BATCHSIZE_PER_CARD = 2

class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        
    def test_one_img_from_path(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_from_path_8(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]

        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)

        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]

        return mask2

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda())

        # maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        # maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        # maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        # maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        maska, edgea = self.net.forward(img1)
        maskb, edgeb = self.net.forward(img2)
        maskc, edgec = self.net.forward(img3)
        maskd, edged = self.net.forward(img4)

        maska = maska.squeeze().cpu().data.numpy()
        maskb = maskb.squeeze().cpu().data.numpy()
        maskc = maskc.squeeze().cpu().data.numpy()
        maskd = maskd.squeeze().cpu().data.numpy()

        edgea = edgea.squeeze().cpu().data.numpy()
        edgeb = edgeb.squeeze().cpu().data.numpy()
        edgec = edgec.squeeze().cpu().data.numpy()
        edged = edged.squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        edge1 = edgea + edgeb[:, ::-1] + edgec[:, :, ::-1] + edged[:, ::-1, ::-1]
        edge2 = edge1[0] + np.rot90(edge1[1])[::-1, ::-1]

        # img1 = np.concatenate([img[None],img[None]])
        # img1 = img1.transpose(0,3,1,2)
        # img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        # maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        # return maska[0,:,:]+maska[1,:,:]
        return mask2, edge2
    
    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32)/255.0 * 3.2 - 1.6
        img6 = V(torch.Tensor(img6).cuda())
        
        maska = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]
        
        return mask3
    
    def test_one_img_from_path_1(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = np.concatenate([img3,img4]).transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5).cuda())
        
        mask = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        mask1 = mask[:4] + mask[4:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]
        
        return mask3

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

maindir = '/content/drive/MyDrive/Road-Network-Inference/'
solver = TTAFrame(ResNet34_EdgeNet)

solver.load('./weights/DBNet_10Cities_zoomed_2.th')
tic = time()
# target_grey = '/home/ck/data/wuhan/train_new_losses/pred_dlinknet88_edge33/'
# target_grey = '/content/drive/MyDrive/Road-Network-Inference/ScRoadExtractor/data/test_outputs/'
# os.makedirs(target_grey, exist_ok=True)
# gsd = {
#   'austin': 0.12894113231909,
#   'baltimore': 0.11554327532748307,
#   'denver': 0.11479921265798877,
#   'new_york_city': 0.11316093533352413,
#   'philadelphia': 0.11444294449474908,
#   'portland': 0.10461124236804738,
#   'salt_lake_city': 0.11307931663810175,
#   'san_francisco': 0.11307931663810175,
#   'san_jose': 0.11800316140730151,
#   'seattle': 0.11869674452828635,
#   'washington': 0.10065539530352251
# }
# 

df = pd.read_csv('/content/drive/MyDrive/Road-Network-Inference/data_backup.csv')
output = []
output.append(df.columns.values.tolist() + ["road_width_meters"])
for i, row in df.iterrows():
    
  sat_name = row['sat_name']
  city_name = row['city_name']
  lat = row['n1_lat']
  gsd = 156543.03392 * math.cos(lat * math.pi / 180) / math.pow(2, 20) 
  x1 = int(row['perp1_x'])
  y1 = int(row['perp1_y'])
  x2 = int(row['perp2_x'])
  y2 = int(row['perp2_y'])
  
  
  midpoint = (int((x1+x2)/2), int((y1+y2)/2))
  rr, cc = line(midpoint[0],midpoint[1],x2,y2)
  points1 = list(zip(rr,cc))
  rr, cc = line(midpoint[0],midpoint[1],x1,y1)
  points2 = list(zip(rr,cc))
  # print(points1)
  # print (points2)
  # mask = solver.test_one_img_from_path(testdir + name)
  # cv2_imshow(cv2.imread(maindir + sat_name))
  print(row['lanes'])
  mask, edge = solver.test_one_img_from_path(maindir + sat_name)

  mask_binary = mask.copy()
  mask_binary[mask_binary > 4] = 255
  mask_binary[mask_binary <= 4] = 0
  # color = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2RGB)
  for y, x in points1:
    if (mask_binary[x][y] ==0):
      x1 = x
      y1 = y
      # color = cv2.circle(color, (y,x), 10, [0,255,255], -1)
      break
  for y, x in points2:
  
    if (mask_binary[x][y] ==0):
      x2 = x
      y2 = y
      # color = cv2.circle(color, (y,x), 10, [0,255,255], -1)
      break
  # cv2_imshow(color)
  # print(x1)
  dist = ((x2-x1)**2 + (y2-y1)**2)**.5
  print(row['n1_lat']) 
  print(row['n1_lon'])     
  print
  output.append(row.values.flatten().tolist() + [gsd*dist])
  # cv2.imwrite(target_grey+row[1][17:-7]+"pred_b.png", mask_binary.astype(np.uint8))
  print(output)

np.savetxt('/content/drive/MyDrive/Road-Network-Inference/samples_out.csv', output, delimiter=',', fmt='%s')




