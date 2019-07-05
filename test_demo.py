import os
import cv2
import numpy as np
import argparse

from WeSamBE import WeSamBE
from utils import post_pixel


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--num',default=5,  type=int, help='the video name')
parser.add_argument('--R',default=20,  type=int, help='threshold R')
#parser.add_argument('--ii',default=20,  type=int, help='save numbers frames')
args = parser.parse_args()

num=args.num#5
print(num)
vn=['cross1', 'cross2', 'ncross', 'objects', 'person', 'train2object', 'xiaoche']
rootDir = r'H:/lixingxin/project3/lxx_bs_code/data/'+vn[num]
savepath='results/'+vn[num]
if not os.path.exists(savepath):
    os.makedirs(savepath) 
    print('创建成功')

imglist=os.listdir(rootDir)
imglist.sort()
imglist.sort(key = lambda x: int(x[3:-4]))


image = cv2.imread(os.path.join(rootDir, imglist[0]))
bgs=WeSamBE(image,N=20,R=args.R,_min=2,phai=16)
post_=post_pixel(image.shape[:2])

for ii,lists in enumerate(imglist):
    print('分割:',ii,lists)        
    path = os.path.join(rootDir, lists)
    frame = cv2.imread(path)

    segMap = bgs(frame)
    print(segMap.shape,segMap.dtype)
    post_map,_= post_(segMap)
    
    cv2.imshow('frame', frame)
    cv2.imshow('segMap', segMap)
    cv2.imshow('post_map', post_map)
    #cv2.imwrite(os.path.join(savepath, lists.split('.')[0]+'.png'),segMap)
    if (cv2.waitKey(1) and 0xff == ord('q')) or ii==1000:#args.ii
        break

