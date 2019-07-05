import numpy as np
import os
import cv2

class WeSamBE(object):

    def __init__(self,I, N=20, R=20, _min=2, phai=16, incen_num=5):
        self.h, self.w=I.shape[:2]
        I_pad = np.pad(I, 1, 'symmetric')
        height,width = I_pad.shape[0],I_pad.shape[1]
        channel=I.shape[2]
        samples = np.zeros((N,height,width,channel))
        
        self.weight=np.zeros((N,self.h,self.w))

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                for n in range(N):
                    x = np.random.randint(-1, 2)
                    y = np.random.randint(-1, 2)
                    ri = i + x
                    rj = j + y
                    samples[n,i, j,:] = I_pad[ri, rj,:]
                a=np.random.choice(range(N), incen_num, replace=False)
                self.weight[a,i-1,j-1]=1
                
        self.samples = samples[:,1:height-1, 1:width-1,:]
        self.N=N
        self.R=R
        self._min=_min
        self.phai=phai
        

    def __call__(self,I_color):
        height,width=self.h,self.w
        segMap = np.zeros((height,width)).astype(np.uint8)
        t=np.zeros((self.N,height,width))

        dist=np.sum(np.abs(I_color.astype(np.int)-self.samples.astype(np.int)),3)
        t[dist<self.R]=1
        weight_sum=np.sum(self.weight*t,0)
        segMap[weight_sum<self._min]=255
        #reward-penalty
        gama=np.sum(t,0)
        incre=(1/gama)*(t==1)
        dec=(1/(self.N-gama))*(t==0)
        self.weight+=incre
        self.weight-=dec

        
        rand_neigh_p=np.random.randint(-1,2,size=(height,width,2))
        rand_update=np.random.randint(0,self.phai,size=(height,width))
        back_pixels=segMap==0
        update_sample=rand_update==0
        update_pixels=np.where(back_pixels & update_sample)
    
        for item in zip(update_pixels[0],update_pixels[1]):
            i,j=item[0],item[1]
            index=np.argmin(self.weight[:,i,j])
            if not np.isscalar(index):
                index=index[0] 
           
            self.samples[index,i,j,:] = I_color[i,j,:]
            self.weight[:,i,j]-=(1/self.N)
            self.weight[index,i,j]+=1
            
            #spatial diffusion
            ri = i + rand_neigh_p[i,j,0]
            rj = j + rand_neigh_p[i,j,1]
            if ri>=height: ri=height-1
            if rj>=width: rj=width-1
            neigh_index=np.argmin(self.weight[:,ri,rj])
            if not np.isscalar(neigh_index):
                neigh_index=neigh_index[0]

            self.samples[neigh_index,ri, rj, :] = I_color[ri, rj,:]

            self.weight[:,ri,rj]-=(1/self.N)
            self.weight[neigh_index,ri,rj]+=1
        
        return segMap



if __name__=='__main__' :
    import argparse
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


    for ii,lists in enumerate(imglist):
        print('分割:',ii,lists)        
        path = os.path.join(rootDir, lists)
        frame = cv2.imread(path)

        segMap = bgs(frame)
        
        cv2.imshow('frame', frame)
        #cv2.imwrite(os.path.join(savepath, lists.split('.')[0]+'.png'),segMap)
        if (cv2.waitKey(1) and 0xff == ord('q')) or ii==1000:#args.ii
            break
    
 