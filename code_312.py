# -*- coding: utf-8 -*-
"""
可运行
差最后提取特征的方法
"""

import matplotlib.pyplot as plt
import pylab
import cv2
import numpy as np
import os
import sys
import random
import shutil
def thred(img,thre=100):
    res=(img>thre)*255
    return res   
def dokernel(img,kernel,stride):
    #a=dokernel(img,kernel=[6,1],stride=[6,1])
    '''
    #无权重kernel扫描图片
    #kernel=np.ones
    #padding类似valid
    #(h,w)>LetfUp_kernel   
    '''
    img=img/255
    ker_h=kernel[0]
    ker_w=kernel[1]
    strd_h=stride[0]
    strd_w=stride[1]
    height=img.shape[0]//strd_h
    weight=img.shape[1]//strd_w  
    fumap=np.zeros((height,weight))
    for h in range(height):
        for w in range(weight):
            fumap[h,w]=sum(sum(img[strd_h*h:strd_h*h+ker_h,strd_w*w:strd_w*w+ker_w]))*255/(ker_h*ker_w)

    return fumap
#def DirectFilter(img): 
#    '''
#    方向滤波器：3个方向
#    '''
#    r=np.reshape(np.mean(img,axis=1),(-1)) #横向    
#    c=np.reshape(np.mean(img,axis=0),(-1)) #竖向
#    
#    height=img.shape[0]
#    weight=img.shape[1]     
#    diag=np.zeros((height+weight-1)) #对角线向
#    count=0
#    for i in range(-height+1,weight-1):
#        temp=np.mean(img.diagonal(offset=i))
#        diag[count]=temp
#        count+=1
#    return r,c,diag
#

def Count01_c(imglist):
    
    '''
    对imglist进行处理:
    分别记录连续0/1个数
    '''
    zeroslist=[]  #按行列的zeros
    oneslist=[]
    n_zeroslist=np.zeros((len(imglist),len(imglist[0][0])))
    n_oneslist=np.zeros((len(imglist),len(imglist[0][0])))
    for PIC in range(len(imglist)):       
        img=imglist[PIC]
        N_ZEROs=np.zeros((len(img),len(img[0])),dtype='int32')
        N_ONEs=np.zeros((len(img),len(img[0])),dtype='int32')
        
        for i in range(len(img[0])):
            C_0=0
            N_0=0
            C_1=0
            N_1=0
            for j in range(len(img)):
                if img[j,i]==0:          #pix=0,计数
                    C_0+=1
                elif C_0!=0:         #pix变1,写入数据,清空计数器
                    N_ZEROs[N_0,i]=C_0
                    C_0=0
                    N_0+=1
                
                if img[j,i]==255:       #pix=255,计数
                    C_1+=1
                elif C_1!=0:        #pix变0,写入数据,清空计数器
                    N_ONEs[N_1,i]=C_1
                    C_1=0
                    N_1+=1
                    
            n_zeroslist[PIC,i]=(N_0)
            n_oneslist[PIC,i]=(N_1)
            
            N_ZEROs[0,i]=0
            N_ZEROs[N_0-1,i]=0
            N_ONEs[0,i]=0
            N_ONEs[N_0-1,i]=0
        zeroslist.append(N_ZEROs)     
        oneslist.append(N_ONEs)
        
    return zeroslist,oneslist,n_zeroslist,n_oneslist

def Count01_r(imglist):
    '''
    对imglist进行处理:
    分别记录连续0/1个数(横向)
    #收尾均去除以减小误差
    input:imglist:list[img np array(0/255)]
    output:
        zeroslist:list[np array(一张图片连续为0的计数矩阵)]
        oneslist,
        n_zeroslist:np array[一张图片][一行上0像素块 块数]
        n_oneslist
    '''
    zeroslist=[]  #按行列的zeros
    oneslist=[]
    n_zeroslist=np.zeros((len(imglist),len(imglist[0])))
    n_oneslist=np.zeros((len(imglist),len(imglist[0])))
    for PIC in range(len(imglist)):
        img=imglist[PIC]    
        N_ZEROS=np.zeros((len(img),len(img[0])),dtype='int32')
        N_ONES=np.zeros((len(img),len(img[0])),dtype='int32')
        for j in range(len(img)):
            C_0=0
            N_0=0
            C_1=0
            N_1=0
            for i in range(len(img[0])):
                if img[j,i]==0:          #pix=0,计数
                    C_0+=1
                elif C_0!=0:         #pix变1,写入数据,清空计数器
                    N_ZEROS[j,N_0]=C_0
                    C_0=0
                    N_0+=1
                
                if img[j,i]==255:       #pix=255,计数
                    C_1+=1
                elif C_1!=0:        #pix变0,写入数据,清空计数器
                    N_ONES[j,N_1]=C_1
                    C_1=0
                    N_1+=1
            n_zeroslist[PIC,j]=(N_0)
            n_oneslist[PIC,j]=(N_1)
            N_ZEROS[j,0]=0
            N_ZEROS[j,N_0-1]=0
            N_ONES[j,0]=0
            N_ONES[j,N_1-1]=0
        zeroslist.append(N_ZEROS)     
        oneslist.append(N_ONES)
           
    return zeroslist,oneslist,n_zeroslist,n_oneslist
    

#def DirectFilter2(img): 
#    '''
#    方向滤波器：3个方向
#    '''
#    r=np.reshape(np.mean(img,axis=1),(-1)) #横向    
#    c=np.reshape(np.mean(img,axis=0),(-1)) #竖向    
#    height=img.shape[0]
#    weight=img.shape[1]     
#    diag=np.zeros((height+weight-1)) #对角线向
#    count=0
#    for i in range(-height+1,weight-1):
#        temp=np.mean(img.diagonal(offset=i))
#        diag[count]=temp
#        count+=1
#    return r,c,diag

    
#%%
##保存数组   np.save('f:/r.npy',r)
#img=imglist[0]
#height=img.shape[0]
#weight=img.shape[1]  
#thistime=[]
#for i in range(-height+1,weight-1):
#    temp=img.diagonal(offset=i)
#    thistime.append(temp)
#%% 1-获取good data  
'''
L_thr_P:list[np array(postive thred后图片)]
'''
L_thr_P=[]
L_ers_P=[]
name_path=['F:/codeanddata/data_initial/good620','F:/codeanddata/data_initial/badall',  ##src
      'F:/codeanddata/data_initial/res_good/','F:/codeanddata/data_initial/res_bad/']##res
name_Sufx=['-asrc.bmp','-bfil.bmp','-cthr.bmp','-ders.bmp']
floderdir=os.listdir(name_path[0])
count=0
for F in floderdir:
    picpath=name_path[0]+'/'+F
    picdir=os.listdir(picpath)    
    k=np.ones((1,2), np.uint8)
    for P in picdir[:100]:
        lPic=[]
        lPic.append(cv2.imread(picpath+'/'+P,0)) #src
        lPic.append(dokernel(lPic[0],kernel=[6,1],stride=[6,1]))#fil
        lPic.append(thred(lPic[1],thre=100))        #thr   
    '''
    添加腐蚀处理
    '''
        lPic.append(cv2.erode(np.uint8(lPic[2]),k,iterations = 1))       
        for i in range(4):
            cv2.imwrite(name_path[2]+str(count)+name_Sufx[i],lPic[i])
        L_thr_P.append(lPic[2])  #thr
        L_ers_P.append(lPic[3])  #ers
        count+=1
        
##获取bad data  
'''
(imgdir)->最终有序path
L_thr_N:list[np array(negative thred后图片)]
'''
L_thr_N=[]
L_ers_N=[]
floderdir=os.listdir(name_path[1])

imgdir=[]
for F in floderdir:
    picpath=name_path[1]+'/'+F
    temp=os.listdir(picpath)
    for i in temp:
        imgdir.append(picpath+'/'+i)
random.shuffle(imgdir)

count=0
for F in imgdir[:300]:   
    lPic=[]
    lPic.append(cv2.imread(F,0)) #src
    lPic.append(dokernel(lPic[0],kernel=[6,1],stride=[6,1]))#fil
    lPic.append(thred(lPic[1],thre=100))        #thr 
    '''
    添加腐蚀处理
    '''
    lPic.append(cv2.erode(np.uint8(lPic[2]),k,iterations = 1))           
    for i in range(4):
        cv2.imwrite(name_path[3]+str(count)+name_Sufx[i],lPic[i])
    L_thr_N.append(lPic[2])  #thr
    L_ers_N.append(lPic[3])  #ers
    count+=1
        
#%% 2-获取统计数据
aR0_P,aR1_P,aR0_n_P,aR1_n_P=Count01_r(L_ers_P)
aR1_P,aR1_P,aC0_n_P,aC1_n_P=Count01_c(L_ers_P)
aR0_N,aR1_N,aR0_n_N,aR1_n_N=Count01_r(L_ers_N)
aC0_N,aC1_N,aC0_n_N,aC1_n_N=Count01_c(L_ers_N)
#%% 3-分析数据
#获取hist
##P

AHist_R0_P=[]
AHist_R1_P=[]
AHist_C0_P=[]
AHist_C1_P=[]
L_Count_P=[aR0_P,aR1_P,aR1_P,aR1_P]
L_Hist_P=[AHist_R0_P,AHist_R1_P,AHist_C0_P,AHist_C1_P]

for l in range(4):    
    for i in range(len(aR0_P)):
        hist=np.bincount(L_Count_P[l][i].reshape((-1))) 
        L_Hist_P[l].append(hist)

##N
AHist_R0_N=[]
AHist_R1_N=[]
AHist_C0_N=[]
AHist_C1_N=[]
L_Count_N=[aR0_N,aR1_N,aC0_N,aC1_N]
L_Hist_N=[AHist_R0_N,AHist_R1_N,AHist_C0_N,AHist_C1_N]

for l in range(4):    
    for i in range(len(aR0_N)):
        hist=np.bincount(L_Count_N[l][i].reshape((-1))) 
        L_Hist_N[l].append(np.reshape(hist,(-1)))

###分析1 加和:
list_c=[]
for i in L_Hist_P:
    count_p=np.zeros((100))
    for j in range(len(i)): 
        count=0
        for k in i[j]:
            count_p[count]+=k
            count+=1
    count_p[0]=0
    list_c.append(count_p)
            
a=count_p[:40]            
            
######
        
        
        
        
        
        
        
        
###### 分析连续个数数据:
### 方法1:print n range:     
#lnum=[aR0_n_P,aR1_n_P,aC0_n_P,aC1_n_P,aR0_n_N,aR1_n_N,aC0_n_N,aC1_n_N]
#lnumName=['aR0_n_P','aR1_n_P','aC0_n_P','aC1_n_P','aR0_n_N','aR1_n_N','aC0_n_N','aC1_n_N']
#print('[下行的范围],[上行的范围]:')
#for l in range(len(lnum)):
#    temp=np.c_[np.min(lnum[l],axis=1),np.max(lnum[l],axis=1)]
#    res=[[np.min(temp[:,0]),np.max(temp[:,0])],[np.min(temp[:,1]),np.max(temp[:,1])]]
#    print(lnumName[l],'->',res)
#'''
#[下行的范围],[上行的范围]:
#aR0_n_P -> [[69.0,], [101.0]]
##aR1_n_P -> [[68.0, 77.0], [100.0, 137.0]]
#aC0_n_P -> [[0.0, 0.0], [14.0, 32.0]]
##aC1_n_P -> [[0.0, 0.0], [15.0, 32.0]]
#
#aR0_n_N -> [[0.0,], [28.0]]
##aR1_n_N -> [[0.0, 78.0], [27.0, 126.0]]
#aC0_n_N -> [[0.0, 1.0], [10.0, 36.0]]
##aC1_n_N -> [[0.0, 1.0], [10.0, 36.0]]
#'''   
## 方法2:hist
lnum=[aR0_n_P,aC0_n_P,aR0_n_N,aC0_n_N]
num_hist=[]
diff=[]
for i in lnum:
    hist=np.bincount(np.uint8(i).reshape((-1))) 
    num_hist.append(hist)
## 方法3:计算单幅pic中不同行列差最大值
for i in lnum:
    c=np.max(i,axis=1)-np.min(i,axis=1)
    diff.append(c)
#%% test阈值
tp=[]
fp=[]
tn=[]
fn=[] 
###good
for i in range(len(L_thr_P)):
    if (sum(L_Hist_P[0][i][7:])<20
        and sum(L_Hist_P[1][i][5:])<20
        and sum(L_Hist_P[2][i][10:])<20
        and sum(L_Hist_P[3][i][8:])<20
        and sum(100<aR0_n_P[i])<10
        and sum(aR0_n_P[i]<70)<10
        ) :
        tp.append(i)
    else:
        fp.append(i)  
for i in range(len(L_thr_N)):
    if (sum(L_Hist_N[0][i][7:])<20
        and sum(L_Hist_N[1][i][5:])<20
        and sum(L_Hist_N[2][i][10:])<20
        and sum(L_Hist_N[3][i][8:])<20
        and sum(100<aR0_n_N[i])<10
        and sum(aR0_n_N[i]<70)<10
        ) :
        fn.append(i)
    else:
        tn.append(i)
#a=np.zeros((100,6))    
#for i in range(100):
#    a[i,0]=sum(L_Hist_P[0][i][7:])
#    a[i,1]=sum(L_Hist_P[1][i][5:])
#    a[i,2]=sum(L_Hist_P[2][i][10:])
#    a[i,3]=sum(L_Hist_P[3][i][8:])
#    a[i,4]=sum(100<aR0_n_P[i])
#    a[i,5]=sum(aR0_n_P[i]<70)
#%%check time
newpathlist=['f:/fp/','f:/tn/','f:/fn/']
##good 
goodpath=path[2]
newpath='f:/fp/'
for i in range(len(hist1_g)):
    if i not in tp:
        shutil.copyfile(os.path.join(path,str(i)+'-asrc.bmp'),os.path.join(newpath,str(i)+'-asrc.bmp'))
        shutil.copyfile(os.path.join(path,str(i)+'-bfil.bmp'),os.path.join(newpath,str(i)+'-bfil.bmp'))
        shutil.copyfile(os.path.join(path,str(i)+'-cthr.bmp'),os.path.join(newpath,str(i)+'-cthr.bmp'))

##bad
badpath=name_path[3]
newpath='f:/tn/'
newpath_n='f:/fn/'
for i in range(len(hist1_b)):
    if i in tn:
        shutil.copyfile(os.path.join(path,str(i)+'-asrc.bmp'),os.path.join(newpath,str(i)+'-asrc.bmp'))
        shutil.copyfile(os.path.join(path,str(i)+'-bfil.bmp'),os.path.join(newpath,str(i)+'-bfil.bmp'))
        shutil.copyfile(os.path.join(path,str(i)+'-cthr.bmp'),os.path.join(newpath,str(i)+'-cthr.bmp')) 
    else:
        print(i)
        shutil.copyfile(os.path.join(path,str(i)+'-asrc.bmp'),os.path.join(newpath_n,str(i)+'-asrc.bmp'))
        shutil.copyfile(os.path.join(path,str(i)+'-bfil.bmp'),os.path.join(newpath_n,str(i)+'-bfil.bmp'))
        shutil.copyfile(os.path.join(path,str(i)+'-cthr.bmp'),os.path.join(newpath_n,str(i)+'-cthr.bmp')) 
 
x=range(31)
plt.plot(x,np.r_[num_hist[1],[0,0]],'b')       
plt.plot(x,num_hist[3],'k')