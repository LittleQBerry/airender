import torch 
import torch.nn.functional as F 
import numpy as np
from PIL import Image
from torchvision import transforms
import time

def getGaussianKernel(kernel_size , sigma =0):
    if sigma <=0:
        sigma = 0.3 * ((kernel_size -1 ) * 0.5 -1) +0.8

    center = kernel_size //2 
    xs =(np.arange(kernel_size ,dtype = np.float32) -center)
    kernel1d = np.exp (-(xs**2)/(2* sigma**2))
    kernel = kernel1d[..., None] @ kernel1d[None,...]
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum()
    print(kernel)
    return kernel

def GaussianBlur(batch_img, kernel_size, sigma=None):
    kernel = getGaussianKernel(kernel_size, sigma) # 生成权重
    B, C, H, W = batch_img.shape # C：图像通道数，group convolution 要用到
    # 生成 group convolution 的卷积核
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(C, 1, 1, 1)
    pad = (kernel_size - 1) // 2 # 保持卷积前后图像尺寸不变
    # mode=relfect 更适合计算边缘像素的权重
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')
    weighted_pix = F.conv2d(batch_img_pad, weight=kernel, bias=None, 
                           stride=1, padding=0, groups=C)
    return weighted_pix

def bifilter(batch_img, kernel_size , sigmaColor =None , sigmaSpace=None):
    device =batch_img.device
    if sigmaSpace is None:
        sigmaSpace =0.15 * kernel_size +0.35
    if sigmaColor is None:
        sigmaColor =sigmaSpace
    
    pad = (kernel_size -1) // 2
    batch_img_pad = F.pad(batch_img,pad =[pad,pad,pad,pad] ,mode = 'reflect' )
    
    patches = batch_img_pad.unfold(2,kernel_size,1).unfold(3,kernel_size,1)
    pathch_dim = patches.dim()
    
    diff_color = patches - batch_img.unsqueeze(-1).unsqueeze(-1)
    weights_color  = torch.exp (-(diff_color **2 ) /(2* sigmaColor **2))
    weights_color = weights_color /weights_color.sum(dim=(-1,-2),keepdim=True)

    weights_space =getGaussianKernel(kernel_size,sigmaSpace).to(device)
    weights_space_dim =(pathch_dim - 2) *(1,) +(kernel_size,kernel_size)
    weights_space = weights_space.view(*weights_space_dim).expand_as(weights_color)

    # weights 
    weights = weights_color *weights_space
    weights_sum = weights.sum(dim =(-1,-2))

    weights_pix = (weights * patches).sum(dim=(-1,-2)) /weights_sum

    return weights_pix

import struct
import os
def loadCoc(path,image):
    binfile = open(path, 'rb')
    size = int(os.path.getsize(path)/4)
    data =struct.unpack('f'*size, binfile.read(size*4))
    binfile.close()
    data =np.array(data).reshape(-1)
    data1 =data[4:]
    B,C,W,H = image.shape
    data1 =data1.reshape(1,W,H)

    return data1

def radius(kernel_size,batch_coc):
    #W,H =a.shape
    center_w = (kernel_size-1) //2
    center_h = (kernel_size-1) //2
    dis_numpy = np.ones((kernel_size,kernel_size))

    for i in range (kernel_size):
        for j in range(kernel_size):
            dis_numpy[i,j] =np.sqrt((i-center_w) **2 +(j-center_h) **2)
    dis_tensor =torch.tensor(dis_numpy)
    batch_dis =torch.empty(batch_coc.shape[2],batch_coc.shape[3],kernel_size,kernel_size)
    print(batch_dis.shape)
    for i in range(batch_dis.shape[0]):
        for j in range(batch_dis.shape[1]):
            batch_dis[i][j] =dis_tensor
    
    batch_dis = batch_dis.reshape(1,1,batch_dis.shape[0],batch_dis.shape[1],kernel_size,kernel_size)
    print(batch_dis.shape)
    return batch_dis
# non-conv2d 

def dof_gt(batch_img,batch_coc,hoken_size,batch_d,device):
    dof =0.0
    pad = (hoken_size -1) // 2
    batch_img = batch_img.to(device)
    batch_coc =batch_coc.to(device)
    batch_d =batch_d.to(device)
    batch_img_pad =F.pad(batch_img,pad =[pad,pad,pad,pad] ,mode ='reflect')
    batch_coc_pad =F.pad(batch_coc,pad=[pad,pad,pad,pad],mode="reflect")
    print(batch_img_pad.shape)
    patches = batch_img_pad.unfold(2,hoken_size,1).unfold(3,hoken_size,1)
    pathch_dim = patches.dim()
    patches_coc = batch_coc_pad.unfold(2,hoken_size,1).unfold(3,hoken_size,1)
    pathchc_dim =patches_coc.dim()
    print(patches.shape)
    print(patches_coc.shape)
    patches_coc =abs(patches_coc)
    #batch_d = batch_d
    #coc_weight = (patches_coc-batch_d +2.0) /2.0
    n1 = torch.zeros_like(patches_coc).to(device)
    n2 = torch.ones_like(patches_coc).to(device)
    coc_weight = torch.where(patches_coc>=batch_d,n2,n1)
    #nn_tensor =torch.ones_like(patches_coc).to(device)
    coc_weight =coc_weight / torch.max(patches_coc * patches_coc, n2)
    weights_sum = coc_weight.sum(dim =(-1,-2))
    dof_xyz = (patches * coc_weight).sum(dim=(-1,-2)) /weights_sum
    return dof_xyz

    #distance = radius(hoken_size)
    #distance = 
    #for i in range()
def dof_edge(batch_img,batch_coc,edge_t,hoken_size,batch_d,device):
    pad = (hoken_size -1) //2
    batch_img =batch_img.to(device)
    batch_coc =batch_coc.to(device)
    batch_d = batch_d.to(device)
    batch_edge = edge_t.to(device)
    batch_img_pad =F.pad(batch_img,pad =[pad,pad,pad,pad] ,mode ='reflect')
    batch_coc_pad =F.pad(batch_coc,pad=[pad,pad,pad,pad],mode="reflect")
    batch_edge_pad = F.pad(batch_edge,pad =[pad,pad,pad,pad], mode ='reflect')
    patches_img = batch_img_pad.unfold(2,hoken_size,1).unfold(3,hoken_size,1)
    patches_coc = batch_coc_pad.unfold(2,hoken_size,1).unfold(3,hoken_size,1)
    patches_edge =batch_edge_pad.unfold(2,hoken_size,1).unfold(3,hoken_size,1)
    zeros = torch.zeros_like(patches_coc).to(device)
    ones =torch.ones_like(patches_coc).to(device)
    
    coc_weight = torch.where(patches_coc>=batch_d,ones,zeros)
    coc_weight = coc_weight /torch.max(patches_coc* patches_coc,ones)

    weights =coc_weight * patches_edge
    weights_sum = weights.sum(dim=(-1,-2))
    dof_xyz = (patches_img * weights).sum(dim=(-1,-2)) /weights_sum

    return dof_xyz

#def Dof_gt(image,Coc):


if __name__ == '__main__':
    img_path = '/home3/qinyiming/airender/data/Color.png'
    edge_path = '/home3/qinyiming/airender/data/Color_edge.png'
    device = 'cpu'
    img = Image.open(img_path)
    img =img.convert("RGB")

    img_edge = Image.open(edge_path)
    img_edge = img_edge.convert("L")
    transforms_data = transforms.Compose([
        #transforms.Resize([])
        transforms.ToTensor()
    ])
    img_t = transforms_data(img)
    img_t =img_t.reshape(1, img_t.shape[0],img_t.shape[1],img_t.shape[2])

    edge_t= transforms_data(img_edge)
    edge_t =edge_t.reshape(1, edge_t.shape[0],edge_t.shape[1], edge_t.shape[2])
    #edge_t = (edge_t /255.0 )
    print(torch.max(edge_t))
    #exit()
    '''
    #bi-filter
    weight = bifilter(img_t, kernel_size=15, sigmaColor =0.15, sigmaSpace=5)
    #print(weight)
    print(weight.type)
    image =weight.cpu().clone()
    image =image.squeeze(0)
    print(image.type)
    new_image = transforms.ToPILImage()(image)
    #print(new_image)
    new_image.save('test.png')
    
    #guassianfilter
    gua_weight = GaussianBlur(img_t,kernel_size=15,sigma=3)
    gua_weight =gua_weight.cpu().clone()
    gua_image =gua_weight.squeeze(0)
    gua_image = transforms.ToPILImage()(gua_image)
    gua_image.save("test2.png")
    '''
    coc_data = loadCoc("/home3/qinyiming/airender/data/CoC.bin",img_t)
    coc_batch =torch.tensor(coc_data)
    coc_batch = coc_batch.reshape(1, coc_batch.shape[0],coc_batch.shape[1], coc_batch.shape[2])
    coc_batch =abs(coc_batch)
    boken= 35
    batch_d =radius(boken,coc_batch)
    start=time.time()
    #new_image = dof_gt(img_t,coc_batch,boken,batch_d,device)

    new_image = dof_edge(img_t,coc_batch,edge_t,boken,batch_d,device)
    end =time.time()
    print("dof:%.0f"%(end-start))
    new_image = new_image.cpu().clone()
    new_image =new_image.squeeze(0)
    new_image =transforms.ToPILImage()(new_image)
    new_image.save("dof.png")
    #print(new_image.shape)
    

