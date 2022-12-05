from numba import cuda
import math
from PIL import Image
import numpy as np
import struct
import os
import time


# GPU function
@cuda.jit()
def process_gpu(gt, src, coc, kernel_radius, BOKEH_RADIUS):
    dof_x = 0
    dof_y = 0
    dof_z = 0
    dof_w = 0
    tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    for i in range(-BOKEH_RADIUS, BOKEH_RADIUS, 1):
        for j in range(-BOKEH_RADIUS, BOKEH_RADIUS, 1):
            x = tx + i + BOKEH_RADIUS
            y = ty + j + BOKEH_RADIUS
            coc_val = abs(coc[x, y])
            src_val = src[x, y]
            radius = kernel_radius[i, j]
            # CoCWeight
            if radius <= coc_val:
                w = 1.0
            else:
                w = 0.0
            w /= max(coc_val * coc_val, 1.0)
            # dof[:3] += src_val[:3] * w
            # dof[3] += w
            dof_x += src_val[0] * w
            dof_y += src_val[1] * w
            dof_z += src_val[2] * w
            dof_w += w
    if dof_w > 0.0:
        dof_x /= dof_w
        dof_y /= dof_w
        dof_z /= dof_w
    gt[tx, ty][0] = dof_x
    gt[tx, ty][1] = dof_y
    gt[tx, ty][2] = dof_z





if __name__ == '__main__':
    # load src
    src = Image.open("./Color.dds")
    src = np.asarray(src)
    # load coc
    file_path = './coc.bin'
    binfile = open(file_path, 'rb')
    size = int(os.path.getsize(file_path)/4)
    data = struct.unpack('f'*size,binfile.read(size*4))
    binfile.close()
    data =  np.array(data).reshape(-1)
    data1 = data[4:]
    coc = data1.reshape(800, 1280)

    # define var, padding coc and src
    height, width = coc.shape
    BOKEH_RADIUS = 32

    kernel_radius = np.zeros((2*BOKEH_RADIUS+1, 2*BOKEH_RADIUS+1), dtype=np.float)
    for i in range(-BOKEH_RADIUS, BOKEH_RADIUS, 1):
        for j in range(-BOKEH_RADIUS, BOKEH_RADIUS, 1):
            kernel_radius[i, j] = math.sqrt(i**2 + j**2)

    gt = np.zeros((height, width, 3), dtype=np.int_)
    coc = np.pad(coc, ((32, 32), (32, 32)))
    src = np.pad(src, ((32, 32), (32, 32), (0, 0)))

    # GPU setup
    src_gpu = cuda.to_device(src)
    gt_gpu = cuda.to_device(gt)
    coc_gpu = cuda.to_device(coc)
    kernel_radius_gpu = cuda.to_device(kernel_radius)
    threadsperblock = (32, 32)
    blockspergrid_x = int(math.ceil(height/threadsperblock[0]))
    blockspergrid_y = int(math.ceil(width/threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    # GPU device starts processing
    cuda.synchronize()
    start_gpu = time.time()
    process_gpu[blockspergrid, threadsperblock](gt_gpu, src_gpu, coc_gpu, kernel_radius_gpu, BOKEH_RADIUS)
    end_gpu = time.time()
    cuda.synchronize()
    time_gpu = (end_gpu - start_gpu)
    print("GPU process time: " + str(time_gpu))
    gt = gt_gpu.copy_to_host()

    # save
    gt_img = Image.fromarray(np.uint8(gt))
    gt_img.save("test3.png")
