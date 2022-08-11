# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 10:11:56 2021

@author: wu
"""

import numpy as np
from numpy.core import *
from scipy.fftpack import *
import matplotlib.pyplot as plt
from numpy.lib import *
import cv2
from skimage import metrics
from PIL import Image
from scipy import special
import math
import time
import os

path = "C://SCU//2021//UD//holo//code//data//image//lena.bmp"
file_in_path = "C://SCU//2021//UD//holo//code//data//image//"
X0 = Image.open(path).convert('L')

name = os.path.splitext(path)
fro_name, ext_name = name
fro_name = fro_name[len(file_in_path):]

N0 = 1024
X0 = X0.resize((N0, N0), Image.ANTIALIAS)
X0 = np.array(X0)

M1, N1 = np.shape(X0)
N = 2048
M = N
Ui = np.zeros([M, N])
Ui[int(M / 2 - M1 / 2): int(M / 2 + M1 / 2), int(N / 2 - N1 / 2): int(N / 2 + N1 / 2)] = X0

RPM = exp(1j * 2 * pi * np.random.rand(N, N))
U1 = Ui * RPM
# U1 = Ui * exp(1j * 0 * pi * np.random.rand(N, N))

wave = 671e-9  # 波长
k = 2 * np.pi / wave
p1 = 8e-6  # 物平面抽样间距，单位微米
p2 = 8e-6  # 衍射面抽样间距，单位微米


def BLAS(Ui, k, z):
    fx = 1 / (N * p1)  # 频谱抽样间隔
    fy = 1 / (N * p2)

    fx_limit = 1 / (np.sqrt((2 * fx * z) ** 2 + 1) * wave)  # 频谱带宽限制，参考Band-limited AS算法
    fy_limit = 1 / (np.sqrt((2 * fy * z) ** 2 + 1) * wave)

    fx = np.linspace(-1 / (p2 * 2), 1 / (p2 * 2), N)
    fy = np.linspace(-1 / (p2 * 2), 1 / (p2 * 2), N)

    fx, fy = np.meshgrid(fx, fy)
    H_tran = np.exp(1j * k * z * np.sqrt(1 - wave ** 2 * (fx ** 2 + fy ** 2)))  # 角谱传递函数，K取负方向
    Rect_f = np.logical_and(np.abs(fx) < fx_limit, np.abs(fy) < fy_limit)  # 限制频率成分
    H_f = H_tran * Rect_f

    U = ifft2(fft2(Ui) * H_f)
    return U


def IBLAS(Ui, k, z):
    fx = np.linspace(-1 / (p2 * 2), 1 / (p2 * 2), N)
    fy = np.linspace(-1 / (p2 * 2), 1 / (p2 * 2), N)
    fx, fy = np.meshgrid(fx, fy)
    H_tran = np.exp(1j * k * z * np.sqrt(1 - wave ** 2 * (fx ** 2 + fy ** 2)))  # 角谱传递函数，K取负方向
    H_f = H_tran
    U = ifft2(fft2(Ui) * H_f)
    return U


def mat2gray(u1):  # 归一化函数
    u1_max = amax(u1)
    u1_min = amin(u1)
    u2 = 255 * (u1 - u1_min) / (u1_max - u1_min)
    return u2


def blaze(H):  # 闪耀光栅
    M, N = np.shape(H)
    m = np.arange(-M / 2, M / 2)
    n = np.arange(-N / 2, N / 2)
    b = 0
    c = 1
    blaze_phase = np.zeros((N, M))
    for i1 in range(N):
        for j1 in range(M):
            blaze_phase[i1, j1] = 2 * np.pi / 2 * np.mod(b * m[j1] + c * n[i1], 2)
    return blaze_phase


def gs_iteration_phase(U1, z0, times):
    for i in range(times):
        g1 = BLAS(U1, k, z0)
        p = angle(g1)
        g2 = np.exp(1j * p)
        I = BLAS(g2, -k, z0)
        U1 = Ui * np.exp(1j * angle(I))
        # print(i)
    return g2, g1


def add_image(Ui, z0, time):
    recon = np.zeros([N, N])
    for i in range(time):
        RPM = np.exp(1j * 2 * np.pi * np.random.rand(N, N))
        U1 = Ui * RPM
        g1 = BLAS(U1, k, z0)
        p = angle(g1)
        g2 = np.exp(1j * p)
        I = BLAS(g2, k, -z0)
        a = abs(I)
        recon = recon + a
        print(i)
    return recon, g1





def get_psnr_ssim(prev, value):
    psnr = metrics.peak_signal_noise_ratio(prev, array(mat2gray(abs(value)), dtype=uint8))
    ssim = metrics.structural_similarity(prev, array(mat2gray(abs(value)), dtype=uint8))
    # print('PSNR:{}'.format(psnr))
    # print('SSIM:{}'.format(ssim))
    return psnr, ssim


def representation(I, z0, times, runtime):
    I_crop = I[int(M / 2 - M1 / 2): int(M / 2 + M1 / 2), int(N / 2 - N1 / 2): int(N / 2 + N1 / 2)]  # 裁剪
    p, s = get_psnr_ssim(X0, I_crop)

    plt.figure("重建")

    plt.imshow(abs(I_crop), cmap="gray")
    plt.text(0, -60, 'psnr:' + str(p) + ',   ssim:' + str(s))
    plt.text(0, -10, 'run time:' + str(round(runtime, 4)) + 's')
    plt.text(550, -10, 'gs ' + str(times) + ' times, add random phase,' + 'z0=' + str(z0))
    plt.axis('off')


def subplot_psnr_holo(times):
    Y_psnr = []
    Z_psnr = []
    X_z0 = arange(0.1, 20, 0.5)
    for i in X_z0:
        g2 = gs_iteration_phase(U1, i, times)[0]
        I = BLAS(g2, k, -i)
        I_crop = I[int(M / 2 - M1 / 2): int(M / 2 + M1 / 2), int(N / 2 - N1 / 2): int(N / 2 + N1 / 2)]  # 裁剪
        Y_psnr.append(get_psnr_ssim(X0, I_crop)[0])
        Z_psnr.append(get_psnr_ssim(X0, I_crop)[1])
    plt.plot(X_z0, Y_psnr, X_z0, Z_psnr)

    plt.xlabel("Z0")
    plt.ylabel("PSNR")
    plt.show()


flag = 1
# 模拟再现
if flag == 1:
    z0 = 0.3
    times = 1
    start = time.perf_counter()

    """
    g2 = gs_iteration_phase(U1, z0, times)[0]
    end = time.perf_counter()
    print("运行耗时", end - start)
    I = BLAS(g2, k, -z0)
    representation(I, z0, times, (end - start))
    """

    subplot_psnr_holo(times)

# 时分复用生成300张全息图
if flag == 0:
    z0 = 0.3
    times = 1
    num = 300

    for i in range(num):
        print(i)
        RPM = np.exp(1j * 2 * np.pi * np.random.rand(N, N))
        U1 = Ui * RPM
        g1 = gs_iteration_phase(U1, z0, times)[1]
        G1 = g1[int(M / 2 - M1 / 2): int(M / 2 + M1 / 2), int(N / 2 - N1 / 2): int(N / 2 + N1 / 2)]
        blaze_phase = blaze(G1)  # 闪耀光栅
        SLM_CGH = zeros([1080, 1920])
        M1, N1 = G1.shape
        holo = mod(angle(G1) + pi + blaze_phase, 2 * pi) * 255 / (2 * pi)

        # holo = mod(angle(G1) + pi, 2 * pi) * 255 / (2 * pi) # 不加闪耀光栅

        SLM_CGH[int(1080 / 2 - M1 / 2):int(1080 / 2 + M1 / 2), int(1920 / 2 - M1 / 2):int(1920 / 2 + M1 / 2)] = holo
        # plt.imsave('C:\\SCU\\2021\\UD\\holo\\code\\data\\TEST\\gs_n_or_tmd_no\\'
        #            +'gs'+str(times)+ str(fro_name) + str(i) + 'z'+str(z0)+'.png',
        #            abs(SLM_CGH), cmap='gray')
        plt.imsave('C:\\SCU\\2021\\UD\\holo\\code\\data\\TEST\\TDM\\Z0P3NOGS\\'
                   + str(fro_name) + str(i) + 'z0p3.png',
                   abs(SLM_CGH), cmap='gray')

plt.show()
