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
from PIL import Image
import time
from skimage import metrics
from mayavi import mlab

N0 = 1024
N = 2048
M = N

X0 = Image.open("../data/circle.png").convert('L')
X2 = X0.resize((N0, N0), Image.ANTIALIAS)
X2 = np.array(X2)

M2, N2 = np.shape(X2)
M1 = M2 / 4
N1 = N2

Ui = np.zeros([2048, 2048])
Ui[int(M / 2 - M2 / 2): int(M / 2 + M2 / 2), int(N / 2 - N2 / 2): int(N / 2 + N2 / 2)] = X2
plt.figure("初始")
plt.imshow(abs(Ui))

waveB = 465e-9  # 波长
waveG = 525e-9
waveR = 643e-9
kr = 2 * np.pi / waveR
kg = 2 * np.pi / waveG
kb = 2 * np.pi / waveB

p1 = 8e-6  # 物平面抽样间距，单位微米
p2 = 8e-6  # 衍射面抽样间距，单位微米


def BLAS(Ui, k, z):
    wave = 2 * np.pi / k
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


def gs_iteration_phase(U, kk, z0, times):
    U = U * exp(1j * 2 * pi * np.random.rand(N, N))
    # U = abs(U)
    for i in range(times):
        g1 = 0
        for k in kk:
            g1 = g1 + BLAS(U, k, z0)
        g2 = np.exp(1j * angle(g1))
        gr, gb, gg = g2, g2, g2
        gr[int(5 * M / 8 - M1): int(5 * M / 8 + M1), int(N / 2 - N1 / 2): int(N / 2 + N1 / 2)] = 0

        gb[int(3 * M / 8 - M1): int(3 * M / 8 + M1), int(N / 2 - N1 / 2): int(N / 2 + N1 / 2)] = 0
        gb[int(11 * M / 16 - M1 / 2): int(11 * M / 16 + M1 / 2), int(N / 2 - N1 / 2): int(N / 2 + N1 / 2)] = 0

        gg[int(3 * M / 8 - M1): int(3 * M / 8 + M1), int(N / 2 - N1 / 2): int(N / 2 + N1 / 2)] = 0
        gg[int(9 * M / 16 - M1 / 2): int(9 * M / 16 + M1 / 2), int(N / 2 - N1 / 2): int(N / 2 + N1 / 2)] = 0

        Ir = BLAS(gr, -kr, z0)
        Ib = BLAS(gb, -kb, z0)
        Ig = BLAS(gg, -kg, z0)
        I = Ig + Ir + Ib
        U = abs(Ui) * np.exp(1j * angle(I))
        # print(i)
    return Ir, Ib, Ig, I


def get_psnr_ssim(prev, value):
    psnr = metrics.peak_signal_noise_ratio(prev, array(mat2gray(abs(value)), dtype=uint8))
    ssim = metrics.structural_similarity(prev, array(mat2gray(abs(value)), dtype=uint8))
    return psnr, ssim


flag = 1
# 模拟再现
if flag == 1:
    z0 = 0.3
    times = 1
    start = time.perf_counter()
    kk = [kr, kg, kb]

    Ir, Ib, Ig, I = gs_iteration_phase(Ui, kk, z0, times)

    end = time.perf_counter()
    print("运行耗时", end - start)
    I_crop = I[int(M / 2 - N1 / 2): int(M / 2 + N1 / 2), int(N / 2 - N1 / 2): int(N / 2 + N1 / 2)]  # 裁剪
    # reference = UW[int(M / 2 - N1 / 2): int(M / 2 + N1 / 2), int(N / 2 - N1 / 2): int(N / 2 + N1 / 2)]
    p, s = get_psnr_ssim(X2, I_crop)
    print(p, s)
    AAA = abs(I_crop)
    plt.figure("重建")
    plt.subplot(2, 2, 1)
    plt.imshow(abs(Ir))
    plt.subplot(2, 2, 2)
    plt.imshow(abs(Ig))
    plt.subplot(2, 2, 3)
    plt.imshow(abs(Ib))
    plt.subplot(2, 2, 4)
    plt.imshow(abs(I_crop))
    plt.show()
    # mlab.imshow(abs(I))
    x = np.linspace(-1 / (p2), 1 / (p2), N)
    y = np.linspace(-1 / (p2), 1 / (p2), N)
    mlab.surf(x, y, abs(I), warp_scale="auto")
    mlab.show()

# 生成n张全息图
if flag == 0:
    z0 = 0.3
    times = 5

    for i in range(1):
        RPM = np.exp(1j * 2 * np.pi * np.random.rand(N, N))
        g1 = gs_iteration_phase(Ui, z0, times)[1]
        G1 = g1[int(M / 2 - M1 / 2): int(M / 2 + M1 / 2), int(N / 2 - N1 / 2): int(N / 2 + N1 / 2)]
        SLM_CGH = zeros([1080, 1920])
        M1, N1 = G1.shape
        holo = mod(angle(G1) + pi, 2 * pi) * 255 / (2 * pi)
        SLM_CGH[int(1080 / 2 - M1 / 2):int(1080 / 2 + M1 / 2), int(1920 / 2 - M1 / 2):int(1920 / 2 + M1 / 2)] = holo
        plt.imsave('C:\\SCU\\2021\\UD\\holo\\code\\data\\TEST\\limit_ono\\' + str(i) + 'gNO.png', abs(SLM_CGH),
                   cmap='gray')

# %%
