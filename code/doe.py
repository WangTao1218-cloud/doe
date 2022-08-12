# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 10:11:56 2021

@author: wu\wang
"""
import numpy as np
from numpy.core import *
from scipy.fftpack import *
import matplotlib.pyplot as plt
from numpy.lib import *
from PIL import Image
import time
from skimage import metrics

N0 = 1024
N = 2048
M = N
# X0 = 1/3 * np.ones([int(N0/4), N0])
X0 = Image.open("C://SCU//2021//UD//holo//code//data//image//lena.jpg").convert('L')
X2 = X0.resize((N0, N0), Image.ANTIALIAS)
X2 = np.array(X2)
M2, N2 = np.shape(X2)
Ui = np.zeros([2048, 2048])
Ui[int(M / 2 - M2 / 2): int(M / 2 + M2 / 2), int(N / 2 - N2 / 2): int(N / 2 + N2 / 2)] = X2

X1 = X0.resize((int(N0), int(N0 / 4)), Image.ANTIALIAS)
X1 = np.array(X1)
M1, N1 = np.shape(X1)

UR = np.zeros([M, N])
UB = np.zeros([M, N])
UG = np.zeros([M, N])
UR[int(5 * M / 16 - M1 / 2): int(5 * M / 16 + M1 / 2), int(N / 2 - N1 / 2): int(N / 2 + N1 / 2)] = X1
UR[int(7 * M / 16 - M1 / 2): int(7 * M / 16 + M1 / 2), int(N / 2 - N1 / 2): int(N / 2 + N1 / 2)] = X1
UB[int(9 * M / 16 - M1 / 2): int(9 * M / 16 + M1 / 2), int(N / 2 - N1 / 2): int(N / 2 + N1 / 2)] = X1
UG[int(11 * M / 16 - M1 / 2): int(11 * M / 16 + M1 / 2), int(N / 2 - N1 / 2): int(N / 2 + N1 / 2)] = X1
UW = UB + UG + UR

plt.figure("初始")
plt.subplot(2, 3, 1)
plt.imshow(UR)
plt.subplot(2, 3, 2)
plt.imshow(UB)
plt.subplot(2, 3, 3)
plt.imshow(UG)
plt.subplot(2, 3, 4)
plt.imshow(UW)
plt.subplot(2, 3, 5)
plt.imshow(Ui)

RPM = exp(1j * 2 * pi * np.random.rand(N, N))

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


def gs_iteration_phase(U, k, z0, times):
    U = U * exp(1j * 2 * pi * np.random.rand(N, N))
    # U = abs(U)
    for i in range(times):
        g1 = BLAS(U, k, z0)
        g2 = np.exp(1j * angle(g1))
        I = BLAS(g2, -k, z0)
        U = abs(Ui) * np.exp(1j * angle(I))
        # print(i)
    return g2, g1


def get_psnr_ssim(prev, value):
    psnr = metrics.peak_signal_noise_ratio(prev, array(mat2gray(abs(value)), dtype=uint8))
    ssim = metrics.structural_similarity(prev, array(mat2gray(abs(value)), dtype=uint8))
    return psnr, ssim


def representation(I, z0, times, runtime):
    I_crop = I[int(M / 2 - N1 / 2): int(M / 2 + N1 / 2), int(N / 2 - N1 / 2): int(N / 2 + N1 / 2)]  # 裁剪
    reference = UW[int(M / 2 - N1 / 2): int(M / 2 + N1 / 2), int(N / 2 - N1 / 2): int(N / 2 + N1 / 2)]
    p, s = get_psnr_ssim(reference, I_crop)

    print(runtime)
    plt.figure("重建")
    plt.subplot(2, 2, 1)
    plt.imshow(abs(I_crop), cmap="gray")
    plt.text(0, -60, 'psnr:' + str(p) + ',   ssim:' + str(s))
    plt.text(0, -10, 'run time:' + str(round(runtime, 4)) + 's')
    plt.text(550, -10, 'z0=' + str(z0))  ## '叠加 ' + str(times) + ' times,' +  add random phase
    # plt.axis('off')


flag = 1
# 模拟再现
if flag == 1:
    z0 = 0.3
    key = 0
    times = 2
    start = time.perf_counter()

    if key == 0:
        # #先舍后加
        # gR = gs_iteration_phase(UR, kr, z0, times)[0]  # 舍弃振幅
        # gB = gs_iteration_phase(UB, kb, z0, times)[0]  # 舍弃振幅
        # gG = gs_iteration_phase(UG, kg, z0, times)[0]  # 舍弃振幅
        # g = gR + gB + gG

        # 1次 先加后舍
        gR = gs_iteration_phase(UR, kr, z0, times)[1]  # 振幅
        gB = gs_iteration_phase(UB, kb, z0, times)[1]  # 振幅
        gG = gs_iteration_phase(UG, kg, z0, times)[1]  # 振幅
        g = np.exp(1j * angle(gR + gB + gG))
        end = time.perf_counter()

        print("运行耗时", end - start)
        Ir = BLAS(g, kr, -z0)
        Ib = BLAS(g, kb, -z0)
        Ig = BLAS(g, kg, -z0)
        I = Ig + Ir + Ib
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
        plt.imshow(abs(I_crop), cmap='gray')

        # representation(I, z0, times, end - start)

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

plt.show()

# %%

