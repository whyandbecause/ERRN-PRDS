import numpy as np
import math
import os
import cv2



# =============================================================================
# def psnr(im1,im2):
#     im1 = im1*255.
#     im2 = im2*255.
#     diff = np.abs(im1 - im2)
#     rmse = np.sqrt(diff).sum()
#     psnr = 20*np.log10(255./rmse)
#     return psnr
# =============================================================================



# def psnr2(target, ref):
#     # target:目标图像  ref:参考图像
#     # assume RGB image
#     target_data = np.array(target)
#     target_data = target_data[0:-1,0:-1]
#
#     ref_data = np.array(ref)
#     ref_data = ref_data[0:-1,0:-1]
#
#     diff = ref_data - target_data
#     diff = diff.flatten('C')
#     rmse = math.sqrt( np.mean(diff ** 2.) )
#     return 20*math.log10(255.0/rmse)
#
#
# def psnr(target, ref):
#     # target:目标图像  ref:参考图像
#     # assume RGB image
#     target_data = np.array(target)
# #    target_data = target_data[0:-1,0:-1]
#
#     ref_data = np.array(ref)
# #    ref_data = ref_data[0:-1,0:-1]
#
#     diff = ref_data - target_data
#     diff = diff.reshape(-1, 1)
#     mse = np.std(diff, ddof=1)
#     return 20*math.log10(255.0/mse)
def psnr1(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def psnr2(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def avg_psnr(target_dir, ref_dir):
    target_files = os.listdir(target_dir)
    ref_files = os.listdir(ref_dir)
    ref_files.sort(key=lambda x: (int(x.split('-')[1].split('x')[0])))
    target_files.sort(key=lambda x: int(x.split('-')[1].split('.')[0]))
    total_psnr = 0
    count = 0
    for (target_file, ref_file) in zip(target_files, ref_files):
        count = count + 1
        targrt = cv2.imread(os.path.join(target_dir, target_file))
        ref = cv2.imread(os.path.join(ref_dir, ref_file))
        total_psnr = total_psnr + psnr1(targrt, ref)

    return total_psnr / count

target_path = 'F:/derain/rain_heavy_test/norain'
ref_path = 'F:/derain/net_3_gru'
avgpsnr = avg_psnr(target_path, ref_path)
print(avgpsnr)