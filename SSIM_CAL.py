import numpy as np
from PIL import Image 
from scipy.signal import convolve2d

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def rgb2gray(rgb):
    r,g,b =rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
    gray = 0.2989*r + 0.5870*g+0.1140*b
    
    return gray


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):

    im1 = rgb2gray(im1)
    im1 = im1
    im2 = rgb2gray(im2)
    im2 = im2
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    return np.mean(np.mean(ssim_map))


#if __name__ == "__main__":
def ssim(im1,im2):
#    im1 = Image.open(r"D:\defog\imgs\初试结果/25.png")    
#    im2 = Image.open(r"D:\defog\imgs\gt/25.png")

    return compute_ssim(np.array(im1),np.array(im2))

import os, cv2
def avg_ssim(target_dir, ref_dir):
    target_files = os.listdir(target_dir)
    ref_files = os.listdir(ref_dir)
    ref_files.sort(key=lambda x: (int(x.split('-')[1].split('x')[0])))
    target_files.sort(key=lambda x: int(x.split('-')[1].split('.')[0]))
    total_ssim = 0
    count = 0
    for (target_file, ref_file) in zip(target_files, ref_files):
        count = count + 1
        targrt = Image.open(os.path.join(target_dir, target_file))
        ref = Image.open(os.path.join(ref_dir, ref_file))
        total_ssim = total_ssim + ssim(targrt, ref)

    return total_ssim / count

target_path = 'F:/derain/rain_heavy_test/norain'
ref_path = 'F:/derain/net_3_gru'
avgpssim = avg_ssim(target_path, ref_path)
print(avgpssim)