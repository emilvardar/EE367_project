import numpy as np
import skimage.io as io
import skimage
import cv2

# Global varaibles
SIGMA = 0.1             # Std

def read_video():
    vidcap = cv2.VideoCapture('tennis_sif.y4m')   # Read in the video
    success,image = vidcap.read()
    count = 0
    frame_holder = []
    while success:
        cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
        frame_holder.append(image.astype(float)/255)
        success,image = vidcap.read()
        count += 1
    return frame_holder, count

def read_image():
    gt_img = io.imread('birds_gray.png').astype(float)/255 # Read in the gray image
    H = np.shape(gt_img)[0]    # The number of rows
    W = np.shape(gt_img)[1]    # The number of columns
    return gt_img, H, W

def PSNR_calc(noisy_frame, gt_frame):
    '''Takes the noisy image as input and calculates the PSNR w.r.t. the GT image. 
    Returns a list containing the PSNR of each noisy image w.r.t. GT image'''
    psnr_list = []
    for i in range(len(noisy_frame)):
        psnr = skimage.metrics.peak_signal_noise_ratio(noisy_frame[i],gt_frame[i])  # Calc. the PSNR
        psnr_list.append(psnr)
    return psnr_list

def noise_adder(gt_frames, num_frames):
    '''Takes the GT images and adds random Gaussian noise to them with std. sigma.
    Returns a list containing the noisy images'''
    noisy_frames_list = []          # List that saves the noisy frames
    for i in range(num_frames):
        noisy_frame = gt_frames[i] + SIGMA * np.random.randn(*gt_frames[i].shape) # Add Gaussian noise with std sigma to the frame in focus
        noisy_frames_list.append(noisy_frame)
    return np.clip(noisy_frames_list,0,1) # Clip the frames so it is in the interval 0 to 1

def mean_finder(noisy_img):
    mean_img = np.zeros_like(noisy_img[0])
    for i in range(len(noisy_img)):
        mean_img += noisy_img[i]
    return mean_img/len(noisy_img)

if __name__ == '__main__':
    frames, num_frames = read_video()
    noisy_frames = noise_adder(frames, num_frames)
    psnr_list = PSNR_calc(noisy_frames, frames)     # Gives the PSNR for the noisy images in the first case
    print(psnr_list)
    #mean_img = mean_finder(noisy_frames)            # Find the mean value of each of the pixels
    #mean_PSNR = PSNR_calc(mean_img, frames)
    #print(mean_PSNR)
    cv2.imwrite("nosiy_imgage.png", np.clip(noisy_frames[46],0,1)*255)

