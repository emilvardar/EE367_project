import numpy as np
import skimage
import cv2
from skimage.restoration import denoise_nl_means

# Global varaibles
SIGMA = 0.1                 # Std
NUM_OF_AVERAGE_FRAME = 2    # Number of frames for averaging in the local linear denoising
SIGMANL = 0.05
PATCH = 7


def read_video():
    '''Reads the video and returns the frames in a list.'''
    vidcap = cv2.VideoCapture('tennis_sif.y4m')   # Read in the video
    success,image = vidcap.read()
    count = 0
    frame_holder = []
    while success:
        #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
        frame_holder.append(image.astype(float)/255)    # Work with interval [0,1]
        shape = image.shape
        success,image = vidcap.read()
        count += 1
    fps = vidcap.get(cv2.CAP_PROP_FPS)                  # The fps of the image
    return frame_holder, count, shape, fps

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

def local_linear_denoising(noisy_img, average_frame_num):
    'PROBLEMATIC BECAUSE THE EDGES CAUSES PROBLEM FIX THAT LATER ON'
    mean_frame_list = []
    for i in range(len(noisy_img)):
        mean_frame = np.zeros_like(noisy_img[0])
        if i < average_frame_num/2:   #If we are at the beginning of the list we cannot add the 3 previous frames. Therefore, add the closest 6 frames
            'Dont know if this even is important'
            continue 
        elif len(noisy_img)-i <= average_frame_num/2:    # If we are at the end of the list we cannot add the 3 later comming frames. Therefore, add the closest 6 frames
            'Dont know if this even is important'
            continue 
        else:    
            for j in range(-int(average_frame_num/2), int(average_frame_num/2)+1): # Some frames before the actual frame and some frames after the actual frame 
                if j != 0:
                    mean_frame += noisy_img[i+j]        
        mean_frame_list.append(mean_frame/average_frame_num) # take the average
    return mean_frame_list

def nlm_denosing_only_spatial(frames):
    nlm_denoised_frames = []
    for i in range(len(frames)):
        denoised_frame = denoise_nl_means(frames[i],h=np.sqrt(2)*SIGMANL, fast_mode=True,patch_size=(PATCH), patch_distance=5, multichannel=True)
        nlm_denoised_frames.append(denoised_frame)
    return nlm_denoised_frames

def video_maker(video_name, frames, shape, fps):
    '''Merges the individual frames together to obtain a video'''
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
    height, width, channels = shape
    out = cv2.VideoWriter(video_name, fourcc, fps, (width, height), True)
    for i in range(len(frames)):
        cv2.imwrite('temp.jpg', frames[i])
        out.write(cv2.imread('temp.jpg'))
    out.release()


if __name__ == '__main__':
    frames, num_frames, shape, fps = read_video()
    noisy_frames = noise_adder(frames, num_frames)
    psnr_list = PSNR_calc(noisy_frames, frames)                                 # Gives the PSNR for the noisy images in the first case
    
    lld_frames = local_linear_denoising(noisy_frames, NUM_OF_AVERAGE_FRAME)      # Find the mean value of each of the pixels
    lld_PSNR = PSNR_calc(lld_frames, frames)
    
    nlm_os_frames = nlm_denosing_only_spatial(noisy_frames)
    nlm_PSNR = PSNR_calc(nlm_os_frames, frames)    
    
    'Print quantitative values for PSNR'
    print('Average PSNR before denoising: ', np.sum(psnr_list)/len(psnr_list), 
    '\nAverage PSNR after denoising with local linear denoiser: ', np.sum(lld_PSNR)/len(psnr_list),
    '\nAverage PSNR after denoising with non-local linear denoiser only in spatial domain: ', np.sum(nlm_PSNR)/len(psnr_list))  

    'Videos for qualitative comparision'
    video_maker('noisy_video.avi', np.clip(noisy_frames,0,1)*255, shape, fps)  
    video_maker('averaged_denoised_video_lld.avi', np.clip(lld_frames,0,1)*255, shape, fps)
    video_maker('averaged_denoised_video_nlm.avi', np.clip(nlm_os_frames,0,1)*255, shape, fps)
