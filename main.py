import numpy as np
import skimage
import cv2

# Global varaibles
SIGMA = 0.1                 # Std
NUM_OF_AVERAGE_FRAME = 2    # Number of frames for averaging the easy method

def read_video():
    vidcap = cv2.VideoCapture('tennis_sif.y4m')   # Read in the video
    success,image = vidcap.read()
    count = 0
    frame_holder = []
    while success:
        #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
        frame_holder.append(image.astype(float)/255)
        shape = image.shape
        success,image = vidcap.read()
        count += 1
    return frame_holder, count, shape

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

def mean_finder(noisy_img, average_frame_num):
    'PROBLEMATIC BECAUSE THE EDGES CAUSES PROBLEM FIX THAT LATER ON'
    mean_frame_list = []
    for i in range(len(noisy_img)):
        mean_frame = np.zeros_like(noisy_img[0])
        if i < average_frame_num/2:   #If we are at the beginning of the list we cannot add the 3 previous frames. Therefore, add the closest 6 frames
            #mean_frame += np.add(noisy_img[0:i]) + np.add(noisy_img[i+1:7]) 
            continue
        elif len(noisy_img)-i <= average_frame_num/2:    # If we are at the end of the list we cannot add the 3 later comming frames. Therefore, add the closest 6 frames
            #mean_frame = np.sum(noisy_img[len(noisy_img)-average_frame_num:i]) + np.sum(noisy_img[i+1:-1]) 
            continue
        else:    
            for j in range(-int(average_frame_num/2), int(average_frame_num/2)+1): # Some frames before the actual frame and some frames after the actual frame 
                if j != 0:
                    mean_frame += noisy_img[i+j]        
        mean_frame_list.append(mean_frame/average_frame_num) # take the average
    return mean_frame_list

def video_maker(video_name, frames, shape):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
    height, width, channels = shape
    out = cv2.VideoWriter(video_name, fourcc, 30, (width, height), True)
    for i in range(len(frames)):
        cv2.imwrite('temp.jpg', frames[i])
        out.write(cv2.imread('temp.jpg'))
    out.release()

if __name__ == '__main__':
    frames, num_frames, shape = read_video()
    noisy_frames = noise_adder(frames, num_frames)
    psnr_list = PSNR_calc(noisy_frames, frames)                                 # Gives the PSNR for the noisy images in the first case
    video_maker('noisy_video.avi', np.clip(noisy_frames,0,1)*255, shape)        # Create the noisy video  
    mean_averaged_frames = mean_finder(noisy_frames, NUM_OF_AVERAGE_FRAME)      # Find the mean value of each of the pixels
    video_maker('averaged_denoised_video.avi', np.clip(mean_averaged_frames,0,1)*255, shape)
    mean_PSNR = PSNR_calc(mean_averaged_frames, frames)
    print('Average PSNR before denoising: ', np.sum(psnr_list)/len(psnr_list), 'Average PSNR after denoising: ', np.sum(mean_PSNR)/len(psnr_list))
