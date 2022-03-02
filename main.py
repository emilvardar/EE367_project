'''
Elin Byman and Emil Vardar

The following code is for the final project for the course EE367: Computational Imaging given at Stanford University.
The code compares 4 different denoising methods for videos. The denoised videos are saved and the PSNR for each denoised
video is given as a quantitative result for comparision. 

'''

import numpy as np
import skimage
import cv2
from skimage.restoration import denoise_nl_means
from timeit import default_timer as timer  
import matplotlib.pyplot as plt

# Global varaibles
FILE_NAME = 'tennis_sif.y4m'
SIGMA = 0.1                 # Standard deviation of noise

# For local linear algorithm using only averaging
NUM_OF_AVERAGE_FRAME = 2    # Number of frames for averaging in the local linear
                            # denoising, excluding current frame. Should be an
                            # even number?

# For local linear algorithm using gaussian weighting
NUM_FRAMES = 5  # Number of frames used in the weighted averaging
                # (Size of 1D Gaussian kernel). Should be an odd number. E.g. 5
                # means that the current frame, 2 frames before and 2 frames
                # after will be included.

SIGMA_GAUSS = 1 # Standard deviation for the Gaussian kernel. Determines the
                # weighting depending on the distance from the current frame.


# For nlm in only spatial domain
SIGMANL = 0.05
PATCH = 7
PATCH_DISTANCE = 5

# For nlm in both spatial and temporal domain
SIGMANL_ST = 0.05           # Standard deviation for the nlm algorithm
PATCH_ST = 7                # Patch size
PATCH_DISTANCE_ST = 5       # Size of neighborhood, expressed as a distance
NUM_FRAMES_ST = 3           # Neighborhood in temporal domain. If bigger than
                            # PATCH_DISTANCE_ST, then PATCH_DISTANCE_ST
                            # determines the neighborhood size in both spatial
                            # and temporal domain.

def read_video():
    '''Reads the video and returns the frames in a list.'''
    vidcap = cv2.VideoCapture(FILE_NAME)   # Read in the video
    success,image = vidcap.read()
    count = 0
    frame_holder = []
    while success:
        frame_holder.append(image.astype(float)/255)    # Work with interval [0,1]
        shape = image.shape
        success,image = vidcap.read()
        count += 1
    fps = vidcap.get(cv2.CAP_PROP_FPS)                  # The fps of the image
    return frame_holder, count, shape, fps

def video_maker(video_name, frames, shape, fps):
    '''Merges the individual frames together to obtain a video'''
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') # can also use *'MJPG'  #SORU
    height, width, channels = shape
    out = cv2.VideoWriter(video_name, fourcc, fps, (width, height), True)
    for i in range(len(frames)):
        cv2.imwrite('temp.jpg', frames[i])
        out.write(cv2.imread('temp.jpg'))
    out.release()

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
    return np.clip(noisy_frames_list,0,1) # Clip the frames so it is in the interval 0 to 1 ##TODO: Diskutera det här. Är det inte konstigt att klippa till 0,1? Tar bort en del noise?

def local_linear_denoising(noisy_img, average_frame_num):
    mean_frame_list = []
    for i in range(len(noisy_img)):
        mean_frame = np.zeros_like(noisy_img[0])

        number_of_frames_averaged = 0
        for j in range(-int(average_frame_num/2), int(average_frame_num/2)+1): # Some frames before the actual frame and some frames after the actual frame
            if (i+j >= 0) & (i+j < len(noisy_img)): #Ignore frames outside of range
             mean_frame += noisy_img[i+j]
             number_of_frames_averaged += 1

        mean_frame_list.append(mean_frame/number_of_frames_averaged) # take the average
    return mean_frame_list

def gauss(n=5,sigma=1):
    r = range(-int(n/2),int(n/2)+1)
    return [1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-float(x)**2/(2*sigma**2)) for x in r]

def local_gaussian_denoising(noisy_img, kernel_size, sigma_gauss):
    mean_weighted_frame_list = []
    gaussian = gauss(n = kernel_size, sigma = sigma_gauss)

    for i in range(len(noisy_img)):
        mean_frame = np.zeros_like(noisy_img[0])

        used_indicies_of_gaussian = []
        for j in range(-int(kernel_size/2), int(kernel_size/2)+1): # Some frames before the actual frame and some frames after the actual frame
            if (i+j >= 0) & (i+j < len(noisy_img)): # Avoid trying to take frames before first frame or after last frame
                used_indicies_of_gaussian.append(int(j + np.floor(kernel_size/2)))
                mean_frame += noisy_img[i+j] * gaussian[int(j + np.floor(kernel_size/2))]

        mean_frame = mean_frame/sum([gaussian[i] for i in used_indicies_of_gaussian]) # Normalize based on the weights used from the Gaussian 1D kernel
        mean_weighted_frame_list.append(mean_frame)

    return mean_weighted_frame_list

def nlm_only_spatial(frames):
    nlm_denoised_frames = []
    for i in range(len(frames)):
        denoised_frame = denoise_nl_means(frames[i], h=np.sqrt(2)*SIGMANL, fast_mode=True, patch_size=PATCH, patch_distance=5, multichannel=True)
        nlm_denoised_frames.append(denoised_frame)
    return nlm_denoised_frames

def nlm_spatial_time(frames):
    nlm_denoised_frames = []

    for i in range(0,len(frames)):
        denoised_frame = np.zeros((frames.shape[1:]))

        for color_channel in range(3):
            frames_current_color_channel = []
            frames_before_current_frame_count = 0
            for j in range(-int(NUM_FRAMES_ST/2), int(NUM_FRAMES_ST/2)+1):
                if (i+j >= 0) & (i+j < len(frames)):
                    if j < 0:
                        frames_before_current_frame_count += 1
                    frames_current_color_channel.append(frames[i+j,:,:,color_channel])

            frames_current_color_channel = np.array(frames_current_color_channel)

            denoised_color_channel_of_frame = \
            denoise_nl_means(frames_current_color_channel, h=np.sqrt(2)*SIGMANL_ST, fast_mode=True,patch_size=(PATCH_ST), patch_distance=PATCH_DISTANCE_ST, multichannel=False) # Denoise the current color channel for all frames previously chosen

            denoised_frame[:,:,color_channel] = \
            denoised_color_channel_of_frame[frames_before_current_frame_count,:,:] # Pick the middle frame as the currently denoised frame.

        nlm_denoised_frames.append(denoised_frame)
        print(i)
    return nlm_denoised_frames

def video_maker_func(noisy_frames, lld_frames, lgd_frames, nlm_os_frames, nlm_st_frames, shape, fps):
    'Videos for qualitative comparision'
    video_maker('noisy_video.avi', np.clip(noisy_frames,0,1)*255, shape, fps)  
    video_maker('denoised_lld.avi', np.clip(lld_frames,0,1)*255, shape, fps)
    video_maker('denoised_lld_lgd.avi', np.clip(lgd_frames,0,1)*255, shape, fps)
    video_maker('denoised_lld_nlm_only_spatial.avi', np.clip(nlm_os_frames,0,1)*255, shape, fps)
    video_maker('denoised_lld_nlm_spaial_and_time.avi', np.clip(nlm_st_frames,0,1)*255, shape, fps)

def make_PSNR_plot(psnr_list, lld_PSNR, lgd_PSNR, nlm_PSNR, nlm_st_PSNR):
    frame_number = np.arange(len(noisy_frames))
    plt.plot(frame_number,psnr_list)
    plt.plot(frame_number,lld_PSNR)
    plt.plot(frame_number,lgd_PSNR) 
    plt.plot(frame_number,nlm_PSNR)
    plt.plot(frame_number,nlm_st_PSNR)
    plt.legend(['Noisy', 'Local Linear Denoising', 'Local Gaussian Denoising', 'NLM Denosing only spatial', 'NLM Denosing spatial and temporal'])
    plt.title('PSNR')
    plt.ylim([15, 28])
    plt.savefig('PSNR plot')
    plt.show()

if __name__ == '__main__':
    start = timer()

    frames, num_frames, shape, fps = read_video()
    
    noisy_frames = noise_adder(frames, num_frames)          # Add noise 
    noisy_frames = noisy_frames[:5]                        # Shorten the video  # TODO: CHANGE THIS
    psnr_list = PSNR_calc(noisy_frames, frames)             # Calc PSNR in the noisy video
    
    lld_frames = local_linear_denoising(noisy_frames, NUM_OF_AVERAGE_FRAME)      # Denoise with only averaging the pixel values
    lld_PSNR = PSNR_calc(lld_frames, frames)
    
    lgd_frames = local_gaussian_denoising(noisy_frames, NUM_FRAMES, SIGMA_GAUSS)    # Denoise with Gaussian filter in temporal domain
    lgd_PSNR = PSNR_calc(lgd_frames, frames)

    nlm_os_frames = nlm_only_spatial(noisy_frames)          # Denoise with NLM algorithm only in spatial domain
    nlm_PSNR = PSNR_calc(nlm_os_frames, frames)    
    
    nlm_st_frames = nlm_spatial_time(noisy_frames)          # Denoise with NLM algorithm both in spatial domain and temporal domain
    nlm_st_PSNR = PSNR_calc(nlm_st_frames, frames)

    #Print quantitative values for PSNR
    print('Average PSNR before denoising: ', np.sum(psnr_list)/len(psnr_list), 
    '\nAverage PSNR after denoising with local linear denoiser: ', np.sum(lld_PSNR)/len(lld_PSNR),
    '\nAverage PSNR after denoising with local Gaussian denoiser: ', np.sum(lld_PSNR)/len(lld_PSNR),
    '\nAverage PSNR after denoising with non-local linear denoiser only in spatial domain: ', np.sum(nlm_PSNR)/len(nlm_PSNR),
    '\nAverage PSNR after denoising with non-local linear denoiser in spatial domain and time domain: ', np.sum(nlm_st_PSNR)/len(nlm_st_PSNR))  
    
    video_maker_func(noisy_frames, lld_frames, lgd_frames, nlm_os_frames, nlm_st_frames, shape, fps)
    make_PSNR_plot(psnr_list, lld_PSNR, lgd_PSNR, nlm_PSNR, nlm_st_PSNR)

    end = timer()
    print('The time it took for running is: ', (end-start)/60, ' minutes.')
    
