import numpy as np
import skimage
import cv2
import skimage.io as io
from skimage.restoration import denoise_nl_means
from timeit import default_timer as timer  
import matplotlib.pyplot as plt

'''
What to do?
1- Local denoiser with Gaussian kernel in time domain
2- NLM in time domain
3- VDB3M implement and compare 
4- Compare our methods with different parameters with them selves
'''

# Global varaibles
SIGMA = 0.1                 # Std for noise
NUM_OF_AVERAGE_FRAME = 2    # Number of frames for averaging in the local linear denoising
FRAME_NUMBER_NLM = 2        # Number of frames for averaging in the non-local mean denoising in time and spatial domain, give even number
SIGMANL = 0.05
PATCH = 7
PATCH_SPATIAL_TIME = 3      # The patch size to be used in the non-local mean denoising in time and spatial domain
SEARCH_AREA = 5             # The area we are going to search on for non-local mean denoising in time and spatial domain


def local_denoiser_in_time_domain():
    return 'weighted_denoised_frames'

def cut_out_frames(used_frames, pixel_num_ver, pixel_num_hor,ver_pixels=0, hor_pixels=0):
    temp_frame = np.zeros((SEARCH_AREA,(FRAME_NUMBER_NLM+1)*SEARCH_AREA,3))
    try:
        for i in range(len(used_frames)):
            temp_frame[:,i*SEARCH_AREA:(i+1)*SEARCH_AREA,:] = \
                used_frames[i][pixel_num_ver-int(SEARCH_AREA/2):pixel_num_ver+int(SEARCH_AREA/2)+1, pixel_num_hor-int(SEARCH_AREA/2):pixel_num_hor+int(SEARCH_AREA/2)+1,:] # Gives the specific patch in frame, first index rows, second index columns, last index the colors
        return temp_frame
    except:
        for i in range(len(used_frames)):
            r1, g1, b1 = used_frames[i][:, :, 0], used_frames[i][:, :, 1], used_frames[i][:, :, 2]
            for k in range(SEARCH_AREA):
                r1 = np.insert(r1,hor_pixels+k,np.zeros(ver_pixels),axis=1)
                g1 = np.insert(g1,hor_pixels+k,np.zeros(ver_pixels),axis=1)
                b1 = np.insert(b1,hor_pixels+k,np.zeros(ver_pixels),axis=1)
            padded_img = np.dstack((r1, g1, b1))
            temp_frame[:,i*SEARCH_AREA:(i+1)*SEARCH_AREA,:] = \
                padded_img[pixel_num_ver-int(SEARCH_AREA/2):pixel_num_ver+int(SEARCH_AREA/2)+1, pixel_num_hor-int(SEARCH_AREA/2):pixel_num_hor+int(SEARCH_AREA/2)+1,:] # Gives the specific patch in frame, first index rows, second index columns, last index the colors
        return temp_frame
    

def nlm_in_spatial_and_time_searh_area(frames, shape):
    'MÅSTE FIXA DET SVARTA PIXLARNA LÄNGST NER PÅ VIDEON OCKSÅ TROR ATT DET BLIR JÄTTEBRA EFTER DET'
    nlm_denoised_frames = []
    vertical_pixels = shape[0]
    horizontal_pixels = shape[1]
    for i in range(len(frames)):                      # Go through all the frames
        if i < int(FRAME_NUMBER_NLM/2) or i > (len(frames)-int(FRAME_NUMBER_NLM/2)-1):    # Denoise the first frames and last frames only with spatial denoiser
            denoised_frame = denoise_nl_means(frames[i], h=np.sqrt(2)*SIGMANL, fast_mode=True, patch_size=(PATCH_SPATIAL_TIME), patch_distance=SEARCH_AREA, channel_axis=True)                    # Don't deno
            
        else:
            denoised_frame = np.zeros_like(frames[i])       # Create a zero frame to be filled in 
            
            # Go through all the possible patches in the frame in focus with some patches infront and some patches after it for denoising in time domain as well 
            # Ver_pixel, hor_pixel == middle point of search area
            for ver_pixels in range(int(SEARCH_AREA/2), vertical_pixels-int(SEARCH_AREA/2)-1, SEARCH_AREA):   
                for hor_pixels in range(int(SEARCH_AREA/2),horizontal_pixels-int(SEARCH_AREA/2)-1, SEARCH_AREA):     
                    temp_frame = cut_out_frames(frames[i-int(FRAME_NUMBER_NLM/2):i+int(FRAME_NUMBER_NLM/2)+1], ver_pixels, hor_pixels)   # Send the specific frames we are going to use for NLM denoising 
                    total_denoise =\
                        denoise_nl_means(temp_frame,h=np.sqrt(2)*SIGMANL, fast_mode=True, patch_size=(PATCH_SPATIAL_TIME), patch_distance=SEARCH_AREA, channel_axis=True)
                    denoised_frame[ver_pixels-int(SEARCH_AREA/2):ver_pixels+int(SEARCH_AREA/2)+1, hor_pixels-int(SEARCH_AREA/2):hor_pixels+int(SEARCH_AREA/2)+1,:] \
                        = total_denoise[:,SEARCH_AREA:2*SEARCH_AREA,:]
                hor_pixels += SEARCH_AREA
                temp_frame = cut_out_frames(frames[i-int(FRAME_NUMBER_NLM/2):i+int(FRAME_NUMBER_NLM/2)+1], ver_pixels, hor_pixels,vertical_pixels,horizontal_pixels)
                total_denoise =\
                    denoise_nl_means(temp_frame,h=np.sqrt(2)*SIGMANL, fast_mode=True, patch_size=(PATCH_SPATIAL_TIME), patch_distance=SEARCH_AREA, channel_axis=True)
                denoised_frame[ver_pixels-int(SEARCH_AREA/2):ver_pixels+int(SEARCH_AREA/2)+1, hor_pixels-int(SEARCH_AREA/2):,:] \
                    = total_denoise[:,SEARCH_AREA:SEARCH_AREA+horizontal_pixels-hor_pixels+1,:]
        nlm_denoised_frames.append(denoised_frame)
        print(i)
    return nlm_denoised_frames


def nlm_in_spatial_and_time_WHOLE_IMAGE(frames, shape):
    nlm_denoised_frames = []
    vertical_pixels = shape[0]
    horizontal_pixels = shape[1]
    for i in range(int(FRAME_NUMBER_NLM/2),len(frames)-int(FRAME_NUMBER_NLM/2)):                      # Go through all the frames
        denoised_frame = np.zeros_like(frames[i])       # Create a zero frame to be filled in 
        
        # Go through all the possible patches in the frame in focus with some patches infront and some patches after it for denoising in time domain as well 
        # Ver_pixel, hor_pixel == middle point of search area
        for ver_pixels in range(int(SEARCH_AREA/2),vertical_pixels-int(SEARCH_AREA/2)-1,SEARCH_AREA):   
            for hor_pixels in range(int(SEARCH_AREA/2),horizontal_pixels-int(SEARCH_AREA/2)-1,SEARCH_AREA):     
                temp_frame = cut_out_frames(frames[i-int(FRAME_NUMBER_NLM/2):i+int(FRAME_NUMBER_NLM/2)+1], ver_pixels, hor_pixels)   # Send the specific frames we are going to use for NLM denoising 
                total_denoise =\
                    denoise_nl_means(temp_frame,h=np.sqrt(2)*SIGMANL, fast_mode=True, patch_size=(PATCH_SPATIAL_TIME), patch_distance=2*SEARCH_AREA, multichannel=True)
                denoised_frame[ver_pixels-int(SEARCH_AREA/2):ver_pixels+int(SEARCH_AREA/2)+1, hor_pixels-int(SEARCH_AREA/2):hor_pixels+int(SEARCH_AREA/2)+1,:] \
                    = total_denoise[:,SEARCH_AREA:2*SEARCH_AREA,:]
        
        nlm_denoised_frames.append(denoised_frame)
        print(i)
    return nlm_denoised_frames

'''
FOLLOWING TAKES TO LONG TIME 
def nlm_in_spatial_and_time_per_pixel(frames, shape):
    nlm_denoised_frames = []
    vertical_pixels = shape[0]
    horizontal_pixels = shape[1]
    for i in range(int(FRAME_NUMBER_NLM/2),len(frames)-int(FRAME_NUMBER_NLM/2)):                      # Go through all the frames
        denoised_frame = np.zeros_like(frames[i])       # Create a zero frame to be filled in 
        
        # Go through all the possible patches in the frame in focus with some patches infront and some patches after it for denoising in time domain as well 
        # Ver_pixel, hor_pixel == middle point of search area
        for ver_pixels in range(int(SEARCH_AREA/2),vertical_pixels-int(SEARCH_AREA/2)-1):   
            for hor_pixels in range(int(SEARCH_AREA/2),horizontal_pixels-int(SEARCH_AREA/2)-1):     
                temp_frame = cut_out_frames(frames[i-int(FRAME_NUMBER_NLM/2):i+int(FRAME_NUMBER_NLM/2)+1], ver_pixels, hor_pixels)   # Send the specific frames we are going to use for NLM denoising 
                total_denoise =\
                    denoise_nl_means(temp_frame,h=np.sqrt(2)*SIGMANL, fast_mode=True, patch_size=(PATCH_SPATIAL_TIME), patch_distance=2*SEARCH_AREA, multichannel=True)
                denoised_frame[ver_pixels, hor_pixels,:] = total_denoise[int(SEARCH_AREA/2),int(3/2*SEARCH_AREA),:]
        
        nlm_denoised_frames.append(denoised_frame)
        print(i)
    return nlm_denoised_frames
'''

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
            mean_frame = noisy_img[i]
            
        elif len(noisy_img)-i <= average_frame_num/2:    # If we are at the end of the list we cannot add the 3 later comming frames. Therefore, add the closest 6 frames
            mean_frame = noisy_img[i]
            
        else:    
            for j in range(-int(average_frame_num/2), int(average_frame_num/2)+1): # Some frames before the actual frame and some frames after the actual frame 
                if j != 0:
                    mean_frame += noisy_img[i+j]        
        mean_frame_list.append(mean_frame/average_frame_num) # take the average
    return mean_frame_list

def nlm_denosing_only_spatial(frames):
    nlm_denoised_frames = []
    for i in range(len(frames)):
        denoised_frame = denoise_nl_means(frames[i], h=np.sqrt(2)*SIGMANL, fast_mode=True, patch_size=(PATCH_SPATIAL_TIME), patch_distance=5, channel_axis=True)
        nlm_denoised_frames.append(denoised_frame)
    return nlm_denoised_frames

def make_PSNR_plot(psnr_list, lld_PSNR, nlm_PSNR, nlm_PSNR_better):
    frame_number = np.arange(len(noisy_frames))
    plt.plot(frame_number,psnr_list)
    plt.plot(frame_number,lld_PSNR)
    plt.plot(frame_number,nlm_PSNR)
    plt.plot(frame_number,nlm_PSNR_better)
    plt.legend(['Noisy PSNR', 'Local Linear Denoising', 'NLM Denoising only spatial', 'NLM Denosing spatial and temporal'])
    plt.ylim([18, 25])
    plt.show()

def video_maker(video_name, frames, shape, fps):
    '''Merges the individual frames together to obtain a video'''
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
    height, width, channels = shape
    out = cv2.VideoWriter(video_name, fourcc, fps, (width, height), True)
    for i in range(len(frames)):
        cv2.imwrite('temp.jpg', frames[i])
        out.write(cv2.imread('temp.jpg'))
    out.release()

def video_maker_func(noisy_frames,lld_frames,nlm_os_frames,nlm_tands_frames_denoised,shape,fps):
    'Videos for qualitative comparision'
    video_maker('noisy_video.avi', np.clip(noisy_frames,0,1)*255, shape, fps)  
    video_maker('averaged_denoised_video_lld.avi', np.clip(lld_frames,0,1)*255, shape, fps)
    video_maker('averaged_denoised_video_nlm.avi', np.clip(nlm_os_frames,0,1)*255, shape, fps)
    video_maker('averaged_denoised_video_nlm_spaial_and_time.avi', np.clip(nlm_tands_frames_denoised,0,1)*255, shape, fps)

if __name__ == '__main__':
    start = timer()
    frames, num_frames, shape, fps = read_video()
    
    noisy_frames = noise_adder(frames, num_frames)
    noisy_frames = noisy_frames[0:15]                                            # Shorten the video so it does not take so long time
    psnr_list = PSNR_calc(noisy_frames, frames)                                  # Gives the PSNR for the noisy images in the first case
    
    lld_frames = local_linear_denoising(noisy_frames, NUM_OF_AVERAGE_FRAME)      # Find the mean value of each of the pixels
    lld_PSNR = PSNR_calc(lld_frames, frames)
    
    nlm_os_frames = nlm_denosing_only_spatial(noisy_frames)
    nlm_PSNR = PSNR_calc(nlm_os_frames, frames)    
    
    nlm_tands_frames_denoised = nlm_in_spatial_and_time_searh_area(noisy_frames, shape)
    nlm_PSNR_better = PSNR_calc(nlm_tands_frames_denoised, frames) 
    
    '''
    nlm_tands_frames_denoised_per_pixel = nlm_in_spatial_and_time_per_pixel(noisy_frames, shape)
    nlm_PSNR_better_pixel = PSNR_calc(nlm_tands_frames_denoised_per_pixel, frames[int(FRAME_NUMBER_NLM/2):]) 
    '''

    'Print quantitative values for PSNR'
    print('Average PSNR before denoising: ', np.sum(psnr_list)/len(psnr_list), 
    '\nAverage PSNR after denoising with local linear denoiser: ', np.sum(lld_PSNR)/len(lld_PSNR),
    '\nAverage PSNR after denoising with non-local linear denoiser only in spatial domain: ', np.sum(nlm_PSNR)/len(nlm_PSNR),
    '\nAverage PSNR after denoising with non-local linear denoiser in spatial domain and time domain: ', np.sum(nlm_PSNR_better)/len(nlm_PSNR_better))  
    
    video_maker_func(noisy_frames,lld_frames,nlm_os_frames,nlm_tands_frames_denoised,shape,fps)
    make_PSNR_plot(psnr_list, lld_PSNR, nlm_PSNR, nlm_PSNR_better)

    end = timer()
    print('The time it took for running is: ', (end-start)/60, ' minutes.')
    
