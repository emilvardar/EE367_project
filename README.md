

Elin Byman and Emil Vardar

The code in 'main.py' is for the final project for the course EE367: Computational Imaging given at Stanford University.
The code compares 4 different denoising methods that we have created for videos. These 4 diffrent denoising methods are:
Local Averaging Denoiser, Local Gaussian Denoiser, Non-local Means Denoiser in the spatial domain, and finally Non-local Means
Denoiser in the spatial and temporal domain. For more information about these, see the paper 'Video Denoising with Local Linear 
Denoising and Non-Local Means' by Elin Byman and Emil Vardar. 


The code in 'main.py' basically reads in a couple of videos, separates these videos into frames, and adds noise to 
each of these frames separately. Afterward, the different denoising methods are applied to each frame. Finally, the 
resulting denoised videos are set together (for qualitative comparison), and the mean-PSNR for the resulting video is 
calculated (for quantitative comparison). 


To reconstruct the results given in this paper it is enough to run the code 'main.py' and the MATLAB code given in:
'M. Maggioni and A. Foi, V-BM4D software for video denoising, ver.1.0, Tampere University of Technology, 2014. 
[Online]. Available: https://webpages.tuni.fi/foi/GCF-BM3D' (To run this code follow the instructions on this page). 
Observe that the MATLAB code given in this page is not written by us. 


Be sure that the videos are correctly named as stated in the file 'main.py' and that they are in the same directory as
the code 'main.py'


The authors of the code and the paper can be contacted via the university e-mail: eevardar/byman@stanford.edu.

