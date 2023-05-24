from utils import *
from dataset import Dataset_Bayer
import matplotlib.pyplot as plt
import scipy.io as io
import numpy as np
from debayer import Debayer5x5, Layout
from models import ResNetColor as Net
ModelsFiles = 'resnet'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################ mode ###########################
IF_TRAIN = 0 # 1--train；0--eval
if IF_TRAIN == 1:
    train_mask=True
else:
    train_mask=False


img_size=128
SR = 0.225/2                                # Sampling ratio mentioned in paper : 0.075/2, 0.15/2, 0.225/2, 0.3/2, 0.375/2    
SR_num = math.floor(SR * img_size**2)       
SR_Point = SR/1.5                        
SR_Point_num = math.floor(SR_Point * img_size**2) 

dataset = 'CelebA'
LR = '2e-4'
epochs = 50
CheckpointMode = 'M2' # M2/M3: The "M2" in this folder corresponds to the "FSI-DL" in the paper. The "M3" in this folder corresponds to the "Ours" in the paper.
checkpoint = './Results/%s/%d/best_checkpoint_resnetcolor_%s_%d_%s_%s_%d.pth.tar' % (CheckpointMode,SR_num,dataset, SR_num, CheckpointMode, LR, epochs)
CheckpointMode_eval = CheckpointMode+'_Eval'
imgmode = 'grbg'
f = Debayer5x5(layout=Layout.GRBG).cuda()
Net = torch.load(checkpoint)['model'].to(device)

Net.eval()
model = Net
batch_size = 1
img_w = 128
img_h = 128

############################ datapath ###########################
data_folder = "./data/"
test_data_names = ['test_'+ dataset +'_128']
mask_save_path = './tempMyResultsEval/mask_%s_%d_%s_%s_%d.mat' % (dataset, SR_num, CheckpointMode, LR, epochs)
results_save_path = './tempMyResultsEval/results_%s_%d_%s_%s_%d.mat' % (dataset, SR_num, CheckpointMode, LR, epochs)
RGB_results_save_path = './tempMyResultsEval/RGBresults_%s_%d_%s_%s_%d.mat' % (dataset, SR_num, CheckpointMode, LR, epochs)
label_save_path = './tempMyResultsEval/RGBlabel_%s_%d_%s_%s_%d.mat' % (dataset, SR_num, CheckpointMode, LR, epochs)


############################ Evaluate ###########################
for test_data_name in test_data_names:
    print("\nFor %s:\n" % test_data_name)

    # Custom dataloader  
    test_dataset = Dataset_Bayer(data_folder,
                             split='test',
                             hr_img_type='[0, 1]',
                             test_data_name=test_data_name)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                              pin_memory=True)
    
    test_results = np.zeros([img_w, img_h, len(test_loader), 3])
    RGB_test_results = np.zeros([img_w, img_h, 3, len(test_loader)])
    label_test_results = np.zeros([img_w, img_h, 3, len(test_loader)])

    
    # Prohibit gradient computation explicitly because I had some problems with memory
    with torch.no_grad():
        # Batches  
        for i, (hr_img_Bayer, hr_img_RGB) in enumerate(test_loader):
            # Move to default device
            hr_img_Bayer = torch.reshape(hr_img_Bayer, [batch_size, 1, 128, 128])
            hr_img_RGB = torch.reshape(hr_img_RGB, [batch_size, 3, 128, 128])
            hr_img_Bayer = hr_img_Bayer.to(device)  # (batch_size (1), 3, w / 4, h / 4), in [0, 1]
            hr_img_RGB = hr_img_RGB.to(device)  # (batch_size (1), 3, w, h), in [0, 1]

            # Forward prop.
            sr_img_Bayer, mask_temp, im_fft_masked_view, im_ifft_masked, mask_full = model(hr_img_Bayer, CheckpointMode_eval, SR_Point_num)
            sr_img_RGB = f(sr_img_Bayer)

            test_results[:, :, i, 0] = hr_img_Bayer[-1, 0, :, :].detach().cpu().numpy() # ground truth
            test_results[:, :, i, 1] = im_ifft_masked[-1, 0, :, :].detach().cpu().numpy() # after FFT
            test_results[:, :, i, 2] = sr_img_Bayer[-1, 0, :, :].detach().cpu().numpy() # the network’s output

            RGB_test_results[:,:,:,i] = sr_img_RGB[-1, :, :, :].detach().cpu().permute(1,2,0).numpy()
            label_test_results[:,:,:,i] = hr_img_RGB[-1, :, :, :].detach().cpu().permute(1,2,0).numpy()
            


        # Plot and save results 
        im_fft_masked_view1 = im_fft_masked_view.detach().cpu().numpy()
        mask_temp1 = mask_temp.detach().cpu().numpy()
        im_ifft_masked1 = im_ifft_masked.detach().cpu().numpy()
        samples = mask_temp1.sum()

        hr_img_Bayer1 = hr_img_Bayer.detach().cpu()
        sr_img1 = sr_img_RGB.detach().cpu()
        hr_img1 = hr_img_RGB.detach().cpu()
        mask_full1 = mask_full.detach().cpu().numpy()

        plt.subplot(231)
        plt.imshow(mask_temp1)
        plt.title('learned mask')
        plt.subplot(232)
        plt.imshow(im_fft_masked_view1)
        plt.title('masked freq.')
        plt.subplot(233)
        plt.imshow(im_ifft_masked1[-1, 0, :, :])
        plt.title('FSI downsam')
        plt.subplot(234)
        plt.imshow(hr_img_Bayer1[-1, 0, :, :])
        plt.title('RAW_GT')
        plt.subplot(2, 3, 5)
        plt.imshow(hr_img1[-1, :, :, :].squeeze().permute(1, 2, 0))
        plt.title('GT')
        plt.subplot(236)
        sr_img1[-1, :, :, :] = minmaxscaler(sr_img1[-1, :, :, :])
        plt.imshow(sr_img1[-1, :, :, :].squeeze().permute(1, 2, 0))
        plt.title('net_output')
        plt.show()

        # io.savemat(mask_save_path, {'mask_half': mask_temp1, 'mask_full': mask_full1})
        # io.savemat(results_save_path, {'results': test_results})
        # io.savemat(RGB_results_save_path, {'RGBresults': RGB_test_results})
        # io.savemat(label_save_path, {'labelresults': label_test_results})





