import os
import time
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch
from torch import nn
from dataset import Dataset_Bayer as Dataset
from utils import *
import numpy as np
from debayer import Debayer5x5, Layout
from models import ResNetColor as Net


############################ System setting ###########################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  
grad_clip = None                             # clip if gradients are exploding
workers = 4                                  # Number of worker threads

############################ Network structure ###########################
large_kernel_size = 9  # The first and last convolution kernel size
small_kernel_size = 3  # Middle layer convolution kernel size
n_channels = 64        # Number of interlayer channels
n_blocks = 16          # Number of residual modules
train_mask=True

############################ Sampling ratio and reconstruction method ###########################
img_size = 128                              # Image size
SR = 0.225/2                                # Sampling ratio mentioned in paper : 0.075/2, 0.15/2, 0.225/2, 0.3/2, 0.375/2    
SR_num = math.floor(SR * img_size**2)       # 
SR_Point = SR/1.5                        
SR_Point_num = math.floor(SR_Point * img_size**2) 
nStepPS = 3                                  # Three-step phase shift method
imgmode = 'grbg'                             # Bayer CFA
f = Debayer5x5(layout=Layout.GRBG).cuda()    # Debayer function

############################ Model parameters ###########################
mode = 'M3'                                  # M2/M3:The "M2" in this folder corresponds to the "FSI-DL" in the paper. The "M3" in this folder corresponds to the "Ours" in the paper.

lr = 2e-4                                    # learning rate
LR = '2e-4'
epochs = 50                                  # epoch
batch_size = 8                               # batch size
start_epoch = 0                              # start at this epoch
print_freq = 5000                            # Frequency of printing the training results

############################ Storage path ###########################
data_folder = './data/'                      # json 
dataset = 'CelebA'                           # Dataset Setup
train_data_names = 'train_CelebA_128'        # Training set



model_save_path = './temp/checkpoint_resnetcolor_%s_%d_%s_%s_%d.pth.tar' % (dataset, SR_num, mode, LR, epochs)
bestmodel_save_path =  './temp/best_checkpoint_resnetcolor_%s_%d_%s_%s_%d.pth.tar' % (dataset, SR_num, mode, LR, epochs)
checkpoint = None                           # Pre-trained model paths

def main():
    ############################ training ###########################
    global start_epoch, epoch, checkpoint 
    best_loss = 1

    ############################ Loading the pre-trained model ###########################
    if checkpoint is None:
       
        model = Net(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
                         n_channels=n_channels, n_blocks=n_blocks, train_mask=train_mask,
                         mode=mode, SR_Point=SR_Point)
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
    
    ############################ Move to default device ###########################
    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    
    ############################ dataloaders ###########################
    train_dataset = Dataset(data_folder,
                              split='train',
                              hr_img_type='[0, 1]',
                              train_data_name=train_data_names)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=True, drop_last=True)  

    
    ############################# Start training ############################
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        # if IF_TRAIN != 1:
        #     if mode == 'M3':
        #         optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              imgmode=imgmode)

        # Save the current trained model
        torch.save({'epoch': epoch,
                    'model': model,
                    'optimizer': optimizer},
                    model_save_path)

        model_save_path1 = './temp/checkpoint_resnetcolor_%s_%d_%s_%s_%d.pth.tar' % (dataset, SR_num, mode, LR, epoch)            
        torch.save({'epoch': epoch,
                    'model': model,
                    'optimizer': optimizer},
                    model_save_path1)

        # Save the optimal model
        if loss0 < best_loss:
            best_loss = loss0
            torch.save({'epoch': epoch,
                        'model': model,
                        'optimizer': optimizer},
                       bestmodel_save_path)


def train(train_loader, model, criterion, optimizer, epoch, imgmode):
    ############################ One epoch's training. ############################
    global loss0
    model.train()                  
    batch_time = AverageMeter()    
    data_time = AverageMeter()    
    losses_pred = AverageMeter()  
    losses_mask = AverageMeter()   
    start = time.time()

    ############################ Batches ###########################
    for i, (hr_img_Bayer, hr_img_RGB) in enumerate(train_loader):
        data_time.update(time.time() - start)
       
        hr_img_Bayer = torch.reshape(hr_img_Bayer, [batch_size, 1, img_size, img_size])
        hr_img_RGB = torch.reshape(hr_img_RGB, [batch_size, 3, img_size, img_size])
        hr_img_Bayer = hr_img_Bayer.to(device)    
        hr_img_RGB = hr_img_RGB.to(device)    
        ############################ Forward propagation ###########################
        sr_img_Bayer, mask_temp, im_fft_masked_view, im_ifft_masked, mask_full = model(hr_img_Bayer, mode, SR_Point_num)  # (N, 3, 96, 96), in [-1, 1]
        sr_img_RGB = f(sr_img_Bayer)
        # Calculating image loss
        loss_pred = criterion(sr_img_RGB, hr_img_RGB)
        lam_mask = 1 if mode == 'M3' else 0  
        # Calculate the mask loss
        loss_specmask = ((torch.sum(mask_temp)-SR_Point_num) / (img_size ** 2)) ** 2  
        # Calculate the total loss
        loss = loss_pred + lam_mask * loss_specmask
        loss0 = loss.cpu()
        loss0 = loss0.detach().numpy()
        ############################ Back propagation ###########################
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)
        ############################ Updating models ###########################
        optimizer.step()
        ############################ Recording the loss ##########################
        losses_pred.update(loss_pred.item(), hr_img_Bayer.size(0))
        losses_mask.update(loss_specmask.item(), hr_img_Bayer.size(0))
        ############################ Record time ###########################
        batch_time.update(time.time() - start)
        ############################ Reset time ###########################
        start = time.time()
        ############################ Printing the current state ###########################
        temp = mask_temp.detach().cpu().numpy()
        real_sampling_times = np.sum(temp)*1.5
        if i % print_freq == 0:  # print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]----'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----'
                  'Loss_pred {loss_p.val:.4f} ({loss_p.avg:.4f})----'
                  'Loss_mask {loss_m.val:.4f} ({loss_m.avg:.4f})----'
                  'Sampling times {sampling_times}'.format(epoch, i, len(train_loader),
                                                         batch_time=batch_time,
                                                         data_time=data_time,
                                                         loss_p=losses_pred,
                                                         loss_m=losses_mask,
                                                         sampling_times=real_sampling_times))
            ############################ Plotting the results ###########################
            mask_temp1 = mask_temp.detach().cpu().numpy()
            # mask_full1 = mask_full.detach().cpu().numpy()
            im_fft_masked_view1 = im_fft_masked_view.detach().cpu().numpy()
            im_ifft_masked1 = im_ifft_masked.detach().cpu().numpy()
            hr_img_Bayer1 = hr_img_Bayer.detach().cpu()
            sr_img1 = sr_img_RGB.detach().cpu()
            hr_img1 = hr_img_RGB.detach().cpu()
            plt.subplot(2,3,1)
            plt.imshow(mask_temp1)
            plt.title('learned mask')
            plt.subplot(2,3,2)
            # plt.imshow(im_fft_masked_view1[-1,0,:,:])
            plt.imshow(im_fft_masked_view1)
            plt.title('masked freq')
            plt.subplot(2,3,3)
            plt.imshow(im_ifft_masked1[-1, 0, :, :])
            plt.title('FSI downsam')
            plt.subplot(2,3,4)
            plt.imshow(hr_img_Bayer1[-1, :, :, :].permute(1,2,0))
            plt.title('RAW_GT')
            plt.subplot(2,3,5)
            plt.imshow(hr_img1[-1, :, :, :].squeeze().permute(1,2,0))
            plt.title('GT')
            plt.subplot(2, 3, 6)
            sr_img1[-1, :, :, :] = minmaxscaler(sr_img1[-1, :, :, :])
            plt.imshow(sr_img1[-1, :, :, :].squeeze().permute(1,2,0))
            plt.title('net_output')
            plt.suptitle('Epoch:%d' % epoch)
            plt.show()
            ############################ Save mask ###########################
            if mode == 'M3':
                # if real_sampling_times <= sampling_times:
                np.save('./temp/mask_%s_%d_%s_%s_%d' % (dataset, SR_num, mode, LR, epoch), mask_temp1)

    del hr_img_Bayer, hr_img_RGB, sr_img_Bayer  

if __name__ == '__main__':
    main()


