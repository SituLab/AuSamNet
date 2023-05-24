import os
from utils import *
from debayer import Debayer5x5, Layout
import matplotlib.pyplot as plt
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imgmode = 'grbg'
f = Debayer5x5(layout=Layout.GRBG).cuda()


def MyEval(image_dir, checkpoint, CheckpointMode,sampling_times):


    CheckpointMode_eval = CheckpointMode+'_Eval'

    Net = torch.load(checkpoint)['model'].to(device)
    Net.eval()
    model = Net
    # batch_size = 1

    image = Image.open(os.path.join(image_dir), mode='r')
    image = image.convert('RGB')
    img = convert_image(image, source='pil', target='[0, 1]')
    img_temp = torch.zeros([1, 1, 128, 128]).cuda()
    img_temp[0, 0, :, :] = img[0, :, :]
    # img_color_temp = f(img_temp).detach().cpu()
    
   
    with torch.no_grad():
        img_net_bayer = model(img_temp, CheckpointMode_eval, sampling_times)[0]
        img_net_RGB = f(img_net_bayer)
        img_net_color = img_net_RGB.detach().cpu()
        img_net_color[-1, :, :, :] = minmaxscaler(img_net_color[-1, :, :, :])
    return img_temp, img_net_bayer, img_net_color


############################ full sampling ###########################
image_dir0 = os.path.join('./Results/ExpResults/full_sampling_bayer.png')
image_dir1 = os.path.join('./Results/ExpResults/full_sampling_color.png')

image = Image.open(image_dir0, mode='r').convert('RGB')
img = convert_image(image, source='pil', target='[0, 1]')
img_temp = torch.zeros([1, 1, 128, 128]).cuda()
img_temp[0, 0, :, :] = img[0, :, :]

img_GT0 = Image.open(image_dir1).convert('RGB')
img_GT00 = convert_image(img_GT0, source='pil', target='[0, 1]')
img_GT = torch.zeros([1, 3, 128, 128]).cpu()
img_GT[-1,:,:,:] = img_GT00

########################################### Basic Parameters ###########################################
dataset = 'CelebA'
SR = 0.375/2
SR_num = math.floor(SR*128**2)
SR_num_str = str(int(SR_num))

SR_Point =  SR/1.5
SR_Point_num = math.floor(SR_Point * 128**2)

LR = '2e-4'
epochs = 50

########################################### computation ###########################################

CheckpointMode = 'M2' # M2/M3
checkpoint = './Results/%s/%s/best_checkpoint_resnetcolor_%s_%d_%s_%s_%d.pth.tar'%(CheckpointMode,SR_num,dataset,SR_num,CheckpointMode,LR,epochs)
image_dir = './Results/ExpResults/%s_%s.png'%(CheckpointMode,SR_num_str)
img_bayer2, img_net_bayer2, img_net2 = MyEval(image_dir, checkpoint, CheckpointMode, SR_Point_num)


CheckpointMode = 'M3' # M2/M3
checkpoint = './Results/%s/%s/best_checkpoint_resnetcolor_%s_%d_%s_%s_%d.pth.tar'%(CheckpointMode,SR_num,dataset,SR_num,CheckpointMode,LR,epochs)
image_dir = './Results/ExpResults/%s_%s.png'%(CheckpointMode,SR_num_str)
img_bayer3, img_net_bayer3, img_net3 = MyEval(image_dir, checkpoint, CheckpointMode, SR_Point_num)


plt.subplot(1,3,1)
plt.imshow(img_GT[-1, :, :, :].squeeze().permute(1, 2, 0))
plt.title('GT')

plt.subplot(1,3,2)
plt.imshow(img_net2[-1, :, :, :].squeeze().permute(1, 2, 0))
plt.title('M2')

plt.subplot(1,3,3)
plt.imshow(img_net3[-1, :, :, :].squeeze().permute(1, 2, 0))
plt.title('M3')

plt.show()


img = convert_image(img_net2[-1, :, :, :].squeeze().permute(1, 2, 0).numpy(), source='pil', target= '[0, 1]')
save_nme = './Results/ExpResults/M2_net_%s.png'%(SR_num_str)
# save_image(img,save_nme)

img = convert_image(img_net3[-1, :, :, :].squeeze().permute(1, 2, 0).numpy(), source='pil', target= '[0, 1]')
save_nme = './Results/ExpResults/M3_net_%s.png'%(SR_num_str)
# save_image(img,save_nme)