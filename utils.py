
import numpy as np
from PIL import Image
import os
import json
import torchvision.transforms.functional as FT
import torch
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)

def convert_image(img, source, target):
    """
    Convert an image from a source format to a target format.

    :param img: image
    :param source: source format, one of 'pil' (PIL image), '[0, 1]' or '[-1, 1]' (pixel value ranges)
    :param target: target format, one of 'pil' (PIL image), '[0, 255]', '[0, 1]', '[-1, 1]' (pixel value ranges),
                   'imagenet-norm' (pixel values standardized by imagenet mean and std.),
                   'y-channel' (luminance channel Y in the YCbCr color format, used to calculate PSNR and SSIM)
    :return: converted image
    """
    assert source in {'pil', '[0, 1]', '[-1, 1]'}, "Cannot convert from source format %s!" % source
    assert target in {'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm',
                      'y-channel'}, "Cannot convert to target format %s!" % target

    # Convert from source to [0, 1]
    if source == 'pil':
        img = FT.to_tensor(img)

    elif source == '[0, 1]':
        pass  # already in [0, 1]

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    # Convert from [0, 1] to target
    if target == 'pil':
        img = FT.to_pil_image(img)

    elif target == '[0, 255]':
        img = 255. * img

    elif target == '[0, 1]':
        pass  # already in [0, 1]

    elif target == '[-1, 1]':
        img = 2. * img - 1.

    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda

    elif target == 'y-channel':
        # Based on definitions at https://github.com/xinntao/BasicSR/wiki/Color-conversion-in-SR
        # torch.dot() does not work the same way as numpy.dot()
        # So, use torch.matmul() to find the dot product between the last dimension of an 4-D tensor and a 1-D tensor
        img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights) / 255. + 16.

    return img

class ImageTransforms(object):

    def __init__(self, split, hr_img_type):
        self.split = split.lower()
        self.hr_img_type = hr_img_type
        assert self.split in {'train', 'test', 'verify'}

    def __call__(self, img):
        hr_img = convert_image(img, source='pil', target=self.hr_img_type)
    
        return hr_img

def RGB2Bayer(RGBimg, mode):
    # RGBimg 为torch格式，channel*mRow*nCol

    (Channel, mRow, nCol) = RGBimg.size()
    if mode == 'grbg':
        R = torch.Tensor([[0, 1], [0, 0]])
        G = torch.Tensor([[1, 0], [0, 1]])
        B = torch.Tensor([[0, 0], [1, 0]])
        ColorMap = torch.zeros((Channel, mRow, nCol)).to(RGBimg.device)
        ColorMap[0,:,:] = torch.tile(R,(int(mRow/2),int(nCol/2)))
        ColorMap[1,:,:] = torch.tile(G,(int(mRow/2),int(nCol/2)))
        ColorMap[2,:,:] = torch.tile(B,(int(mRow/2),int(nCol/2)))
    elif mode == 'gbrg':
        B = torch.Tensor([[0, 1], [0, 0]])
        G = torch.Tensor([[1, 0], [0, 1]])
        R = torch.Tensor([[0, 0], [1, 0]])
        ColorMap = torch.zeros((Channel, mRow, nCol)).to(RGBimg.device)
        ColorMap[0,:,:] = torch.tile(R,(int(mRow/2),int(nCol/2)))
        ColorMap[1,:,:] = torch.tile(G,(int(mRow/2),int(nCol/2)))
        ColorMap[2,:,:] = torch.tile(B,(int(mRow/2),int(nCol/2)))
    elif mode == 'rggb':
        R = torch.Tensor([[1, 0], [0, 0]])
        G = torch.Tensor([[0, 1], [1, 0]])
        B = torch.Tensor([[0, 0], [0, 1]])
        ColorMap = torch.zeros((Channel, mRow, nCol)).to(RGBimg.device)
        ColorMap[0,:,:] = torch.tile(R,(int(mRow/2),int(nCol/2)))
        ColorMap[1,:,:] = torch.tile(G,(int(mRow/2),int(nCol/2)))
        ColorMap[2,:,:] = torch.tile(B,(int(mRow/2),int(nCol/2)))
    elif mode == 'bggr':
        B = torch.Tensor([[1, 0], [0, 0]])
        G = torch.Tensor([[0, 1], [1, 0]])
        R = torch.Tensor([[0, 0], [0, 1]])
        ColorMap = torch.zeros((Channel, mRow, nCol)).to(RGBimg.device)
        ColorMap[0,:,:] = torch.tile(R,(int(mRow/2),int(nCol/2)))
        ColorMap[1,:,:] = torch.tile(G,(int(mRow/2),int(nCol/2)))
        ColorMap[2,:,:] = torch.tile(B,(int(mRow/2),int(nCol/2)))
    else:
        print("unsupport bayer format:", mode)

    ColorImg = torch.zeros((Channel, mRow, nCol)).to(RGBimg.device)
    Bayerimg = torch.zeros((mRow, nCol)).to(RGBimg.device)
    for i in range(RGBimg.size(0)):
        ColorImg[i, :, :] = torch.mul(RGBimg[i, :, :], ColorMap[i, :, :])
        Bayerimg = Bayerimg + ColorImg[i, :, :]

    return Bayerimg

def GenerateCircleSpecMask(mRow,nCol,SR_Point):
    HalfSpecMask = getHalfSpecMask(mRow, nCol)

    if mRow > nCol:
        nPixel = mRow
    else:
        nPixel = nCol

    # For squares with even side lengths
    CenterX = mRow / 2
    CenterY = nCol / 2
    [Centerx, Centery] = np.meshgrid(np.arange(-CenterX,CenterX,1),np.arange(-CenterY,CenterY,1))
    CenterCircle = Centerx ** 2 + Centery ** 2

    # Top Center
    Center2TopX = 1
    Center2TopY = CenterY
    [Center2Topx, Center2Topy] = np.meshgrid(np.arange(Center2TopX,mRow+1,1), np.arange(-CenterY,CenterY,1))
    Center2TopCircle = Center2Topx ** 2 + Center2Topy ** 2

    # Left&Right Center
    Center2LeftX = CenterX
    Center2LeftY = 1
    [Center2Leftx, Center2Lefty] = np.meshgrid(np.arange(-CenterX,CenterX,1), np.arange(Center2LeftY,nCol+1,1))
    Center2LeftCircle = Center2Leftx ** 2 + Center2Lefty ** 2

    Center2RightX = CenterX
    Center2RightY = nCol
    [Center2Rightx, Center2Righty] = np.meshgrid(np.arange(mRow,0,-1), np.arange(-CenterY,CenterY,1))
    Center2RightCircle = Center2Rightx ** 2 + Center2Righty ** 2

    # Diag Center
    Center2DiagLeftX = 1
    Center2DiagLeftY = 1
    [Center2DiagLeftx, Center2DiagLefty] = np.meshgrid(np.arange(1,mRow+1,1), np.arange(1, nCol+1,1))
    Center2DiagLeftCircle = Center2DiagLeftx ** 2 + Center2DiagLefty ** 2

    Center2DiagRightX = 1
    Center2DiagRightY = nCol
    [Center2DiagRightx, Center2DiagRighty] = np.meshgrid(np.arange(mRow,0,-1), np.arange(1, nCol+1,1))
    Center2DiagRightCircle = Center2DiagRightx ** 2 + Center2DiagRighty ** 2


    # determine the radius of the circle according to the spectrum sampling ratio
    SamplingDot = math.floor(SR_Point * mRow * nCol)

    for radius in range(0, nPixel*100 + 1, 1):
    # for radius in range(0.0,nPixel+0.01,0.01):
        radius0 = radius/100.0
        MaskTemp = np.zeros((mRow, nCol))
        # Distance2Center
        MaskTemp[find(CenterCircle <= radius0 * radius0)] = 1
        # Distance2Top
        MaskTemp[find(Center2TopCircle <= radius0 * radius0)] = 1
        # Distance2Left
        MaskTemp[find(Center2LeftCircle <= radius0 * radius0)] = 1
        # Distance2Right
        MaskTemp[find(Center2RightCircle <= radius0 * radius0)] = 1
        # Distance2DiagLeft
        MaskTemp[find(Center2DiagLeftCircle <= radius0 * radius0)] = 1
        # Distance2DiagRight
        MaskTemp[find(Center2DiagRightCircle <= radius0 * radius0)] = 1

        MaskTemp = MaskTemp * HalfSpecMask
        if MaskTemp.sum() >= SamplingDot:
            #print(radius0)
            break

    return MaskTemp

def getHalfSpecMask(mRow, nCol):
    Mask = np.zeros([mRow, nCol])
    if np.mod(mRow, 2) == 0 and np.mod(nCol, 2) == 0:
        smallerMask = getHalfSpecMask(mRow - 1, nCol - 1)
        Mask[1::, 1::] = smallerMask
        Mask[1:int(mRow / 2)+1, 0] = 1
        Mask[0, 0] = 1
        Mask[0, 1:int(nCol / 2)+1] = 1
    else:
        if np.mod(mRow, 2) == 1 and np.mod(nCol, 2) == 1:
            Mask[0:int(np.ceil(mRow / 2)), :] = 1
            Mask[int(np.ceil(mRow / 2)-1), int(np.ceil(nCol / 2))::] = 0
        else:
            if np.mod(mRow, 2) == 1 and np.mod(nCol, 2) == 0:
                Mask[0:int(np.ceil(mRow / 2)), 1::] = 1
                Mask[int(np.ceil(mRow / 2)-1), int(np.ceil(nCol / 2) + 1)::] = 0
                Mask[0:int(np.ceil(mRow / 2)), 0] = 1
            else:
                Mask[0:int(mRow/2)+1, :] = 1
                Mask[int(mRow/2), int(np.ceil(nCol/2))::] = 0
                Mask[1, int(np.ceil(nCol / 2))::] = 0
    return Mask

def completeSpec(halfSpec):

    mRow, nCol = halfSpec.shape
    fullSpec = torch.zeros_like(halfSpec)

    if np.mod(mRow, 2) == 1 and np.mod(nCol, 2) == 1:
        halfSpec = halfSpec + torch.rot90(halfSpec, 2)
        halfSpec[int(np.ceil(mRow/2))-1, int(np.ceil(nCol/2))-1] *= 0.5
        fullSpec = halfSpec
    else:
        if np.mod(mRow, 2) == 0 and np.mod(nCol, 2) == 0:
            RightBottomHalfSpec = halfSpec[1::, 1::]
            RightBottomFullSpec = completeSpec(RightBottomHalfSpec)
            fullSpec[1::, 1::] = RightBottomFullSpec
            TopLine = halfSpec[0, 1::]
            TopLine = completeSpec(TopLine.reshape([1, len(TopLine)])).reshape(len(TopLine))
            fullSpec[0, 1::] = TopLine

            LeftColumn = halfSpec[1::, 0]
            LeftColumn = completeSpec(LeftColumn.reshape([len(LeftColumn), 1])).reshape(len(LeftColumn))

            fullSpec[1::, 0] = LeftColumn
            fullSpec[0, 0] = halfSpec[0, 0]
        else:
            if np.mod(mRow, 2) == 1 and np.mod(nCol, 2) == 0:
                LeftColumn = halfSpec[0::, 0]
                LeftColumn = completeSpec(LeftColumn.reshape([len(LeftColumn), 1])).reshape(len(LeftColumn))

                RightHalfSpec = halfSpec[:, 1::]
                RightFullSpec = completeSpec(RightHalfSpec)

                fullSpec[0::, 0] = LeftColumn
                fullSpec[:, 1::] = RightFullSpec
            else:
                halfSpec = halfSpec.t()
                fullSpec = completeSpec(halfSpec)
                fullSpec = fullSpec.t()
    return fullSpec

def RescaleProbMap(x, sampling_times):
    # sampling_ratio = sampling_times/(3*(x.shape[0] * x.shape[1]))
    sampling_ratio = sampling_times / (x.shape[0] * x.shape[1])
    xbar = torch.mean(x)
    r = sampling_ratio / xbar
    beta = (1 - sampling_ratio) / (1 - xbar)
    # compute adjucement
    le = torch.le(r, 1).float()
    return le * x * r + (1 - le) * (1 - (1 - x) * beta)

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def find(condition):
    res = np.nonzero(condition)
    return res

def minmaxscaler(data):
    datamin = data.min()
    datamax = data.max()
    return (data - datamin)/(datamax - datamin)

def create_data_lists(train_folders, test_folders, verify_folders, min_size, output_folder):
    """
    Create lists for images in the training set and each of the test sets.
    :param train_folders: folders containing the training images; these will be merged
    :param test_folders: folders containing the test images; each test folder will form its own test set 
    :param min_size: minimum width and height of images to be considered
    :param output_folder: save data lists here
    """
    print("\nCreating data lists... this may take some time.\n")

    for d in train_folders:
        train_images = list()
        train_name = d.split("/")[-1] 
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                train_images.append(img_path)
    print("There are %d images in the %s training data.\n" % (len(train_images), train_name))
    with open(os.path.join(output_folder, train_name + '_train_images.json'), 'w') as j:
        json.dump(train_images, j)

    for d in test_folders:
        test_images = list()
        test_name = d.split("/")[-1]
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                test_images.append(img_path)
        print("There are %d images in the %s test data.\n" % (len(test_images), test_name))
        with open(os.path.join(output_folder, test_name + '_test_images.json'), 'w') as j:
            json.dump(test_images, j)

    for d in verify_folders:
        verify_images = list()
        verify_name = d.split("/")[-1]
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                verify_images.append(img_path)
        print("There are %d images in the %s verify data.\n" % (len(verify_images), verify_name))
        with open(os.path.join(output_folder, verify_name + '_verify_images.json'), 'w') as j:
            json.dump(verify_images, j)

    print("JSONS containing lists of Train / Test / Verify images have been saved to %s\n" % output_folder)

def reGetMask(mask_half0,SR_Point_num):
    # Calculating the difference
    mask_diff = mask_half0.sum() - SR_Point_num
    # Contrast that with the fixcircle
    mRow, nCol = mask_half0.size()
    mask_half = mask_half0.cpu().numpy()
    SR_Point = SR_Point_num / (mRow * nCol)
    fixmask = GenerateCircleSpecMask(mRow,nCol,SR_Point)
    temp = mask_half - mask_half * fixmask
    # For squares with even side lengths
    CenterX = mRow / 2
    CenterY = nCol / 2
    [Centerx, Centery] = np.meshgrid(np.arange(-CenterX,CenterX,1),np.arange(-CenterY,CenterY,1))
    CenterCircle = Centerx ** 2 + Centery ** 2
    tempCC = CenterCircle * temp
    
    temp_1D = temp.reshape((-1,1))
    tempCC_1D = tempCC.reshape((-1,1))
    ind = np.argsort(-tempCC_1D, axis=0)
    temp_1D[ind[0:int(mask_diff)]] = 0
    temp_2D = temp_1D.reshape(mRow, nCol)
  
    mask0 = mask_half * fixmask + temp_2D
    mask = torch.from_numpy(mask0)
    return mask


def GetMask(mask_temp,SR_Point_num):
    for ratio in range(200):
        ratio_temp = ratio/100
        mask_half = torch.ge(mask_temp, ratio_temp).float()
        if mask_half.sum() <= SR_Point_num:
            if mask_half.sum() == SR_Point_num:
                print(mask_half.sum())
            else:
                mask_half = torch.ge(mask_temp, (ratio-1)/100).float()
                mask_half = reGetMask(mask_half,SR_Point_num)
                #print(mask_half.sum())
            break

    return mask_half

