
import torch
from torch import nn
from utils import *
import scipy.io as sio


class ConvolutionalBlock(nn.Module):
    """
    A convolutional block, comprising convolutional, BN, activation layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channe;s
        :param kernel_size: kernel size
        :param stride: stride
        :param batch_norm: include a BN layer?
        :param activation: Type of activation; None if none
        """
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}

        # A container that will hold the layers in this convolutional block
        layers = list()

        # A convolutional layer
        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2))

        # A batch normalization (BN) layer, if wanted
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        # An activation layer, if wanted
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        # Put together the convolutional block as a sequence of the layers in this container
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        """
        Forward propagation.

        :param input: input images, a tensor of size (N, in_channels, w, h)
        :return: output images, a tensor of size (N, out_channels, w, h)
        """
        output = self.conv_block(input)  # (N, out_channels, w, h)

        return output


class SubPixelConvolutionalBlock(nn.Module):
    """
    A subpixel convolutional block, comprising convolutional, pixel-shuffle, and PReLU activation layers.
    """

    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
        """
        :param kernel_size: kernel size of the convolution
        :param n_channels: number of input and output channels
        :param scaling_factor: factor to scale input images by (along both dimensions)
        """
        super(SubPixelConvolutionalBlock, self).__init__()

        # A convolutional layer that increases the number of channels by scaling factor^2, followed by pixel shuffle and PReLU
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2),
                              kernel_size=kernel_size, padding=kernel_size // 2)
        # These additional channels are shuffled to form additional pixels, upscaling each dimension by the scaling factor
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.prelu = nn.PReLU()

    def forward(self, input):
        """
        Forward propagation.

        :param input: input images, a tensor of size (N, n_channels, w, h)
        :return: scaled output images, a tensor of size (N, n_channels, w * scaling factor, h * scaling factor)
        """
        output = self.conv(input)  # (N, n_channels * scaling factor^2, w, h)
        output = self.pixel_shuffle(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.prelu(output)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return output


class ResidualBlock(nn.Module):
    """
    A residual block, comprising two convolutional blocks with a residual connection across them.
    """

    def __init__(self, kernel_size=3, n_channels=64):
        """
        :param kernel_size: kernel size
        :param n_channels: number of input and output channels (same because the input must be added to the output)
        """
        super(ResidualBlock, self).__init__()

        # The first convolutional block
        self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation='PReLu')

        # The second convolutional block
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation=None)

    def forward(self, input):
        """
        Forward propagation.

        :param input: input images, a tensor of size (N, n_channels, w, h)
        :return: output images, a tensor of size (N, n_channels, w, h)
        """
        residual = input  # (N, n_channels, w, h)
        output = self.conv_block1(input)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)
        output = output + residual  # (N, n_channels, w, h)

        return output


class ResNetColor(nn.Module):
    def  __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, 
                 train_mask=True, mode='M1', SR_Point=0.3):

        super(ResNetColor, self).__init__() 
        
        if mode == 'M3' :
            self.mask = torch.nn.Parameter(torch.rand(128, 128), requires_grad=train_mask)
            self.slope1= torch.nn.Parameter(torch.tensor(5.1), requires_grad=train_mask)
            self.slope2 = torch.nn.Parameter(torch.tensor(200.1), requires_grad=train_mask)

        if mode == 'M2':
            mask_temp0 = GenerateCircleSpecMask(128, 128, SR_Point)
            self.mask = torch.from_numpy(mask_temp0).float().to('cuda')


        if mode == 'M2_Eval':
            mask_temp0 = GenerateCircleSpecMask(128, 128, SR_Point)
            self.mask = torch.from_numpy(mask_temp0).float().to('cuda')


        if mode == 'M3_Eval':
            self.mask = GenerateCircleSpecMask(128,128,SR_Point) 
            self.slope1= torch.nn.Parameter(torch.tensor(5.1), requires_grad=train_mask)
            self.slope2 = torch.nn.Parameter(torch.tensor(200.1), requires_grad=train_mask)

        # The first convolutional block
        self.conv_block1 = ConvolutionalBlock(in_channels=1, out_channels=n_channels, kernel_size=large_kernel_size,
                                                batch_norm=False, activation='PReLu')
        # A sequence of n_blocks residual blocks, each containing a skip-connection across the block
        self.residual_blocks = nn.Sequential(
                    *[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for i in range(n_blocks)])
        # Another convolutional block
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                                kernel_size=small_kernel_size,
                                                batch_norm=True, activation=None)
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=1, kernel_size=large_kernel_size,
                                                batch_norm=False, activation='leakyrelu')


 

    def forward(self, hr_img_Bayer, mode, SR_Point_num):
        mask_temp = torch.zeros(128, 128).to('cuda')
        mask_full = torch.zeros(128, 128).to('cuda')

        if mode == 'M3':
            half_mask = getHalfSpecMask(128, 128)
            mask_half = torch.from_numpy(half_mask).to('cuda')
            mask_temp = torch.sigmoid(self.slope1 * self.mask)
            mask_temp = RescaleProbMap(mask_temp, int(SR_Point_num))
            threshs = torch.distributions.uniform.Uniform(0, 1).sample(mask_temp.shape).to('cuda')
            mask_temp = torch.sigmoid(self.slope2 * (mask_temp - threshs))
            mask_temp = mask_temp * mask_half
            mask_full= completeSpec(mask_temp)

        if mode == 'M2':
            mask_temp = self.mask
            mask_full = completeSpec(mask_temp)

        
        if mode == 'M3_Eval':
            half_mask = getHalfSpecMask(128, 128)
            mask_half = torch.from_numpy(half_mask).to('cuda')
            mask_temp = torch.sigmoid(self.slope1 * self.mask)
            mask_temp = RescaleProbMap(mask_temp, int(SR_Point_num))
            mask_temp = torch.sigmoid(self.slope2 * mask_temp)
            mask_temp = mask_temp * mask_half
            mask_temp = GetMask(mask_temp, SR_Point_num)
            mask_full = completeSpec(mask_temp).to('cuda')

        if mode == 'M2_Eval':
            mask_temp = self.mask
            mask_full = completeSpec(mask_temp)

    

        im_ifft_masked = torch.zeros_like(hr_img_Bayer)
        for i in range(hr_img_Bayer.shape[0]):
            im_fft = torch.fft.fftshift(torch.fft.fft2(torch.reshape(hr_img_Bayer[i, 0, :, :], [128, 128])))
            im_fft_masked = torch.complex(mask_full, torch.zeros_like(mask_full)) * im_fft
            im_fft_masked_view = torch.log(1 + torch.abs(im_fft_masked))
            im_ifft_masked[i, 0, :, :] = (torch.fft.ifft2(torch.fft.fftshift(im_fft_masked)))

        
        output = self.conv_block1(im_ifft_masked)  
        residual = output  
        output = self.residual_blocks(output) 
        output = self.conv_block2(output) 
        output = output + residual 
        sr_img_Bayer = self.conv_block3(output)  
     

        return sr_img_Bayer, mask_temp, im_fft_masked_view, im_ifft_masked, mask_full



