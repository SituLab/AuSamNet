function [spec,img_norm_G,imgColor] = MyEval(IntensityMat, g, mode)
%% 
if(~exist('mode','var'))
    mode = 'color';  
end



%% 
[img, spec] = getFSPIReconstruction( IntensityMat, 3, 120);
img_norm = (img - min(img(:)))./(max(img(:))-min(img(:)));


if strcmp(mode,'color')
%% color correction g=0.6-0.75
img_norm_G = img_norm;
img_norm_G(1:2:end,1:2:end) = g .* img_norm(1:2:end,1:2:end); % copy green(G)
img_norm_G(2:2:end,2:2:end) = g .* img_norm(2:2:end,2:2:end); % copy green(G)
img_norm_G = (img_norm_G - min(img_norm_G(:)))./(max(img_norm_G(:))-min(img_norm_G(:)));
%% demosaic
imgColor = im2double(demosaic(uint8(img_norm_G * 255), 'grbg'));
figure, subplot(1,3,1), specshow(spec), title('spec'), axis off;
subplot(1,3,2), imshow(img_norm_G), title('Gray');
subplot(1,3,3), imshow(imgColor), title('RGB');

elseif strcmp(mode,'gray')
    imgColor = img_norm;
    img_norm_G = img_norm;
    figure, subplot(1,2,1), specshow(spec), title('spec'), axis off;
    subplot(1,2,2), imshow(img_norm), title('Gray');
end

end