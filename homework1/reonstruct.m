
% bp_imageRecon.m
 
%% 清理
clear,clc
close all
 
%% 载入数据
col=256;
row=256;
I=imread('./lena.bmp'); % 重新载入原始图片，用作对比
% 统一将形状转为row*col
I = imresize(I,[row,col]); % Resize image

load comp
com.lw=double(com.lw)/63;
com.b=double(com.b)/63;
com.d=double(com.d)/63;
com.lw=com.lw*(maxlw-minlw)+minlw;
com.b=com.b*(maxb-minb)+minb;
com.d=com.d*(maxd-mind)+mind;
 
%% 重建
for i=1:4096
   Y(:,i)=com.lw*(com.d(:,i)) +com.b;
end
 
%% 反归一化
Y=uint8(Y*255);
 
%% 图像块恢复
I1=re_divide(Y,col,4); % 将重建后的图片存储在I1变量中
 
%% 计算性能
fprintf('PSNR :\n  ');
psnr=10*log10(255^2*row*col/sum(sum((I-I1).^2)));
disp(psnr)
a=dir();
for i=1:length(a)
   if (strcmp(a(i).name,'comp.mat')==1) 
       si=a(i).bytes;
       break;
   end
end
fprintf('rate: \n  ');
rate=double(si)/(256*256);
disp(rate) % Display value of variable
figure(1) % Create figure window
imshow(I) % Display image
title('原始图像');
figure(2)
imshow(I1)
title('重建图像');
