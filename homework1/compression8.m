% bp_imageCompress.m
% 基于BP神经网络的图像压缩
 
%% 清理
clc % Clear Command Window
clear all
rng(0) % Control random number generation
 
%% 压缩率控制
K=4; % 设置将图片分成的小像素块的大小（KxK）
N=2; % 设置创建的BP神经网络的隐层数
row=256; % 设置所有进行压缩的图像的原始大小都为256x256
col=256;
 
%% 数据输入
I=imread('./lena.bmp');
 
% 统一将形状转为row*col
I = imresize(I,[row,col]); % Resize image
 
%% 图像块划分，形成K^2*N矩阵
P=block_divide(I,K); % 调用自定义函数block_divide将图像进行划分，形成K^2 x N大小的矩阵
 
%% 归一化
P=double(P)/255; % 对每一个像素点进行归一化处理
 
%% 建立BP神经网络 feedforwardnet(hiddenSizes,trainFcn)
net=feedforwardnet(N,'trainlm'); % Feedforward neural network
T=P;
net.trainParam.goal=0.001; % 设置BP网络训练参数
net.trainParam.epochs=500;
tic % Start stopwatch timer
net=train(net,P,T); % Train neural network
toc % Read elapsed time from stopwatch
 
%% 保存结果
com.lw=net.lw{2};
com.b=net.b{2};
[~,len]=size(P); % 训练样本的个数
com.d=zeros(N,len);
for i=1:len
    com.d(:,i)=tansig(net.iw{1}*P(:,i)+net.b{1}); % Hyperbolic tangent sigmoid transfer function
end
minlw= min(com.lw(:));
maxlw= max(com.lw(:));
com.lw=(com.lw-minlw)/(maxlw-minlw);
minb= min(com.b(:));
maxb= max(com.b(:));
com.b=(com.b-minb)/(maxb-minb);
maxd=max(com.d(:));
mind=min(com.d(:));
com.d=(com.d-mind)/(maxd-mind);
 
com.lw=uint8(com.lw*63);
com.b=uint8(com.b*63);
com.d=uint8(com.d*63);
 
save comp com minlw maxlw minb maxb maxd mind
