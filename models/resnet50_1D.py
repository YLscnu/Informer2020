import torch
from torchsummary import summary

class Bottlrneck(torch.nn.Module):
    def __init__(self,In_channel,Med_channel,Out_channel,downsample=False):
        super(Bottlrneck, self).__init__()
        self.stride = 1
        if downsample == True:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Out_channel, 1),
            torch.nn.BatchNorm1d(Out_channel),
            torch.nn.ReLU(),
        )

        if In_channel != Out_channel:
            self.res_layer = torch.nn.Conv1d(In_channel, Out_channel,1,self.stride)
        else:
            self.res_layer = None

    def forward(self,x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x)+residual

#-------------------------------------------------------------------------------------------#
# 输入通道就是72，也就是48(老代)+24(要预测的个数)，输出通道数应该还是72，然后最后一个维度是512，一直变化为最后输出的是1
# 上述Bottlrneck上面是单个残差块的实现，下面是ResNet50整体模型的实现
#-------------------------------------------------------------------------------------------#

class ResNet50(torch.nn.Module):
    def __init__(self,in_channels=2,classes=5):
        super(ResNet50, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels,64,kernel_size=7,stride=2,padding=3),
            torch.nn.MaxPool1d(3,2,1),

            Bottlrneck(64,64,256,False),
            Bottlrneck(256,64,256,False),
            Bottlrneck(256,64,256,False),
            #
            Bottlrneck(256,128,512, True),
            Bottlrneck(512,128,512, False),
            Bottlrneck(512,128,512, False),
            Bottlrneck(512,128,512, False),
            #
            Bottlrneck(512,256,1024, True),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            #
            Bottlrneck(1024,512,2048, True),
            Bottlrneck(2048,512,2048, False),
            Bottlrneck(2048,512,2048, False),

            # torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(2048,1024),
            torch.nn.Linear(1024,512),
            torch.nn.Linear(512,256),
            torch.nn.Linear(256,classes)
        )

    def forward(self,x):
        x = self.features(x)
        x = x.permute(0,2,1)
        x = self.classifer(x)
        return x

# if __name__ == '__main__':
#     x = torch.randn(size=(1,1,224))
#         # x = torch.randn(size=(1,64,224))
#         # model = Bottlrneck(64,64,256,True)
#     model = ResNet(in_channels=1)

#     output = model(x)
#     print(f'输入尺寸为:{x.shape}')
#     print(f'输出尺寸为:{output.shape}')
#     print(model)
#     summary(model,(1,224),device='cpu')
#-----------------------------------------------------------------#
# ----------------------------------------------------------------#
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv1d-1              [-1, 64, 112]             512
#          MaxPool1d-2               [-1, 64, 56]               0
#             Conv1d-3              [-1, 256, 56]          16,640
#             Conv1d-4               [-1, 64, 56]           4,160
#        BatchNorm1d-5               [-1, 64, 56]             128
#               ReLU-6               [-1, 64, 56]               0
#             Conv1d-7               [-1, 64, 56]          12,352
#        BatchNorm1d-8               [-1, 64, 56]             128
#               ReLU-9               [-1, 64, 56]               0
#            Conv1d-10              [-1, 256, 56]          16,640
#       BatchNorm1d-11              [-1, 256, 56]             512
#              ReLU-12              [-1, 256, 56]               0
#        Bottlrneck-13              [-1, 256, 56]               0
#            Conv1d-14               [-1, 64, 56]          16,448
#       BatchNorm1d-15               [-1, 64, 56]             128
#              ReLU-16               [-1, 64, 56]               0
#            Conv1d-17               [-1, 64, 56]          12,352
#       BatchNorm1d-18               [-1, 64, 56]             128
#              ReLU-19               [-1, 64, 56]               0
#            Conv1d-20              [-1, 256, 56]          16,640
#       BatchNorm1d-21              [-1, 256, 56]             512
#              ReLU-22              [-1, 256, 56]               0
#        Bottlrneck-23              [-1, 256, 56]               0
#            Conv1d-24               [-1, 64, 56]          16,448
#       BatchNorm1d-25               [-1, 64, 56]             128
#              ReLU-26               [-1, 64, 56]               0
#            Conv1d-27               [-1, 64, 56]          12,352
#       BatchNorm1d-28               [-1, 64, 56]             128
#              ReLU-29               [-1, 64, 56]               0
#            Conv1d-30              [-1, 256, 56]          16,640
#       BatchNorm1d-31              [-1, 256, 56]             512
#              ReLU-32              [-1, 256, 56]               0
#        Bottlrneck-33              [-1, 256, 56]               0
#            Conv1d-34              [-1, 512, 28]         131,584
#            Conv1d-35              [-1, 128, 28]          32,896
#       BatchNorm1d-36              [-1, 128, 28]             256
#              ReLU-37              [-1, 128, 28]               0
#            Conv1d-38              [-1, 128, 28]          49,280
#       BatchNorm1d-39              [-1, 128, 28]             256
#              ReLU-40              [-1, 128, 28]               0
#            Conv1d-41              [-1, 512, 28]          66,048
#       BatchNorm1d-42              [-1, 512, 28]           1,024
#              ReLU-43              [-1, 512, 28]               0
#        Bottlrneck-44              [-1, 512, 28]               0
#            Conv1d-45              [-1, 128, 28]          65,664
#       BatchNorm1d-46              [-1, 128, 28]             256
#              ReLU-47              [-1, 128, 28]               0
#            Conv1d-48              [-1, 128, 28]          49,280
#       BatchNorm1d-49              [-1, 128, 28]             256
#              ReLU-50              [-1, 128, 28]               0
#            Conv1d-51              [-1, 512, 28]          66,048
#       BatchNorm1d-52              [-1, 512, 28]           1,024
#              ReLU-53              [-1, 512, 28]               0
#        Bottlrneck-54              [-1, 512, 28]               0
#            Conv1d-55              [-1, 128, 28]          65,664
#       BatchNorm1d-56              [-1, 128, 28]             256
#              ReLU-57              [-1, 128, 28]               0
#            Conv1d-58              [-1, 128, 28]          49,280
#       BatchNorm1d-59              [-1, 128, 28]             256
#              ReLU-60              [-1, 128, 28]               0
#            Conv1d-61              [-1, 512, 28]          66,048
#       BatchNorm1d-62              [-1, 512, 28]           1,024
#              ReLU-63              [-1, 512, 28]               0
#        Bottlrneck-64              [-1, 512, 28]               0
#            Conv1d-65              [-1, 128, 28]          65,664
#       BatchNorm1d-66              [-1, 128, 28]             256
#              ReLU-67              [-1, 128, 28]               0
#            Conv1d-68              [-1, 128, 28]          49,280
#       BatchNorm1d-69              [-1, 128, 28]             256
#              ReLU-70              [-1, 128, 28]               0
#            Conv1d-71              [-1, 512, 28]          66,048
#       BatchNorm1d-72              [-1, 512, 28]           1,024
#              ReLU-73              [-1, 512, 28]               0
#        Bottlrneck-74              [-1, 512, 28]               0
#            Conv1d-75             [-1, 1024, 14]         525,312
#            Conv1d-76              [-1, 256, 14]         131,328
#       BatchNorm1d-77              [-1, 256, 14]             512
#              ReLU-78              [-1, 256, 14]               0
#            Conv1d-79              [-1, 256, 14]         196,864
#       BatchNorm1d-80              [-1, 256, 14]             512
#              ReLU-81              [-1, 256, 14]               0
#            Conv1d-82             [-1, 1024, 14]         263,168
#       BatchNorm1d-83             [-1, 1024, 14]           2,048
#              ReLU-84             [-1, 1024, 14]               0
#        Bottlrneck-85             [-1, 1024, 14]               0
#            Conv1d-86              [-1, 256, 14]         262,400
#       BatchNorm1d-87              [-1, 256, 14]             512
#              ReLU-88              [-1, 256, 14]               0
#            Conv1d-89              [-1, 256, 14]         196,864
#       BatchNorm1d-90              [-1, 256, 14]             512
#              ReLU-91              [-1, 256, 14]               0
#            Conv1d-92             [-1, 1024, 14]         263,168
#       BatchNorm1d-93             [-1, 1024, 14]           2,048
#              ReLU-94             [-1, 1024, 14]               0
#        Bottlrneck-95             [-1, 1024, 14]               0
#            Conv1d-96              [-1, 256, 14]         262,400
#       BatchNorm1d-97              [-1, 256, 14]             512
#              ReLU-98              [-1, 256, 14]               0
#            Conv1d-99              [-1, 256, 14]         196,864
#      BatchNorm1d-100              [-1, 256, 14]             512
#             ReLU-101              [-1, 256, 14]               0
#           Conv1d-102             [-1, 1024, 14]         263,168
#      BatchNorm1d-103             [-1, 1024, 14]           2,048
#             ReLU-104             [-1, 1024, 14]               0
#       Bottlrneck-105             [-1, 1024, 14]               0
#           Conv1d-106              [-1, 256, 14]         262,400
#      BatchNorm1d-107              [-1, 256, 14]             512
#             ReLU-108              [-1, 256, 14]               0
#           Conv1d-109              [-1, 256, 14]         196,864
#      BatchNorm1d-110              [-1, 256, 14]             512
#             ReLU-111              [-1, 256, 14]               0
#           Conv1d-112             [-1, 1024, 14]         263,168
#      BatchNorm1d-113             [-1, 1024, 14]           2,048
#             ReLU-114             [-1, 1024, 14]               0
#       Bottlrneck-115             [-1, 1024, 14]               0
#           Conv1d-116              [-1, 256, 14]         262,400
#      BatchNorm1d-117              [-1, 256, 14]             512
#             ReLU-118              [-1, 256, 14]               0
#           Conv1d-119              [-1, 256, 14]         196,864
#      BatchNorm1d-120              [-1, 256, 14]             512
#             ReLU-121              [-1, 256, 14]               0
#           Conv1d-122             [-1, 1024, 14]         263,168
#      BatchNorm1d-123             [-1, 1024, 14]           2,048
#             ReLU-124             [-1, 1024, 14]               0
#       Bottlrneck-125             [-1, 1024, 14]               0
#           Conv1d-126              [-1, 256, 14]         262,400
#      BatchNorm1d-127              [-1, 256, 14]             512
#             ReLU-128              [-1, 256, 14]               0
#           Conv1d-129              [-1, 256, 14]         196,864
#      BatchNorm1d-130              [-1, 256, 14]             512
#             ReLU-131              [-1, 256, 14]               0
#           Conv1d-132             [-1, 1024, 14]         263,168
#      BatchNorm1d-133             [-1, 1024, 14]           2,048
#             ReLU-134             [-1, 1024, 14]               0
#       Bottlrneck-135             [-1, 1024, 14]               0
#           Conv1d-136              [-1, 2048, 7]       2,099,200
#           Conv1d-137               [-1, 512, 7]         524,800
#      BatchNorm1d-138               [-1, 512, 7]           1,024
#             ReLU-139               [-1, 512, 7]               0
#           Conv1d-140               [-1, 512, 7]         786,944
#      BatchNorm1d-141               [-1, 512, 7]           1,024
#             ReLU-142               [-1, 512, 7]               0
#           Conv1d-143              [-1, 2048, 7]       1,050,624
#      BatchNorm1d-144              [-1, 2048, 7]           4,096
#             ReLU-145              [-1, 2048, 7]               0
#       Bottlrneck-146              [-1, 2048, 7]               0
#           Conv1d-147               [-1, 512, 7]       1,049,088
#      BatchNorm1d-148               [-1, 512, 7]           1,024
#             ReLU-149               [-1, 512, 7]               0
#           Conv1d-150               [-1, 512, 7]         786,944
#      BatchNorm1d-151               [-1, 512, 7]           1,024
#             ReLU-152               [-1, 512, 7]               0
#           Conv1d-153              [-1, 2048, 7]       1,050,624
#      BatchNorm1d-154              [-1, 2048, 7]           4,096
#             ReLU-155              [-1, 2048, 7]               0
#       Bottlrneck-156              [-1, 2048, 7]               0
#           Conv1d-157               [-1, 512, 7]       1,049,088
#      BatchNorm1d-158               [-1, 512, 7]           1,024
#             ReLU-159               [-1, 512, 7]               0
#           Conv1d-160               [-1, 512, 7]         786,944
#      BatchNorm1d-161               [-1, 512, 7]           1,024
#             ReLU-162               [-1, 512, 7]               0
#           Conv1d-163              [-1, 2048, 7]       1,050,624
#      BatchNorm1d-164              [-1, 2048, 7]           4,096
#             ReLU-165              [-1, 2048, 7]               0
#       Bottlrneck-166              [-1, 2048, 7]               0
# AdaptiveAvgPool1d-167              [-1, 2048, 1]               0
#           Linear-168                    [-1, 5]          10,245
# ================================================================
# Total params: 15,983,237
# Trainable params: 15,983,237
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.00
# Forward/backward pass size (MB): 10.16
# Params size (MB): 60.97
# Estimated Total Size (MB): 71.13
# ----------------------------------------------------------------