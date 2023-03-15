import torch
from torchsummary import summary

class Bottlrneck(torch.nn.Module):
    def __init__(self,In_channel,Med_channel,downsample=False):
        super(Bottlrneck, self).__init__()
        self.stride = 1
        if downsample == True:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(In_channel, Med_channel, 3, self.stride, padding=1),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU()
        )

        if In_channel != Med_channel:
            self.res_layer = torch.nn.Conv1d(In_channel, Med_channel,1,self.stride)
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

class ResNet34(torch.nn.Module):
    def __init__(self,in_channels=2,classes=5):
        super(ResNet34, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels,64,kernel_size=7,stride=2,padding=3),
            torch.nn.MaxPool1d(3,2,1),

            Bottlrneck(64,64, False),
            Bottlrneck(64,64, False),
            Bottlrneck(64,64, False),
            #
            Bottlrneck(64,128, True),
            Bottlrneck(128,128, False),
            Bottlrneck(128,128, False),
            Bottlrneck(128,128, False),
            #
            Bottlrneck(128,256, True),
            Bottlrneck(256,256, False),
            Bottlrneck(256,256, False),
            Bottlrneck(256,256, False),
            Bottlrneck(256,256, False),
            Bottlrneck(256,256, False),
            #
            Bottlrneck(256,512, True),
            Bottlrneck(512,512, False),
            Bottlrneck(512,512, False),

            # torch.nn.AdaptiveAvgPool1d(1)  # 我们做一维的时候不需要
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(512,256),
            torch.nn.Linear(256,classes)
        )

    def forward(self,x):
        x = self.features(x)
        # x = x.view(-1,512)  # 我们做一维的时候不需要
        x = x.permute(0,2,1)  # 我们做一维的时候需要开启
        x = self.classifer(x)
        return x

# if __name__ == '__main__':
#     x = torch.randn(size=(1,1,224))
#         # x = torch.randn(size=(1,64,224))
#         # model = Bottlrneck(64,64,256,True)
#     model = ResNet34(in_channels=1)

#     output = model(x)
#     print(f'输入尺寸为:{x.shape}')
#     print(f'输出尺寸为:{output.shape}')
#     print(model)
#     summary(model,(1,224),device='cpu')
#-----------------------------------------------------------------#
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv1d-1              [-1, 64, 112]             512
#          MaxPool1d-2               [-1, 64, 56]               0
#             Conv1d-3               [-1, 64, 56]          12,352
#        BatchNorm1d-4               [-1, 64, 56]             128
#               ReLU-5               [-1, 64, 56]               0
#             Conv1d-6               [-1, 64, 56]          12,352
#        BatchNorm1d-7               [-1, 64, 56]             128
#               ReLU-8               [-1, 64, 56]               0
#         Bottlrneck-9               [-1, 64, 56]               0
#            Conv1d-10               [-1, 64, 56]          12,352
#       BatchNorm1d-11               [-1, 64, 56]             128
#              ReLU-12               [-1, 64, 56]               0
#            Conv1d-13               [-1, 64, 56]          12,352
#       BatchNorm1d-14               [-1, 64, 56]             128
#              ReLU-15               [-1, 64, 56]               0
#        Bottlrneck-16               [-1, 64, 56]               0
#            Conv1d-17               [-1, 64, 56]          12,352
#       BatchNorm1d-18               [-1, 64, 56]             128
#              ReLU-19               [-1, 64, 56]               0
#            Conv1d-20               [-1, 64, 56]          12,352
#       BatchNorm1d-21               [-1, 64, 56]             128
#              ReLU-22               [-1, 64, 56]               0
#        Bottlrneck-23               [-1, 64, 56]               0
#            Conv1d-24              [-1, 128, 28]           8,320
#            Conv1d-25              [-1, 128, 28]          24,704
#       BatchNorm1d-26              [-1, 128, 28]             256
#              ReLU-27              [-1, 128, 28]               0
#            Conv1d-28              [-1, 128, 28]          49,280
#       BatchNorm1d-29              [-1, 128, 28]             256
#              ReLU-30              [-1, 128, 28]               0
#        Bottlrneck-31              [-1, 128, 28]               0
#            Conv1d-32              [-1, 128, 28]          49,280
#       BatchNorm1d-33              [-1, 128, 28]             256
#              ReLU-34              [-1, 128, 28]               0
#            Conv1d-35              [-1, 128, 28]          49,280
#       BatchNorm1d-36              [-1, 128, 28]             256
#              ReLU-37              [-1, 128, 28]               0
#        Bottlrneck-38              [-1, 128, 28]               0
#            Conv1d-39              [-1, 128, 28]          49,280
#       BatchNorm1d-40              [-1, 128, 28]             256
#              ReLU-41              [-1, 128, 28]               0
#            Conv1d-42              [-1, 128, 28]          49,280
#       BatchNorm1d-43              [-1, 128, 28]             256
#              ReLU-44              [-1, 128, 28]               0
#        Bottlrneck-45              [-1, 128, 28]               0
#            Conv1d-46              [-1, 128, 28]          49,280
#       BatchNorm1d-47              [-1, 128, 28]             256
#              ReLU-48              [-1, 128, 28]               0
#            Conv1d-49              [-1, 128, 28]          49,280
#       BatchNorm1d-50              [-1, 128, 28]             256
#              ReLU-51              [-1, 128, 28]               0
#        Bottlrneck-52              [-1, 128, 28]               0
#            Conv1d-53              [-1, 256, 14]          33,024
#            Conv1d-54              [-1, 256, 14]          98,560
#       BatchNorm1d-55              [-1, 256, 14]             512
#              ReLU-56              [-1, 256, 14]               0
#            Conv1d-57              [-1, 256, 14]         196,864
#       BatchNorm1d-58              [-1, 256, 14]             512
#              ReLU-59              [-1, 256, 14]               0
#        Bottlrneck-60              [-1, 256, 14]               0
#            Conv1d-61              [-1, 256, 14]         196,864
#       BatchNorm1d-62              [-1, 256, 14]             512
#              ReLU-63              [-1, 256, 14]               0
#            Conv1d-64              [-1, 256, 14]         196,864
#       BatchNorm1d-65              [-1, 256, 14]             512
#              ReLU-66              [-1, 256, 14]               0
#        Bottlrneck-67              [-1, 256, 14]               0
#            Conv1d-68              [-1, 256, 14]         196,864
#       BatchNorm1d-69              [-1, 256, 14]             512
#              ReLU-70              [-1, 256, 14]               0
#            Conv1d-71              [-1, 256, 14]         196,864
#       BatchNorm1d-72              [-1, 256, 14]             512
#              ReLU-73              [-1, 256, 14]               0
#        Bottlrneck-74              [-1, 256, 14]               0
#            Conv1d-75              [-1, 256, 14]         196,864
#       BatchNorm1d-76              [-1, 256, 14]             512
#              ReLU-77              [-1, 256, 14]               0
#            Conv1d-78              [-1, 256, 14]         196,864
#       BatchNorm1d-79              [-1, 256, 14]             512
#              ReLU-80              [-1, 256, 14]               0
#        Bottlrneck-81              [-1, 256, 14]               0
#            Conv1d-82              [-1, 256, 14]         196,864
#       BatchNorm1d-83              [-1, 256, 14]             512
#              ReLU-84              [-1, 256, 14]               0
#            Conv1d-85              [-1, 256, 14]         196,864
#       BatchNorm1d-86              [-1, 256, 14]             512
#              ReLU-87              [-1, 256, 14]               0
#        Bottlrneck-88              [-1, 256, 14]               0
#            Conv1d-89              [-1, 256, 14]         196,864
#       BatchNorm1d-90              [-1, 256, 14]             512
#              ReLU-91              [-1, 256, 14]               0
#            Conv1d-92              [-1, 256, 14]         196,864
#       BatchNorm1d-93              [-1, 256, 14]             512
#              ReLU-94              [-1, 256, 14]               0
#        Bottlrneck-95              [-1, 256, 14]               0
#            Conv1d-96               [-1, 512, 7]         131,584
#            Conv1d-97               [-1, 512, 7]         393,728
#       BatchNorm1d-98               [-1, 512, 7]           1,024
#              ReLU-99               [-1, 512, 7]               0
#           Conv1d-100               [-1, 512, 7]         786,944
#      BatchNorm1d-101               [-1, 512, 7]           1,024
#             ReLU-102               [-1, 512, 7]               0
#       Bottlrneck-103               [-1, 512, 7]               0
#           Conv1d-104               [-1, 512, 7]         786,944
#      BatchNorm1d-105               [-1, 512, 7]           1,024
#             ReLU-106               [-1, 512, 7]               0
#           Conv1d-107               [-1, 512, 7]         786,944
#      BatchNorm1d-108               [-1, 512, 7]           1,024
#             ReLU-109               [-1, 512, 7]               0
#       Bottlrneck-110               [-1, 512, 7]               0
#           Conv1d-111               [-1, 512, 7]         786,944
#      BatchNorm1d-112               [-1, 512, 7]           1,024
#             ReLU-113               [-1, 512, 7]               0
#           Conv1d-114               [-1, 512, 7]         786,944
#      BatchNorm1d-115               [-1, 512, 7]           1,024
#             ReLU-116               [-1, 512, 7]               0
#       Bottlrneck-117               [-1, 512, 7]               0
# AdaptiveAvgPool1d-118               [-1, 512, 1]               0
#           Linear-119                  [-1, 256]         131,328
#           Linear-120                    [-1, 5]           1,285
# ================================================================
# Total params: 7,357,445
# Trainable params: 7,357,445
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.00
# Forward/backward pass size (MB): 3.23
# Params size (MB): 28.07
# Estimated Total Size (MB): 31.30
# ----------------------------------------------------------------