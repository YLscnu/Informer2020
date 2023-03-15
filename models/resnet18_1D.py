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
# 上述Bottlrneck上面是单个残差块的实现，下面是ResNet18整体模型的实现
#-------------------------------------------------------------------------------------------#

class ResNet18(torch.nn.Module):
    def __init__(self,in_channels=2,classes=5):
        super(ResNet18, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels,64,kernel_size=7,stride=2,padding=3),
            torch.nn.MaxPool1d(3,2,1),

            Bottlrneck(64,64, False),
            Bottlrneck(64,64, False),
            #
            Bottlrneck(64,128, True),
            Bottlrneck(128,128, False),
            #
            Bottlrneck(128,256, True),
            Bottlrneck(256,256, False),
            #
            Bottlrneck(256,512, True),
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
#     model = ResNet18(in_channels=1)

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
#            Conv1d-17              [-1, 128, 28]           8,320
#            Conv1d-18              [-1, 128, 28]          24,704
#       BatchNorm1d-19              [-1, 128, 28]             256
#              ReLU-20              [-1, 128, 28]               0
#            Conv1d-21              [-1, 128, 28]          49,280
#       BatchNorm1d-22              [-1, 128, 28]             256
#              ReLU-23              [-1, 128, 28]               0
#        Bottlrneck-24              [-1, 128, 28]               0
#            Conv1d-25              [-1, 128, 28]          49,280
#       BatchNorm1d-26              [-1, 128, 28]             256
#              ReLU-27              [-1, 128, 28]               0
#            Conv1d-28              [-1, 128, 28]          49,280
#       BatchNorm1d-29              [-1, 128, 28]             256
#              ReLU-30              [-1, 128, 28]               0
#        Bottlrneck-31              [-1, 128, 28]               0
#            Conv1d-32              [-1, 256, 14]          33,024
#            Conv1d-33              [-1, 256, 14]          98,560
#       BatchNorm1d-34              [-1, 256, 14]             512
#              ReLU-35              [-1, 256, 14]               0
#            Conv1d-36              [-1, 256, 14]         196,864
#       BatchNorm1d-37              [-1, 256, 14]             512
#              ReLU-38              [-1, 256, 14]               0
#        Bottlrneck-39              [-1, 256, 14]               0
#            Conv1d-40              [-1, 256, 14]         196,864
#       BatchNorm1d-41              [-1, 256, 14]             512
#              ReLU-42              [-1, 256, 14]               0
#            Conv1d-43              [-1, 256, 14]         196,864
#       BatchNorm1d-44              [-1, 256, 14]             512
#              ReLU-45              [-1, 256, 14]               0
#        Bottlrneck-46              [-1, 256, 14]               0
#            Conv1d-47               [-1, 512, 7]         131,584
#            Conv1d-48               [-1, 512, 7]         393,728
#       BatchNorm1d-49               [-1, 512, 7]           1,024
#              ReLU-50               [-1, 512, 7]               0
#            Conv1d-51               [-1, 512, 7]         786,944
#       BatchNorm1d-52               [-1, 512, 7]           1,024
#              ReLU-53               [-1, 512, 7]               0
#        Bottlrneck-54               [-1, 512, 7]               0
#            Conv1d-55               [-1, 512, 7]         786,944
#       BatchNorm1d-56               [-1, 512, 7]           1,024
#              ReLU-57               [-1, 512, 7]               0
#            Conv1d-58               [-1, 512, 7]         786,944
#       BatchNorm1d-59               [-1, 512, 7]           1,024
#              ReLU-60               [-1, 512, 7]               0
#        Bottlrneck-61               [-1, 512, 7]               0
# AdaptiveAvgPool1d-62               [-1, 512, 1]               0
#            Linear-62                  [-1, 256]         131,328
#            Linear-63                    [-1, 5]           1,285
# ================================================================
# Total params: 3,979,397
# Trainable params: 3,979,397
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.00
# Forward/backward pass size (MB): 1.70
# Params size (MB): 15.18
# Estimated Total Size (MB): 16.88
# ----------------------------------------------------------------