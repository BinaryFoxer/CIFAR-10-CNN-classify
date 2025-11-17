import torch 
import torch.nn as nn
import torch.nn.functional as F

# 一个标准的pytorch模型类，继承自nn.Module(所有神经网络的基类)，可以使用pytorch提供的内置功能
class customResNet(nn.Module):      
    def __init__(self, num_classes=10):
        super(customResNet,self).__init__()

        # 初始卷积层
        # 总参数：3x3x3x64+64+64=1856
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 经过四个阶段
        # ----------------------------------------
        # 一个残差块，通道数：64->64, 图片尺寸：32x32
        # 无shortcut: stride=1
        # 总参数:3x3x64x64+64x2+3x3x64x64+64x2=73984
        self.layer1 = self._make_layer(64, 64, 1)

        # 两个残差块，通道数：64->128, 图片尺寸：16x16(只有第一个残差块的第一层卷积层设置stride控制下采样)
        # 第一个残差块有shortcut: stride=2, in_channels != out_channels
        # 第二个残差块无shortcut: stride=1, in_channels == out_channels
        # 总参数:
        # 残差块1（有shortcut）:
        # conv1:64x128x3x3      bn1:128x2
        # conv2:128x128x3x3     bn2:128x2       
        # shortcut:64x128x1x1   bn3:128x2
        # 共230144
        # 残差块2（无shortcut）:
        # conv1:128x128x3x3     bn1:128x2
        # conv2:128x128x3x3     bn2:128x2       
        # 共295424
        # layer2参数共:525568
        self.layer2 = self._make_layer(64, 128, 2, stride=2)

        # 两个残差块，通道数：128->256, 图片尺寸：8x8(只有第一个残差块的第一层卷积层设置stride控制下采样)
        # 第一个残差块有shortcut: stride=2, in_channels != out_channels
        # 第二个残差块无shortcut: stride=1, in_channels == out_channels
        # 总参数:128x256x3x3+256x2+256x256x3x3+256x2+128x256x1x1
        # +256x256x3x3+256x2+256x256x3x3+256x2=2099200
        # 总参数:
        # 残差块1（有shortcut）:
        # conv1:128x256x3x3     bn1:256x2
        # conv2:256x256x3x3     bn2:256x2       
        # shortcut:128x256x1x1  bn3:256x2
        # 共919040
        # 残差块2（无shortcut）:
        # conv1:256x256x3x3     bn1:256x2
        # conv2:256x256x3x3     bn2:256x2       
        # 共1180672
        # layer3参数共:2099712
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        # 一个残差块，通道数：256->512, 图片尺寸：4x4
        # 有shortcut: stride=2
        # 总参数:256x512x3x3+512x2+512x512x3x3+512x2+1x1x256x512=3672064
        self.layer4 = self._make_layer(256, 512, 1, stride=2)

        # 全局平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

        # 初始化参数，递归地返回模型及其所有子模块，包括卷积层、批量归一化层、残差块等
        for m in self.modules():
            # 检查m是否是卷积层的实例
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        '''
        内部残差块设置
        '''
        layers = []

        # 第一个残差块可能使用下采样
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        # 后续块保持相同维度
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        '''
        customResNet的前向传播函数
        '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)   # 改变张量x的形状
        x = self.fc(x)

        return x
        

class ResidualBlock(nn.Module):
    '''
    定义残差块
    '''
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            ) 
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)  # 跳跃连接到第二个卷积(bn2)层的输出
        out = self.relu(out)

        return out
        

