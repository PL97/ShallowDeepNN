import torch.nn as nn


class CBR(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, kernel_size=3):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = nn.functional.relu(x)
        return x


class CBR_LargeT(nn.Module):
    def __init__(self, input_channels=3, kernel_size=7, stride=1):
        super(CBR_LargeT, self).__init__()
        self.in_channels = 32
        self.kernel_size = kernel_size
        self.stride = stride

        self.CBR1 = CBR(input_channels, self.in_channels, self.stride, self.kernel_size)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.CBR2 = CBR(self.in_channels, self.in_channels * 2, self.stride, self.kernel_size)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.CBR3 = CBR(self.in_channels * 2, self.in_channels * 4, self.stride, self.kernel_size)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.CBR4 = CBR(self.in_channels * 4, self.in_channels * 8, self.stride, self.kernel_size)
        self.pool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.CBR5 = CBR(self.in_channels * 8, self.in_channels * 16, self.stride, self.kernel_size)

        # classifier
        self.fc = nn.Linear(self.in_channels * 16, 2)

    def forward(self, x):
        x = self.CBR1(x)
        x = self.pool1(x)

        x = self.CBR2(x)
        x = self.pool2(x)

        x = self.CBR3(x)
        x = self.pool3(x)

        x = self.CBR4(x)
        x = self.pool4(x)

        x = self.CBR5(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CBR_LargeW(nn.Module):
    def __init__(self, input_channels=3, kernel_size=7, stride=1):
        super(CBR_LargeW, self).__init__()
        self.in_channels = 64
        self.kernel_size = kernel_size
        self.stride = stride

        self.CBR1 = CBR(input_channels, self.in_channels, self.stride, self.kernel_size)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.CBR2 = CBR(self.in_channels, self.in_channels * 2, self.stride, self.kernel_size)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.CBR3 = CBR(self.in_channels * 2, self.in_channels * 4, self.stride, self.kernel_size)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.CBR4 = CBR(self.in_channels * 4, self.in_channels * 8, self.stride, self.kernel_size)
        self.pool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        # classifier
        self.fc = nn.Linear(self.in_channels * 8, 2)

    def forward(self, x):
        x = self.CBR1(x)
        x = self.pool1(x)

        x = self.CBR2(x)
        x = self.pool2(x)

        x = self.CBR3(x)
        x = self.pool3(x)

        x = self.CBR4(x)
        x = self.pool4(x)

        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CBR_Small(nn.Module):
    def __init__(self, input_channels=3, kernel_size=7, stride=1):
        super(CBR_Small, self).__init__()
        self.in_channels = 32
        self.kernel_size = kernel_size
        self.stride = stride

        self.CBR1 = CBR(input_channels, self.in_channels, self.stride, self.kernel_size)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.CBR2 = CBR(self.in_channels, self.in_channels * 2, self.stride, self.kernel_size)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.CBR3 = CBR(self.in_channels * 2, self.in_channels * 4, self.stride, self.kernel_size)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.CBR4 = CBR(self.in_channels * 4, self.in_channels * 8, self.stride, self.kernel_size)
        self.pool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        # classifier
        self.fc = nn.Linear(self.in_channels * 8, 2)

    def forward(self, x):
        x = self.CBR1(x)
        x = self.pool1(x)

        x = self.CBR2(x)
        x = self.pool2(x)

        x = self.CBR3(x)
        x = self.pool3(x)

        x = self.CBR4(x)
        x = self.pool4(x)

        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CBR_Tiny(nn.Module):
    def __init__(self, input_channels=3, kernel_size=5, stride=1):
        super(CBR_Tiny, self).__init__()
        self.in_channels = 64
        self.kernel_size = kernel_size
        self.stride = stride

        self.CBR1 = CBR(input_channels, self.in_channels, self.stride, self.kernel_size)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.CBR2 = CBR(self.in_channels, self.in_channels * 2, self.stride, self.kernel_size)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.CBR3 = CBR(self.in_channels * 2, self.in_channels * 4, self.stride, self.kernel_size)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.CBR4 = CBR(self.in_channels * 4, self.in_channels * 8, self.stride, self.kernel_size)
        self.pool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        # classifier
        self.fc = nn.Linear(self.in_channels * 8, 2)

    def forward(self, x):
        x = self.CBR1(x)
        x = self.pool1(x)

        x = self.CBR2(x)
        x = self.pool2(x)

        x = self.CBR3(x)
        x = self.pool3(x)

        x = self.CBR4(x)
        x = self.pool4(x)

        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

