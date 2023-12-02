import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride_shape=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride_shape, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(output_channels)
        self.relu_1 = nn.ReLU()  # First ReLU Activation Function
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(output_channels)
        self.relu_2 = nn.ReLU()  # Second ReLU Activation Function
        self.downsample = None
        if stride_shape != 1 or input_channels != output_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride_shape),
            )

    def forward(self, input_tensor):
        # Store the input tensor as the residual connection
        residual = input_tensor

        output_tensor = self.conv1(input_tensor)
        output_tensor = self.batch_norm1(output_tensor)
        output_tensor = self.relu_1(output_tensor)
        output_tensor = self.conv2(output_tensor)
        output_tensor = self.batch_norm2(output_tensor)

        # If downsample exists, apply it to the input tensor
        if self.downsample is not None:
            residual = self.downsample(input_tensor)

        output_tensor += residual
        output_tensor = self.relu_2(output_tensor)

        return output_tensor


"""ResNet Class:"""


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.resblock1 = ResBlock(64, 64, stride_shape=1)
        self.resblock2 = ResBlock(64, 128, stride_shape=2)
        self.resblock3 = ResBlock(128, 256, stride_shape=2)
        self.dropout = nn.Dropout(p=0.5)
        self.resblock4 = ResBlock(256, 512, stride_shape=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        output_tensor = self.conv1(input_tensor)
        output_tensor = self.batch_norm1(output_tensor)
        output_tensor = self.relu(output_tensor)
        output_tensor = self.maxpool(output_tensor)
        output_tensor = self.resblock1(output_tensor)
        output_tensor = self.resblock2(output_tensor)
        output_tensor = self.resblock3(output_tensor)
        output_tensor = self.dropout(output_tensor)
        output_tensor = self.resblock4(output_tensor)
        output_tensor = self.avgpool(output_tensor)
        output_tensor = self.flatten(output_tensor)
        output_tensor = self.dropout2(output_tensor)
        output_tensor = self.fc(output_tensor)
        output_tensor = self.sigmoid(output_tensor)

        return output_tensor
