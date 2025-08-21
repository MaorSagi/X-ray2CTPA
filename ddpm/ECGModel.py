import torch
import torch.nn as nn


class ECGCNNModel(nn.Module):
    """CNN model for ECG data processing."""

    def __init__(self, ):
        super(ECGCNNModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(9, 9), padding='same')
        self.norm1 = nn.BatchNorm2d(16)
        self.norm2 = nn.BatchNorm2d(16)
        self.norm3 = nn.BatchNorm2d(16)
        self.norm4 = nn.BatchNorm2d(16)
        self.norm32 = nn.BatchNorm2d(32)
        self.norm64 = nn.BatchNorm2d(64)
        self.norm1D = nn.BatchNorm1d(64)
        self.norm1D32 = nn.BatchNorm1d(32)
        self.norm128 = nn.BatchNorm2d(128)
        self.act1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d((4, 1))
        self.maxpool2 = nn.MaxPool2d((2, 1))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(7, 7), padding='same')
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), padding='same')
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), padding='same')
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding='same')
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding='same')
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, stride=(1, 12), kernel_size=(1, 12),
                               padding='valid')
        self.fc1 = nn.Linear(in_features=1152, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=2)
        self.drop = nn.Dropout(0.1)
        self.flaten = nn.Flatten()
        # self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act1(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act1(x)
        x = self.maxpool1(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.act1(x)
        x = self.maxpool2(x)
        x = self.conv5(x)
        x = self.norm32(x)
        x = self.act1(x)
        x = self.maxpool2(x)
        x = self.conv6(x)
        x = self.norm64(x)
        x = self.act1(x)
        x = self.maxpool2(x)
        x = self.conv7(x)
        x = self.norm128(x)
        x = self.act1(x)
        x = self.flaten(x)
        x = self.fc1(x)
        x = self.norm1D(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.norm1D32(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc3(x)
        # x = self.sigmoid(x)
        x = self.softmax(x)

        return x

    # def __init__(self, input_channels=12, num_classes=1):
    #     super().__init__()
    #
    #     self.features = nn.Sequential(
    #         nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
    #         nn.BatchNorm1d(64),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
    #
    #         nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
    #         nn.BatchNorm1d(128),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool1d(kernel_size=2, stride=2),
    #
    #         nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
    #         nn.BatchNorm1d(256),
    #         nn.ReLU(inplace=True),
    #
    #         nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
    #         nn.BatchNorm1d(256),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool1d(kernel_size=2, stride=2),
    #     )
    #
    #     self.classifier = nn.Sequential(
    #         nn.AdaptiveAvgPool1d(1),
    #         nn.Flatten(),
    #         nn.Linear(256, 128),
    #         nn.ReLU(inplace=True),
    #         nn.Dropout(0.5),
    #         nn.Linear(128, num_classes)
    #     )
    #
    #     self._input_shape = (input_channels, 1000)  # Default ECG length of 1000
    #
    # def forward(self, x):
    #     x = self.features(x)
    #     x = self.classifier(x)
    #     return x

