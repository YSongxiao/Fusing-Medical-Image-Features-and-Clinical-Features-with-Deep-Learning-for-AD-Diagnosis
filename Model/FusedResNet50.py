import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1(in_planes, out_planes, stride=2):
    '''
    :param in_planes: input channel
    :param out_planes: output channel
    :param stride:
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


class BottleNeck(nn.Module):
    def __init__(self, in_planes, out_planes, clinical_feature_shape, stride=1, expansion=4, downsampling=False):
        '''
        :param in_planes: input channel
        :param out_planes: output channel
        :param text_feature_shape: the shape of text_feature to be fused
        :param stride:
        :param expansion: default 4
        :param downsampling: whether residual block need to be downsampled
        '''
        super(BottleNeck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        self.clinical_feature_shape = clinical_feature_shape
        self.text_feature = None
        self.gate = nn.Sequential(
            nn.Linear(in_features=clinical_feature_shape[-1], out_features=1),
            nn.ReLU(inplace=True)
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, stride=stride, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_planes, out_channels=out_planes*expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_planes * expansion)
        )
        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=out_planes*expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_planes*expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def fuse(self, x):
        text_vec = self.text_feature
        channel = text_vec.size(1)
        batch_size = text_vec.size(0)
        text_vec = text_vec.reshape(batch_size, channel, 1, 1)
        temp_x = torch.mul(x, text_vec)
        temp_x = F.adaptive_avg_pool2d(temp_x, (1, 1))  # squeeze to batch_size*channel*1*1
        temp_x = torch.sigmoid(temp_x)
        x = torch.mul(x, temp_x)
        return x

    def forward(self, x, clinical_feature):  # modified:  x ---> attention(residual) ---> conv(residual)
        residual = x
        self.text_feature = self.gate(clinical_feature)
        x = self.fuse(x)
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class BottleneckNoAtt(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        '''
        :param in_places: input channel
        :param places: output channel
        :param stride:
        :param downsampling:
        :param expansion:
        '''
        super(BottleneckNoAtt, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class FusedResNet50(nn.Module):
    def __init__(self, clinical_path, clinical_feature_dim, clinical_feature_shape, num_classes=2, expansion=4):
        '''
        :param clinical_path:
        :param clinical_feature_dim:
        :param clinical_feature_shape: the number of clinical features' channel
        :param num_classes:
        :param expansion:
        '''
        super(FusedResNet50, self).__init__()
        self.expansion = expansion
        self.clinical_path = clinical_path
        self.num_classes = num_classes
        self.clinical_feature_shape = clinical_feature_shape

        self.inc_channel64 = nn.Sequential(
            nn.Conv1d(in_channels=clinical_feature_shape[1], out_channels=64, kernel_size=1,
                      padding=0),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.inc_channel256 = nn.Sequential(
            nn.Conv1d(in_channels=clinical_feature_shape[1], out_channels=256, kernel_size=1,
                      padding=0),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True)
        )

        self.inc_channel512 = nn.Sequential(
            nn.Conv1d(in_channels=clinical_feature_shape[1], out_channels=512, kernel_size=1,
                      padding=0),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True)
        )

        self.inc_channel4 = nn.Sequential(
            nn.Conv1d(in_channels=clinical_feature_shape[1], out_channels=2 * clinical_feature_shape[1], kernel_size=1,
                      padding=0),
            nn.BatchNorm1d(num_features=2 * clinical_feature_shape[1]),
            nn.ReLU(inplace=True)
        )

        self.inc_channel41 = nn.Sequential(
            nn.Conv1d(in_channels=clinical_feature_shape[1], out_channels=clinical_feature_shape[1], kernel_size=1,
                      padding=0),
            nn.BatchNorm1d(num_features=clinical_feature_shape[1]),
            nn.ReLU(inplace=True)
        )

        self.conv1 = conv1(in_planes=1, out_planes=64)

        # Layer1: 3 x BottleNeck
        self.bnk1_1 = BottleneckNoAtt(in_places=64, places=64, stride=1, downsampling=True)
        self.bnk1_2 = BottleneckNoAtt(in_places=256, places=64, stride=1)
        self.bnk1_3 = BottleneckNoAtt(in_places=256, places=64, stride=1)

        # Layer2: 4 x BottleNeck
        self.bnk2_1 = BottleNeck(in_planes=256, out_planes=128, clinical_feature_shape=list(clinical_feature_shape), stride=2, downsampling=True)
        self.bnk2_2 = BottleNeck(in_planes=512, out_planes=128, clinical_feature_shape=list(clinical_feature_shape), stride=1)
        self.bnk2_3 = BottleNeck(in_planes=512, out_planes=128, clinical_feature_shape=list(clinical_feature_shape), stride=1)
        self.bnk2_4 = BottleNeck(in_planes=512, out_planes=128, clinical_feature_shape=list(clinical_feature_shape), stride=1)

        # Layer3: 6 x BottleNeck
        self.bnk3_1 = BottleneckNoAtt(in_places=512, places=256, stride=2, downsampling=True)
        self.bnk3_2 = BottleneckNoAtt(in_places=1024, places=256, stride=1)
        self.bnk3_3 = BottleneckNoAtt(in_places=1024, places=256, stride=1)
        self.bnk3_4 = BottleneckNoAtt(in_places=1024, places=256, stride=1)
        self.bnk3_5 = BottleneckNoAtt(in_places=1024, places=256, stride=1)
        self.bnk3_6 = BottleneckNoAtt(in_places=1024, places=256, stride=1)

        # Layer4: 3 x BottleNeck
        shape4 = list(clinical_feature_shape)
        shape4[1] = shape4[1] * expansion
        self.bnk4_1 = BottleNeck(in_planes=1024, out_planes=512, clinical_feature_shape=list(clinical_feature_shape), stride=2, downsampling=True)
        self.bnk4_2 = BottleNeck(in_planes=2048, out_planes=512, clinical_feature_shape=shape4, stride=1)
        self.bnk4_3 = BottleNeck(in_planes=2048, out_planes=512, clinical_feature_shape=shape4, stride=1)

        # pooling
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # concat the clinical feature
        self.text_interface = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        self.bn = nn.BatchNorm1d(num_features=1)
        self.relu = nn.ReLU()

        # classification
        self.fc = nn.Linear(2048 * 4 + clinical_feature_dim, num_classes)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Conv1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, text):
        f4 = self.clinical_path(text)
        f2 = self.inc_channel256(f4)
        f3 = self.inc_channel512(f4)
        f4x1 = self.inc_channel41(f4)
        f4x4 = self.inc_channel4(f4)
        x = self.conv1(x)
        # layer 1
        x = self.bnk1_1(x)
        x = self.bnk1_2(x)
        x = self.bnk1_3(x)
        # layer 2
        x = self.bnk2_1(x, f2)
        x = self.bnk2_2(x, f3)
        x = self.bnk2_3(x, f3)
        x = self.bnk2_4(x, f3)
        # layer 3
        x = self.bnk3_1(x)
        x = self.bnk3_2(x)
        x = self.bnk3_3(x)
        x = self.bnk3_4(x)
        x = self.bnk3_5(x)
        x = self.bnk3_6(x)
        # layer 4
        x = self.bnk4_1(x, f4x1)
        x = self.bnk4_2(x, f4x4)
        x = self.bnk4_3(x, f4x4)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # concat f4
        f4 = f4.view(f4.size(0), -1)
        x = torch.cat((x, f4), 1)
        feature = x
        x = self.fc(x)
        return feature, x


class ClinicalPath(nn.Module):
    def __init__(self):
        super(ClinicalPath, self).__init__()
        self.encoder_p1 = self.make_encoder_layer_with_pool(in_planes=1, out_planes=64, conv_kernel_size=3,
                                                            padding=1, pool_kernel_size=2)

        self.encoder_p2 = self.make_encoder_layer_with_pool(in_planes=64, out_planes=256, conv_kernel_size=3,
                                                            padding=1, pool_kernel_size=2)

        self.encoder_p3 = self.make_encoder_layer_with_pool(in_planes=256, out_planes=512, conv_kernel_size=3,
                                                            padding=1, pool_kernel_size=2)

        self.encoder_p4 = self.make_encoder_layer_with_pool(in_planes=512, out_planes=1024, conv_kernel_size=3,
                                                            padding=1, pool_kernel_size=2)

        self.encoder_p5 = self.make_encoder_layer_with_pool(in_planes=1024, out_planes=1024, conv_kernel_size=3,
                                                            padding=1, pool_kernel_size=2)

        self.encoder_p6 = self.make_encoder_layer_with_pool(in_planes=1024, out_planes=1024, conv_kernel_size=3,
                                                            padding=1, pool_kernel_size=2)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.reshape(x.size(0), 1, x.size(-1))
        f1 = self.encoder_p1(x)
        f2 = self.encoder_p2(f1)
        f3 = self.encoder_p3(f2)
        f4 = self.encoder_p4(f3)
        f5 = self.encoder_p5(f4)
        f6 = self.encoder_p6(f5)
        return f6

    def make_encoder_layer_with_pool(self, in_planes, out_planes, conv_kernel_size, padding, pool_kernel_size):
        return nn.Sequential(
            nn.Conv1d(in_channels=in_planes, out_channels=out_planes, kernel_size=conv_kernel_size, padding=padding),
            nn.MaxPool1d(kernel_size=pool_kernel_size),
            nn.BatchNorm1d(num_features=out_planes),
            nn.ReLU(inplace=True)
        )

    def make_encoder_layer_without_pool(self, in_planes, out_planes, conv_kernel_size, padding):
        return nn.Sequential(
            nn.Conv1d(in_channels=in_planes, out_channels=out_planes, kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm1d(num_features=out_planes),
            nn.ReLU(inplace=True)
        )
