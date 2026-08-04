import torch
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a, resnet34_ibn_a, resnet18_ibn_a


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    def __init__(self, last_stride, neck, neck_feat, model_name, feature_dim=2048):
        super(Baseline, self).__init__()
        self.in_planes = feature_dim
        if model_name == "resnet18":
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, block=BasicBlock, layers=[2, 2, 2, 2])
        elif model_name == "resnet34":
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, block=BasicBlock, layers=[3, 4, 6, 3])
        elif model_name == "resnet50":
            self.base = ResNet(last_stride=last_stride, block=Bottleneck, layers=[3, 4, 6, 3])
        elif model_name == "resnet101":
            self.base = ResNet(last_stride=last_stride, block=Bottleneck, layers=[3, 4, 23, 3])
        elif model_name == "resnet152":
            self.base = ResNet(last_stride=last_stride, block=Bottleneck, layers=[3, 8, 36, 3])

        elif model_name == "se_resnet50":
            self.base = SENet(
                block=SEResNetBottleneck,
                layers=[3, 4, 6, 3],
                groups=1,
                reduction=16,
                dropout_p=None,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                last_stride=last_stride,
            )
        elif model_name == "se_resnet101":
            self.base = SENet(
                block=SEResNetBottleneck,
                layers=[3, 4, 23, 3],
                groups=1,
                reduction=16,
                dropout_p=None,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                last_stride=last_stride,
            )
        elif model_name == "se_resnet152":
            self.base = SENet(
                block=SEResNetBottleneck,
                layers=[3, 8, 36, 3],
                groups=1,
                reduction=16,
                dropout_p=None,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                last_stride=last_stride,
            )
        elif model_name == "se_resnext50":
            self.base = SENet(
                block=SEResNeXtBottleneck,
                layers=[3, 4, 6, 3],
                groups=32,
                reduction=16,
                dropout_p=None,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                last_stride=last_stride,
            )
        elif model_name == "se_resnext101":
            self.base = SENet(
                block=SEResNeXtBottleneck,
                layers=[3, 4, 23, 3],
                groups=32,
                reduction=16,
                dropout_p=None,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                last_stride=last_stride,
            )
        elif model_name == "senet154":
            self.base = SENet(
                block=SEBottleneck,
                layers=[3, 8, 36, 3],
                groups=64,
                reduction=16,
                dropout_p=0.2,
                last_stride=last_stride,
            )
        elif model_name == "resnet50_ibn_a":
            self.base = resnet50_ibn_a(last_stride, feature_dim)
        elif model_name == "resnet34_ibn_a":
            self.base = resnet34_ibn_a(last_stride, feature_dim)
            # self.in_planes = 512
        elif model_name == "resnet18_ibn_a":
            self.base = resnet18_ibn_a(last_stride, feature_dim)
            # self.in_planes = 512

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == "no":
            pass
            # self.classifier = nn.Linear(self.in_planes, self.num_classes)
        elif self.neck == "bnneck":
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            # self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feat, global_feat = self._extract_features(x)
        if self.neck_feat == "after":
            return feat
        else:
            return global_feat

    def _extract_features(self, x):
        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == "bnneck":
            feat = self.bottleneck(global_feat)  # normalize for angular softmax
        else:  # self.neck == "no"
            feat = global_feat
        return feat, global_feat
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if "classifier" in i:
                continue
            if "fc" in i and i not in self.state_dict():
                continue
            # print(f" i: {i} param dict: {param_dict[i].shape}, state_dict: {self.state_dict()[i].shape}")
            self.state_dict()[i].copy_(param_dict[i])
            # try:
            #     self.state_dict()[i].copy_(param_dict[i])
            # except:
            #     raise ValueError


class BaselineWithClassifier(Baseline):

    def __init__(
            self,
            last_stride,
            neck,
            neck_feat,
            model_name,
            feature_dim,
            attr_sizes,
    ):
        super(BaselineWithClassifier, self).__init__(last_stride, neck, neck_feat, model_name, feature_dim)

        self.outputs_fc = nn.ModuleList()
        for n in attr_sizes:
            self.outputs_fc.append(
                nn.Sequential(
                    nn.Linear(feature_dim, feature_dim),
                    nn.ReLU(),
                    nn.Linear(feature_dim, n),
                )
            )

    def forward(self, x):
        feat, global_feat = self._extract_features(x)
        outputs = []
        for fc in self.outputs_fc:
            outputs.append(fc(feat))

        if self.neck_feat == "after":
            return feat, outputs
        else:
            return global_feat, outputs

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)

        if (
            "model" in param_dict.keys()
        ):  #   if the param dict is the run and not the model
            param_dict = param_dict["model"].state_dict()

        for i in param_dict:
            if "classifier" in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])