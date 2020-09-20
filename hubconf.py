from ark.models.classification.alexnet import alexnet
from ark.models.classification.densenet import (
    densenet121, densenet161, densenet169, densenet201)
from ark.models.classification.drn import (
    drn_c_26, drn_c_42, drn_c_58,
    drn_d_22, drn_d_38, drn_d_54, drn_d_105)
from ark.models.classification.ghostnet import ghostnet_1_0
from ark.models.classification.mixnet import (
    mixnet_s, mixnet_m, mixnet_l)
from ark.models.classification.mobilenetv2 import (
    mobilenetv2_1_0, mobilenetv2_0_75, mobilenetv2_0_50,
    mobilenetv2_0_35, mobilenetv2_0_25, mobilenetv2_0_10)
from ark.models.classification.mobilenetv3 import (
    mobilenetv3_large_1_0, mobilenetv3_large_0_75,
    mobilenetv3_small_1_0, mobilenetv3_small_0_75)
from ark.models.classification.regnet import (
    regnetx_002, regnetx_004, regnetx_006, regnetx_008,
    regnetx_016, regnetx_032, regnetx_040, regnetx_064,
    regnetx_080, regnetx_120, regnetx_160, regnetx_320,
    regnety_002, regnety_004, regnety_006, regnety_008,
    regnety_016, regnety_032, regnety_040, regnety_064,
    regnety_080, regnety_120, regnety_160, regnety_320)
from ark.models.classification.resnest import (
    resnest50, resnest101, resnest200, resnest269)
from ark.models.classification.resnet import (
    resnet18, resnet34, resnet50, resnet101, resnet152)
from ark.models.classification.resnext import (
    resnext50_32x4, resnext101_32x4,
    resnext101_32x8, resnext152_32x4)
from ark.models.classification.shufflenetv2 import (
    shufflenetv2_0_5, shufflenetv2_1_0,
    shufflenetv2_1_5, shufflenetv2_2_0)
from ark.models.classification.squeezenet import squeezenet
from ark.models.classification.vgg import (
    vgg11, vgg13, vgg16, vgg19)
from ark.models.classification.wrn import wrn50_2_0, wrn101_2_0

from ark.models.segmentation.bisenetv2 import bisenetv2
from ark.models.segmentation.cgnet import cgnet
from ark.models.segmentation.contextnet import contextnet12, contextnet14, contextnet18
from ark.models.segmentation.enet import enet
from ark.models.segmentation.erfnet import erfnet
from ark.models.segmentation.esnet import esnet
from ark.models.segmentation.espnet import espnet, espnet_c
from ark.models.segmentation.fastscnn import fastscnn
from ark.models.segmentation.lednet import lednet
from ark.models.segmentation.lwrefinenet import (
    lwrefinenet_resnet50, lwrefinenet_resnet101, lwrefinenet_resnet152)
from ark.models.segmentation.pspnet import (
    pspnet_resnet18, pspnet_resnet34, pspnet_resnet50, pspnet_resnet101)
from ark.models.segmentation.segnet import segnet

dependencies = ['torch']
