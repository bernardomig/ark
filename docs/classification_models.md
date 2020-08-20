# Classification Models

## AlexNet

AlexNet implementation based on the paper [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).

| Name                 | # Params | Eval Size |  Top-1  | Top-5  | Weight                                                                        |
| -------------------- | -------: | --------- | :-----: | :----: | ----------------------------------------------------------------------------- |
| alexnet <sup>1</sup> |   61.101 | 224 @ 256 | 0.56134 | 0.7890 | [weight](https://files.deeplar.tk/ark-weights/alexnet-imagenet-30d827d4ff.pt) |

<sup>1</sup> Weights ported from [pytorch/vision](https://github.com/pytorch/vision).

## DenseNet

| Name                     | # Params | Eval Size |  Top-1  |  Top-5  | Weight                                                                            |
| ------------------------ | -------: | --------- | :-----: | :-----: | --------------------------------------------------------------------------------- |
| densenet121 <sup>1</sup> |    7.979 | 224 @ 256 | 0.74752 | 0.92152 | [weight](https://files.deeplar.tk/ark-weights/densenet121-imagenet-91fe98d048.pt) |
| densenet161 <sup>1</sup> |   28.681 | 224 @ 256 | 0.77348 | 0.93648 | [weight](https://files.deeplar.tk/ark-weights/densenet161-imagenet-87d5fabe27.pt) |
| densenet169 <sup>1</sup> |   14.149 | 224 @ 256 | 0.75912 | 0.93024 | [weight](https://files.deeplar.tk/ark-weights/densenet169-imagenet-30a50ee361.pt) |
| densenet201 <sup>1</sup> |   20.014 | 224 @ 256 | 0.77290 | 0.93478 | [weight](https://files.deeplar.tk/ark-weights/densenet201-imagenet-1ab44cb57e.pt) |

<sup>1</sup> Weights ported from [pytorch/vision](https://github.com/pytorch/vision).

## Dilated Residual Networks (DRNs)

| Name                   | # Params | Eval Size |  Top-1  |  Top-5  | Weight                                                                          |
| ---------------------- | -------: | --------- | :-----: | :-----: | ------------------------------------------------------------------------------- |
| drn_c_26 <sup>1</sup>  |   21.127 | 224 @ 256 | 0.74864 | 0.92252 | [weight](https://files.deeplar.tk/ark-weights/drn_c_26-imagenet-25b1fc3f2b.pt)  |
| drn_c_42 <sup>1</sup>  |   31.235 | 224 @ 256 | 0.76790 | 0.93308 | [weight](https://files.deeplar.tk/ark-weights/drn_c_42-imagenet-cb5cff5775.pt)  |
| drn_c_58 <sup>1</sup>  |   40.542 | 224 @ 256 | 0.78046 | 0.93848 | [weight](https://files.deeplar.tk/ark-weights/drn_c_58-imagenet-0bbc7fb2ed.pt)  |
| drn_d_22 <sup>1</sup>  |   16.394 | 224 @ 256 | 0.73918 | 0.91662 | [weight](https://files.deeplar.tk/ark-weights/drn_d_22-imagenet-784c51952f.pt)  |
| drn_d_38 <sup>1</sup>  |   26.502 | 224 @ 256 | 0.75844 | 0.92842 | [weight](https://files.deeplar.tk/ark-weights/drn_d_38-imagenet-39a7863b82.pt)  |
| drn_d_54 <sup>1</sup>  |   35.809 | 224 @ 256 | 0.78558 | 0.94016 | [weight](https://files.deeplar.tk/ark-weights/drn_d_54-imagenet-8c83e6312c.pt)  |
| drn_d_105 <sup>1</sup> |   54.801 | 224 @ 256 | 0.79228 | 0.94312 | [weight](https://files.deeplar.tk/ark-weights/drn_d_105-imagenet-a27297fdc8.pt) |

<sup>1</sup> Weights ported from [fyu/drn](https://github.com/fyu/drn).

## EfficientNet

| Name                         | # Params | Eval Size |  Top-1  |  Top-5  | Weight                                                                                |
| ---------------------------- | -------: | --------- | :-----: | :-----: | ------------------------------------------------------------------------------------- |
| efficientnet_b0 <sup>1</sup> |    5.289 | 224 @ 256 | 0.77692 | 0.93532 | [weight](https://files.deeplar.tk/ark-weights/efficientnet_b0-imagenet-b28f86c117.pt) |
| efficientnet_b1 <sup>1</sup> |    7.794 | 240 @ 272 | 0.78692 | 0.94086 | [weight](https://files.deeplar.tk/ark-weights/efficientnet_b1-imagenet-7eb107fee3.pt) |
| efficientnet_b2 <sup>1</sup> |    9.110 | 260 @ 292 | 0.80294 | 0.95168 | [weight](https://files.deeplar.tk/ark-weights/efficientnet_b2-imagenet-03de86b71c.pt) |
| efficientnet_b3 <sup>1</sup> |   12.233 | 300 @ 332 | 0.81516 | 0.95672 | [weight](https://files.deeplar.tk/ark-weights/efficientnet_b3-imagenet-4105432b3b.pt) |

<sup>1</sup> Weights ported from [rwightman/gen-efficientnet-pytorch](https://github.com/rwightman/gen-efficientnet-pytorch/).

## GhostNet

| Name                      | # Params | Eval Size | Top-1  | Top-5  | Weight                                                                             |
| ------------------------- | -------: | --------- | :----: | :----: | ---------------------------------------------------------------------------------- |
| ghostnet_1_0 <sup>1</sup> |    5.183 | 224 @ 256 | 0.7396 | 0.9153 | [weight](https://files.deeplar.tk/ark-weights/ghostnet_1_0-imagenet-9364bb1a07.pt) |

<sup>1</sup> Weights from [huawei-noah/ghostnet](https://github.com/huawei-noah/ghostnet).

## MixNets

| Name     | # Params | Eval Size | Top-1 | Top-5 | Weight |
| -------- | -------: | --------- | :---: | :---: | ------ |
| mixnet_s |    4.135 | 224 @ 256 |       |       |        |
| mixnet_m |    5.014 | 224 @ 256 |       |       |        |
| mixnet_l |    7.329 | 224 @ 256 |       |       |        |

## MobileNetV2

| Name                          | # Params | Eval Size |  Top-1  |  Top-5  | Weight                                                                                 |
| ----------------------------- | -------- | --------- | :-----: | :-----: | -------------------------------------------------------------------------------------- |
| mobilenetv2_1_0 <sup>1</sup>  | 3.505    | 224 @ 256 | 0.72376 | 0.90836 | [weight](https://files.deeplar.tk/ark-weights/mobilenetv2_1_0-imagenet-bf3dd96df3.pt)  |
| mobilenetv2_0_75 <sup>1</sup> | 2.636    | 224 @ 256 | 0.69610 | 0.88812 | [weight](https://files.deeplar.tk/ark-weights/mobilenetv2_0_75-imagenet-34c9a7dc0c.pt) |
| mobilenetv2_0_50 <sup>1</sup> | 1.969    | 224 @ 256 | 0.64306 | 0.85178 | [weight](https://files.deeplar.tk/ark-weights/mobilenetv2_0_50-imagenet-228a3ef7f5.pt) |
| mobilenetv2_0_35 <sup>1</sup> | 1.677    | 224 @ 256 | 0.59944 | 0.82040 | [weight](https://files.deeplar.tk/ark-weights/mobilenetv2_0_35-imagenet-2aafa73305.pt) |
| mobilenetv2_0_25 <sup>1</sup> | 1.519    | 224 @ 256 | 0.51874 | 0.75618 | [weight](https://files.deeplar.tk/ark-weights/mobilenetv2_0_25-imagenet-1657878274.pt) |
| mobilenetv2_0_10 <sup>1</sup> | 1.357    | 224 @ 256 | 0.34598 | 0.58422 | [weight](https://files.deeplar.tk/ark-weights/mobilenetv2_0_10-imagenet-00c0863e22.pt) |

<sup>1</sup> Weights from [d-li14/mobilenetv2.pytorch](https://github.com/d-li14/mobilenetv2.pytorch).

## MobileNetV3

| Name                                | # Params | Eval Size |  Top-1  |  Top-5  | Weight                                                                                       |
| ----------------------------------- | -------- | --------- | :-----: | :-----: | -------------------------------------------------------------------------------------------- |
| mobilenetv3_large_1_0 <sup>1</sup>  | 5.483    | 224 @ 256 | 0.74034 | 0.91824 | [weight](https://files.deeplar.tk/ark-weights/mobilenetv3_large_1_0-imagenet-8ceb1626c7.pt)  |
| mobilenetv3_large_0_75 <sup>1</sup> | 3.994    | 224 @ 256 | 0.72504 | 0.90688 | [weight](https://files.deeplar.tk/ark-weights/mobilenetv3_large_0_75-imagenet-e8aac9762b.pt) |
| mobilenetv3_small_1_0 <sup>1</sup>  | 2.543    | 224 @ 256 | 0.66828 | 0.87090 | [weight](https://files.deeplar.tk/ark-weights/mobilenetv3_small_1_0-imagenet-a830b03e78.pt)  |
| mobilenetv3_small_0_75 <sup>1</sup> | 2.042    | 224 @ 256 | 0.64570 | 0.85328 | [weight](https://files.deeplar.tk/ark-weights/mobilenetv3_small_0_75-imagenet-295a26ddbc.pt) |

<sup>1</sup> Weights from [d-li14/mobilenetv2.pytorch](https://github.com/d-li14/mobilenetv3.pytorch).

## RegNet

| Name                     | # Params | Eval Size |  Top-1  |  Top-5  | Weight                                                                            |
| ------------------------ | -------: | --------- | :-----: | :-----: | --------------------------------------------------------------------------------- |
| regnetx_002 <sup>1</sup> |    2.685 | 224 @ 256 | 0.67934 | 0.88102 | [weight](https://files.deeplar.tk/ark-weights/regnetx_002-imagenet-dc40ef1c36.pt) |
| regnetx_004 <sup>1</sup> |    5.158 | 224 @ 256 | 0.71878 | 0.90484 | [weight](https://files.deeplar.tk/ark-weights/regnetx_004-imagenet-f2708d7db0.pt) |
| regnetx_006 <sup>1</sup> |    6.196 | 224 @ 256 | 0.73862 | 0.91680 | [weight](https://files.deeplar.tk/ark-weights/regnetx_006-imagenet-e4462f6128.pt) |
| regnetx_008 <sup>1</sup> |    7.260 | 224 @ 256 | 0.74656 | 0.92074 | [weight](https://files.deeplar.tk/ark-weights/regnetx_008-imagenet-e0d97cccbf.pt) |
| regnetx_016 <sup>1</sup> |    9.190 | 224 @ 256 | 0.76360 | 0.93156 | [weight](https://files.deeplar.tk/ark-weights/regnetx_016-imagenet-bc694a4fb3.pt) |
| regnetx_032 <sup>1</sup> |   15.297 | 224 @ 256 | 0.77928 | 0.93944 | [weight](https://files.deeplar.tk/ark-weights/regnetx_032-imagenet-897d6a187c.pt) |
| regnetx_040 <sup>1</sup> |   22.118 | 224 @ 256 | 0.78486 | 0.94242 | [weight](https://files.deeplar.tk/ark-weights/regnetx_040-imagenet-f5bc702a82.pt) |
| regnetx_064 <sup>1</sup> |   26.209 | 224 @ 256 | 0.78826 | 0.94482 | [weight](https://files.deeplar.tk/ark-weights/regnetx_064-imagenet-14880ce230.pt) |
| regnetx_080 <sup>1</sup> |   39.573 | 224 @ 256 | 0.79198 | 0.94558 | [weight](https://files.deeplar.tk/ark-weights/regnetx_080-imagenet-c1ea72c75e.pt) |
| regnetx_120 <sup>1</sup> |   46.106 | 224 @ 256 | 0.79754 | 0.94852 | [weight](https://files.deeplar.tk/ark-weights/regnetx_120-imagenet-1d9a4fb38e.pt) |
| regnetx_160 <sup>1</sup> |   54.279 | 224 @ 256 | 0.80016 | 0.94914 | [weight](https://files.deeplar.tk/ark-weights/regnetx_160-imagenet-27aec19f17.pt) |
| regnetx_320 <sup>1</sup> |  107.812 | 224 @ 256 | 0.80302 | 0.95042 | [weight](https://files.deeplar.tk/ark-weights/regnetx_320-imagenet-498fddcb01.pt) |
| regnety_002 <sup>1</sup> |    3.163 | 224 @ 256 | 0.70282 | 0.89540 | [weight](https://files.deeplar.tk/ark-weights/regnety_002-imagenet-1febafaf91.pt) |
| regnety_004 <sup>1</sup> |    4.344 | 224 @ 256 | 0.74026 | 0.91748 | [weight](https://files.deeplar.tk/ark-weights/regnety_004-imagenet-d52958c3f5.pt) |
| regnety_006 <sup>1</sup> |    6.055 | 224 @ 256 | 0.75260 | 0.92528 | [weight](https://files.deeplar.tk/ark-weights/regnety_006-imagenet-aec4b5587e.pt) |
| regnety_008 <sup>1</sup> |    6.263 | 224 @ 256 | 0.76314 | 0.93062 | [weight](https://files.deeplar.tk/ark-weights/regnety_008-imagenet-8e1cc4c9d6.pt) |
| regnety_016 <sup>1</sup> |   11.202 | 224 @ 256 | 0.77852 | 0.93716 | [weight](https://files.deeplar.tk/ark-weights/regnety_016-imagenet-0a5a3b2871.pt) |
| regnety_032 <sup>1</sup> |   19.436 | 224 @ 256 | 0.78870 | 0.94402 | [weight](https://files.deeplar.tk/ark-weights/regnety_032-imagenet-86e0086c3c.pt) |
| regnety_040 <sup>1</sup> |   20.647 | 224 @ 256 | 0.79222 | 0.94656 | [weight](https://files.deeplar.tk/ark-weights/regnety_040-imagenet-c9675e1d87.pt) |
| regnety_064 <sup>1</sup> |   30.583 | 224 @ 256 | 0.79712 | 0.94774 | [weight](https://files.deeplar.tk/ark-weights/regnety_064-imagenet-547954424d.pt) |
| regnety_080 <sup>1</sup> |   39.180 | 224 @ 256 | 0.79868 | 0.94832 | [weight](https://files.deeplar.tk/ark-weights/regnety_080-imagenet-ed7c3ea940.pt) |
| regnety_120 <sup>1</sup> |   51.823 | 224 @ 256 | 0.80382 | 0.95128 | [weight](https://files.deeplar.tk/ark-weights/regnety_120-imagenet-5bbf19ecaa.pt) |
| regnety_160 <sup>1</sup> |   83.590 | 224 @ 256 | 0.8030  | 0.94962 | [weight](https://files.deeplar.tk/ark-weights/regnety_160-imagenet-fab6e94e80.pt) |
| regnety_320 <sup>1</sup> |  145.047 | 224 @ 256 | 0.80814 | 0.95240 | [weight](https://files.deeplar.tk/ark-weights/regnety_320-imagenet-9bce7d869e.pt) |

<sup>1</sup> Weights from [facebookresearch/pycls](https://github.com/facebookresearch/pycls).

## ResNest

| Name                    | # Params | Eval Size |  Top-1  |  Top-5  | Weight                                                                           |
| ----------------------- | -------: | --------- | :-----: | :-----: | -------------------------------------------------------------------------------- |
| resnest50 <sup>1</sup>  |   27.483 | 224 @ 256 | 0.8111  | 0.95362 | [weight](https://files.deeplar.tk/ark-weights/resnest50-imagenet-47459d8454.pt)  |
| resnest101 <sup>1</sup> |   48.275 | 256 @ 292 | 0.82792 | 0.9629  | [weight](https://files.deeplar.tk/ark-weights/resnest101-imagenet-7f4e6529c9.pt) |
| resnest200 <sup>1</sup> |   70.202 | 320 @ 366 | 0.83778 | 0.96824 | [weight](https://files.deeplar.tk/ark-weights/resnest200-imagenet-408422922a.pt) |
| resnest269 <sup>1</sup> |  110.929 | 416 @ 476 | 0.84342 | 0.96838 | [weight](https://files.deeplar.tk/ark-weights/resnest269-imagenet-3183fb2545.pt) |

<sup>1</sup> Weights ported from [zhanghang1989/ResNeSt](https://github.com/zhanghang1989/ResNeSt)

## ResNet

| Name                   | # Params | Eval Size |  Top-1  |  Top-5  | Weight                                                                          |
| ---------------------- | -------: | --------- | :-----: | :-----: | ------------------------------------------------------------------------------- |
| resnet18 <sup>1</sup>  |   11.690 | 224 @ 256 | 0.69546 | 0.89088 | [weight](https://files.deeplar.tk/ark-weights/resnet18-imagenet-8d64fdf20f.pt)  |
| resnet34 <sup>1</sup>  |   21.798 | 224 @ 256 | 0.73202 | 0.91336 | [weight](https://files.deeplar.tk/ark-weights/resnet34-imagenet-0d9c6a03d7.pt)  |
| resnet50 <sup>2</sup>  |   25.557 | 224 @ 256 | 0.76786 | 0.93272 | [weight](https://files.deeplar.tk/ark-weights/resnet50-imagenet-c0d2bddaf7.pt)  |
| resnet101 <sup>2</sup> |   44.549 | 224 @ 256 | 0.78408 | 0.94102 | [weight](https://files.deeplar.tk/ark-weights/resnet101-imagenet-cf7a8d28f0.pt) |
| resnet152 <sup>2</sup> |   60.193 | 224 @ 256 | 0.79148 | 0.94404 | [weight](https://files.deeplar.tk/ark-weights/resnet152-imagenet-d3e4a0eff4.pt) |

<sup>1</sup> Weights ported from [pytorch/vision](https://github.com/pytorch/vision).
<sup>2</sup> Weights from [facebookresearch/pycls](https://github.com/facebookresearch/pycls).

## ResNext

| Name                         | # Params | Eval Size |  Top-1  |  Top-5  | Weight                                                                                |
| ---------------------------- | -------: | --------- | :-----: | :-----: | ------------------------------------------------------------------------------------- |
| resnext50_32x4 <sup>2</sup>  |   25.029 | 224 @ 256 | 0.78148 | 0.93824 | [weight](https://files.deeplar.tk/ark-weights/resnext50_32x4-imagenet-3c9f5bcfb5.pt)  |
| resnext101_32x4 <sup>2</sup> |   44.178 | 224 @ 256 | 0.79208 | 0.94466 | [weight](https://files.deeplar.tk/ark-weights/resnext101_32x4-imagenet-4d1f34007f.pt) |
| resnext101_32x8 <sup>1</sup> |   88.791 | 224 @ 256 | 0.79052 | 0.94424 | [weight](https://files.deeplar.tk/ark-weights/resnext101_32x8-imagenet-464ed2f66d.pt) |
| resnext152_32x4 <sup>2</sup> |   59.951 | 224 @ 256 | 0.79616 | 0.94612 | [weight](https://files.deeplar.tk/ark-weights/resnext152_32x4-imagenet-26b33c40d2.pt) |

<sup>1</sup> Weights ported from [pytorch/vision](https://github.com/pytorch/vision).
<sup>2</sup> Weights from [facebookresearch/pycls](https://github.com/facebookresearch/pycls).

## SqueezeNetV1

| Name                      | Eval Size |  Top-1  |  Top-5  | Weight                                                                           |
| ------------------------- | --------- | :-----: | :-----: | -------------------------------------------------------------------------------- |
| squeezenetv1 <sup>1</sup> | 224 @ 256 | 0.57872 | 0.80422 | [weight](https://files.deeplar.tk/ark-weights/squeezenet-imagenet-ed8e93b737.pt) |

<sup>1</sup> Weights ported from [pytorch/vision](https://github.com/pytorch/vision).

## VGG

| Name               | # Params | Eval Size |  Top-1  |  Top-5  | Weight                                                                      |
| ------------------ | -------: | --------- | :-----: | :-----: | --------------------------------------------------------------------------- |
| vgg11 <sup>1</sup> |  132.866 | 224 @ 256 | 0.70234 | 0.89684 | [weight](https://files.deeplar.tk/ark-weights/vgg11-imagenet-3ba49f9647.pt) |
| vgg13 <sup>1</sup> |  133.051 | 224 @ 256 | 0.71380 | 0.90300 | [weight](https://files.deeplar.tk/ark-weights/vgg13-imagenet-37b6a8b641.pt) |
| vgg16 <sup>1</sup> |  138.362 | 224 @ 256 | 0.73052 | 0.91376 | [weight](https://files.deeplar.tk/ark-weights/vgg16-imagenet-b546252746.pt) |
| vgg19 <sup>1</sup> |  143.673 | 224 @ 256 | 0.74000 | 0.91720 | [weight](https://files.deeplar.tk/ark-weights/vgg19-imagenet-6ff0e0ad02.pt) |

<sup>1</sup> Weights ported from [pytorch/vision](https://github.com/pytorch/vision).

## WRN

| Name                    | # Params | Eval Size |  Top-1  |  Top-5  | Weight                                                                           |
| ----------------------- | -------: | --------- | :-----: | :-----: | -------------------------------------------------------------------------------- |
| wrn50_2_0 <sup>1</sup>  |   68.883 | 224 @ 256 | 0.78292 | 0.94026 | [weight](https://files.deeplar.tk/ark-weights/wrn50_2_0-imagenet-58c3842933.pt)  |
| wrn101_2_0 <sup>1</sup> |  126.887 | 224 @ 256 | 0.7862  | 0.9413  | [weight](https://files.deeplar.tk/ark-weights/wrn101_2_0-imagenet-e4c5597950.pt) |

<sup>1</sup> Weights ported from [pytorch/vision](https://github.com/pytorch/vision).
