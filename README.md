# Atrous Convolution and Residual Network (ARN)

## Introduce

1. This is net can be used in image restoration. It can make images with rain or fog be more clear.

2. The net named ARN, it means 'Atrous Convolution and Residual Network'. It had a group of Atrous Convolution and two blocks of Residual model.

3. In the val of dataset it can get this result in quantitative analysis.

   |      |  PSNR   |  SSIM  | iter_time |
   | :--: | :-----: | :----: | :-------: |
   | ARN  | 24.29dB | 85.88% |   5.7ms   |

4. These are qualitative analysis

![903_9](docs\903_9.jpg)

![903_9_result](docs\903_9_result.jpg)

![940_9](d ocs\940_9.jpg)

![940_9_result](docs\940_9_result.jpg)

![b4_00041](docs\b4_00041.jpg)

![b4_00041_result](docs\b4_00041_result.jpg)

## Environmental requirements

1. cuda 10.1 
2. cudnn 7.6.5
3. python=3.7
4. pytorch=1.4.0 
5. torchvision=0.5.0
6. scikit-image
7. numpy, scipy, tqdm, pillow, opencv (maybe loss something, this is my first time writing doc.)

## Model

**checkpoints/ARN_model.pth**

###### Training environment

|               |            Detail             |
| :-----------: | :---------------------------: |
|    System     |         Ubuntu 16.04          |
|      CPU      | Inter Xeon E5-2620 v3 2.40GHz |
|      RAM      |       32GB DDR4 2133MHz       |
|      GPU      |       NVIDIA GTX1080Ti        |
| Learning_rate |             0.007             |
|  Batch_size   |              64               |
|     Epoch     |              200              |

## Datasets

You can download from , 'https://pan.baidu.com/s/1Fdmc5Ua2su9o7rNv0R8UCQ 提取码：2333'(BaiDuYun)

Make these dataset to **'data/train/groundtruth'** and **'data/train/rain'**

The datasets come from 'https://github.com/hotndy/SPAC-SupplementaryMaterials' , 'Rain12600' and 'Rain1400'(https://xmu-smartdsp.github.io/),  and make some process get mine datasets.

## Demo

###### Demo.py

Make the model in to path **'checkpoints/model.pth'**

```python
python demo.py
```

## Train

###### Train.py

If use pre-trained model, make the true path of model.

```python
python train.py
```

## Val

###### val.py

Make the val datasets into true path

```python
python val.py
```

## Reference

[1] Ren D, Zuo W, Hu Q, et al. Progressive image deraining networks: A better and simpler baseline[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 3937-3946.

[2] Chen L C, Zhu Y, Papandreou G, et al. Encoder-decoder with atrous separable convolution for semantic image segmentation[C]//Proceedings of the European conference on computer vision (ECCV). 2018: 801-818.

