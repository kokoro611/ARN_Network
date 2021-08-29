import torch
from net.NetModel import Net
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from net.NetModel_mine import Net_mine


def get_ssim(img1, img2):
    '''
    img1
        图像1
    img2_
        图像2
    Returns
    ssim_score : numpy.float64
        结构相似性指数（structural similarity index，SSIM）.
    References
    https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html
    '''
    img2 = img2.resize(img1.size)
    img1, img2 = np.array(img1), np.array(img2)
    # 此处因为转换为灰度值之后的图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
    ssim_score = ssim(img1, img2, data_range=255, multichannel=True)
    return ssim_score


def get_psnr(img1, img2):
    '''
    img1
        图像1
    img2
        图像2
    Returns
    psnr_score : numpy.float64
        峰值信噪比(Peak Signal to Noise Ratio, PSNR).
    References
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    '''
    img2 = img2.resize(img1.size)
    img1, img2 = np.array(img1), np.array(img2)
    # 此处的第一张图片为真实图像，第二张图片为测试图片
    # 此处因为图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
    psnr_score = psnr(img1, img2, data_range=255)
    return psnr_score


if __name__ == "__main__":

    cuda = True
    psnr_list = []
    ssim_list = []

    if cuda:
        net = Net_mine().cuda()
    else:
        net = Net_mine()
    print('model loading')
    net.load_state_dict(torch.load('./checkpoints/ARN_model.pth'))  # 加载训练好的模型参数
    print('model loaded')

    file_path = 'data/val/rain/'
    label_path = 'data/val/groundtruth/'

    #file_path = 'rainy_image_dataset/testing/rainy_image/'
    #label_path = 'rainy_image_dataset/testing/ground_truth/'

    img_names = os.listdir(file_path)
    pbar = tqdm(total=len(img_names))
    for img_name in img_names:
        pbar.update(1)
        img_path = file_path + '/' + img_name
        image = Image.open(img_path).convert('RGB')
        image = image.convert('RGB')
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        #   添加上batch_size维度
        images = [image]
        images = torch.from_numpy(np.asarray(images))
        if cuda:
            input = images.cuda()
        else:
            input = images
        with torch.no_grad():
            img_tensor = net(input)  # 输出的是张量

            unloader = transforms.ToPILImage()
            img_out = img_tensor.cpu().clone()
            img_out = img_out.squeeze(0)
            img_out = unloader(img_out)

        label_img_path = label_path + '/' + img_name
        label = Image.open(label_img_path).convert('RGB')

        # print(img_out.size,label.size)
        psnr_list.append(get_psnr(img_out, label))
        ssim_list.append(get_ssim(img_out, label))

    pbar.close()

    psnr_out = np.mean(psnr_list)  # more biger more better
    ssim_out = np.mean(ssim_list)  # more nearer 1 more better

    print('PSNR={:.4f}, SSIM={:.4f}'.format(psnr_out, ssim_out))
