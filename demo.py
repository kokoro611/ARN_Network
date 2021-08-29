import torch
from net.NetModel import Net
from net.NetModel_mine import Net_mine
import time
from torchvision.utils import save_image
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2

cuda = False
mode = 'img'  # img or video
model_path = './checkpoints/ARN_model.pth'  # model

if cuda:
    net = Net_mine().cuda()
else:
    net = Net_mine()
print('model loading')
net.load_state_dict(torch.load(model_path))  # 加载训练好的模型参数
print('model loaded')

if mode == 'img':
    file_path = 'data/test/rb1_Rain/'  # the path of img_file path
    img_names = os.listdir(file_path)
    pbar = tqdm(total=len(img_names))
    for img_name in img_names:

        pbar.update(1)

        img_path = file_path + '/' + img_name
        image = Image.open(img_path).convert('RGB')
        image = image.convert('RGB')

        image = np.array(image, dtype=np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        images = [image]
        images = torch.from_numpy(np.asarray(images))

        if cuda:
            input = images.cuda()
        else:
            input = images
        with torch.no_grad():
            output_image = net(input)  # 输出tensor
            save_image(output_image, 'result/' + img_name.replace('.jpg', '') + '_result' + '.jpg')  # 直接保存张量图片，自动转换
    pbar.close()

if mode == 'video':
    fps = 1.0
    # use camera or use video
    capture = cv2.VideoCapture(0)
    capture = cv2.VideoCapture('data/test/video/testb1.mp4')
    fps_origin = capture.get(cv2.CAP_PROP_FPS)
    frame_h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_write = cv2.VideoWriter('result/outb3.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps_origin,
                                  (int(frame_w), int(frame_h)))
    while (True):
        t1 = time.time()
        # read one frame
        ref, frame = capture.read()
        cv2.imshow('o', frame)
        cv2.imwrite('temp/1.jpg', frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(np.uint8(frame))

        # frame = Image.open('temp/1.jpg').convert('RGB')
        frame = frame.convert('RGB')

        # frame.show()

        frame = np.array(frame, dtype=np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        #   添加上batch_size维度
        frame = [frame]
        frame = torch.from_numpy(np.asarray(frame))
        if cuda:
            frame = frame.cuda()
        else:
            frame = frame
        with torch.no_grad():
            frame_out = net(frame)

        save_image(frame_out, 'temp/1.jpg')
        '''
        frame_out = frame_out.cpu().clone()
        unloader = transforms.ToPILImage()
        frame_out = frame_out.squeeze(0)
        frame_out = unloader(frame_out)


        frame_out = np.array(frame_out)
        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)
        '''

        frame_out = cv2.imread('temp/1.jpg')
        fps = (fps + (1. / (time.time() - t1))) / 2
        print(fps)
        frame_out = cv2.putText(frame_out, 'fps=%.2f' % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('video', frame_out)
        video_write.write(frame_out)
        c = cv2.waitKey(1) & 0xff

        if c == 27:
            capture.release()
            break
