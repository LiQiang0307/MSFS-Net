'''
Descripttion: 
version: 
Author: LiQiang
Date: 2021-10-10 21:57:43
LastEditTime: 2022-08-11 13:40:58
'''
import os
import torch
from torchvision.transforms import functional as F
from utils import Adder
from data import test_dataloader
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import time

def deal_RealBlur(input_img):
    factor = 8
    h, w = input_img.shape[2], input_img.shape[3]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input_img = torch.nn.functional.pad(input_img, (0, padw, 0, padh), 'reflect')
    return input_img

def _eval(model, args):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    adder = Adder()
    model.eval()
    with torch.no_grad():
        psnr_adder = Adder()
        ssim_adder = Adder()

        # Hardware warm-up
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, _ = data
            h, w = input_img.shape[2], input_img.shape[3]

            input_img = deal_RealBlur(input_img)
            label_img = deal_RealBlur(label_img)

            input_img = input_img.to(device)
            label_img = label_img.to(device)
            tm = time.time()
            _ = model([input_img,label_img])
            _ = time.time() - tm

            if iter_idx == 20:
                break

        # Main Evaluation
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data
            h, w = input_img.shape[2], input_img.shape[3]

            input_img = deal_RealBlur(input_img)
            label_img = deal_RealBlur(label_img)

            input_img = input_img.to(device)
            label_img = label_img.to(device)

            tm = time.time()

            # 更改预测模型
            predict= model([input_img, label_img])[0]

            elapsed = time.time() - tm
            adder(elapsed)

            pred_clip = torch.clamp(predict, 0, 1)


            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            if args.save_image:
                save_name = os.path.join(args.result_dir, name[0])
                pred_clip = pred_clip[:, :, :h, :w]
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)
            psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            psnr_adder(psnr)

            print('%d iter PSNR: %.3f ' % (iter_idx + 1,psnr))

        print('==========================================================')
        print('The average PSNR is %.3f dB' % (psnr_adder.average()))
