'''
Descripttion: 
version: 
Author: LiQiang
Date: 2021-10-10 21:53:03
LastEditTime: 2022-08-11 12:45:07
'''
import os
# windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import argparse
from torch.backends import cudnn
from models.freNet import make_model as build_net
from eval_R import _eval
from models.local_arch import make_model as build_arch_net

def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    #model = build_net()
    model=build_arch_net()
    # print(model)
    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        pass
    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='RealBlur-J-Gopro+local', type=str)
    parser.add_argument('--data_dir', type=str, default='/liq_AI/zhangyn/RealBlur-J')
    parser.add_argument('--mode', default='test',
                        choices=['train', 'test'], type=str)
    # Train
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--num_epoch', type=int, default=3000)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--valid_freq', type=int, default=20)
    parser.add_argument('--resume', type=str,
                        default='/home/liqiang/桌面/mobile/实验/model.pkl')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr_steps', type=list,
                        default=[(x+1) * 500 for x in range(3000//500)])
    # Test
    parser.add_argument('--test_model', type=str,
                        default='model.pkl')
    parser.add_argument('--save_image', type=bool,
                        default=True, choices=[True, False])

    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', args.model_name, 'weights/')
    args.result_dir = os.path.join(
        'results/', args.model_name, 'result_image/')
    print(args)
    main(args)
