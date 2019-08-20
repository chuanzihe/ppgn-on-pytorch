# ref: https://github.com/yunjey/pytorch-tutorial/blob/master/
# tutorials/03-advanced/deep_convolutional_gan/main.py

import argparse
import os
# from solver import Solver
from sampler import Sampler
from data_loader import get_loader
from torch.backends import cudnn
import ipdb

def main(config):

    cudnn.benchmark = True    

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)
    
    # Train and sample the images
    if config.mode == 'train':

        data_loader = get_loader(image_path=config.image_path,
                                 image_size=config.image_size,
                                 batch_size=config.batch_size,
                                 num_workers=config.num_workers)        
        solver = Solver(config, data_loader)
        # solver.train()

    elif config.mode == 'sample':
        # ipdb.set_trace()
        samplr = Sampler(config)
        samplr.sampling() 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
        
    # model hyper-parameters
    parser.add_argument('--h_dim', type=int, default=4096)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    
    # training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=1) # default=20
    parser.add_argument('--batch_size', type=int, default=64) # default=64
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0002) # default 0.0002, 0.00005
    parser.add_argument('--beta1', type=float, default=0.9)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam
    parser.add_argument('--w_loss_img', type=float, default=2e-6) #2e-6 #1
    parser.add_argument('--w_loss_code', type=float, default=0.01) #0.01 #1
    parser.add_argument('--w_loss_dis', type=float, default=100) #100 #10

    # sampler 
    parser.add_argument('--caffenet', type=bool, default=False)
    parser.add_argument('--epsilon1', type=float, default=1e-5)
    parser.add_argument('--epsilon2', type=float, default=1)
    parser.add_argument('--epsilon3', type=float, default=1e-17)
    parser.add_argument('--enc_path', type=str, default='./models/caffenet/caffenet.pth')
    parser.add_argument('--gen_path', type=str, default='./models/5e-05-w1.0-1.0-100/G-50.pkl')
    parser.add_argument('--cond_path', type=str, default='./models/caffenet/caffenet.pth')

    parser.add_argument('--samp_lr', type=float, default=1)
    parser.add_argument('--samp_lr_end', type=float, default=1)
    parser.add_argument('--samp_xy', type=int, default=0)
    parser.add_argument('--samp_threshold', type=float, default=0)
    parser.add_argument('--samp_unit', type=str, default='945')
    parser.add_argument('--samp_iters', type=int, default=200) #default 200
    parser.add_argument('--samp_reset', type=int, default=0) #reset for diversity
    parser.add_argument('--samp_save', type=int, default=5) #save every



    # misc
    parser.add_argument('--image_size', type=int, default=227)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')
    parser.add_argument('--image_path', type=str, default='/home/chuanzih/pj/data/')
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--sample_step', type=int , default=500)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--model', type=str, default='noiseless')
    parser.add_argument('--synset_file', type=str, default='./misc/synset_words.txt')
    parser.add_argument('--vocab_file', type=str, default='./misc/vocabulary.txt')


    config = parser.parse_args()
    print(config)
    main(config)