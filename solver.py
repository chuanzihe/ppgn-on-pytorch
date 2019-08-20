import os
import time
import torch
import torchvision
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from model import EncPPGN
from model import GenPPGN
from model import DisPPGN
import ipdb

class Solver():
    def __init__(self, config, data_loader):

        self.G = None
        self.D = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.h_dim = config.h_dim
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.data_loader = data_loader
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.sample_size = config.sample_size
        self.lr = config.lr
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.sample_path = config.sample_path
        self.model_path = config.model_path
        self.model = config.model
        self.enc_path = config.enc_path

        # PPGN loss weight  
        self.w_loss_img  = config.w_loss_img # img_recon_los G
        self.w_loss_code = config.w_loss_code # feat_recon_loss E
        self.w_loss_dis = config.w_loss_dis  # softmax loss D        

        # PPGN-caffe default
        self.max_iter = 1300000 
        self.display_every = 50 
        self.snapshot_every = 1000 
        self.snapshot_folder = 'snapshots'
        
        # @todo(chuanzi): 
        # PPGN: we store a bunch of old generated images and feed them to 
        # the discriminator so that it does not overfit to the current data
        self.use_buffer_from = -1
        self.feat_shape = (4096,)
        self.comparator_feat_shape = (256,6,6)
        self.im_size = (3,227,227)
        self.snapshot_at_iter = -1
        self.snapshot_at_iter_file = 'snapshot_at_iter.txt'
        self.resume = config.resume

        self.display = 0
        self.base_lr = 0.00005
        self.weight_decay = 0.0004
        self.lr_policy = "multistep"
        self.gamma = 0.5
        self.D_iter = 0
        self.G_iter = 0
        # self.epoch = 1300000/batch

        # training flags
        self.train_D = True
        self.train_G = True

        self.build_model()
    
    def build_model(self):

        self.E = EncPPGN()
        self.G = GenPPGN()
        self.D = DisPPGN()

        self.G_optimizer = optim.Adam(self.G.parameters(),
                                      self.lr, [self.beta1, self.beta2])
        self.D_optimizer = optim.Adam(self.D.parameters(),
                                      self.lr, [self.beta1, self.beta2])
        
        if torch.cuda.is_available():
            self.G.cuda()
            self.D.cuda()
            self.E.cuda()
        
    def to_variable(self, x, grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, requires_grad=grad)
    
    def to_data(self, x):
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data
    
    def denorm(self, x):
        """Convert range (-1, 1) to (0, 1)"""
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def resume(self):     
        if os.path.isfile(self.resume):
            print("=> loading checkpoint '{}'".format(self.resume))
            checkpoint = torch.load(self.resume)
            self.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(self.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(self.resume))   

    def train_mode(self, loss_D, loss_G):
        """switch training mode of D and G to avoid overfiting"""
        """epslon = 1e-5 to avoid loss = 0 """
        """#@(chuanzi):  not loss = 0 in original occasion"""
        ratio = loss_D.data[0]/(loss_G.data[0] + 1e-5)
        if ratio < 1e-1 and self.train_D:
            self.train_D = False
            self.train_G = True
        if ratio > 5e-1 and not self.train_D:
            self.train_D = True
            self.train_G = True
        if ratio > 1e-1 and self.train_G:
            self.train_G = False
            self.train_D = True
        print ( "train_D=%d, train_G=%d"  % (self.train_D, self.train_G))

    def clean_file(self):
        """delete previous model"""
        final_model = os.path.join(self.model_path, 'G-%d.pkl' %(self.num_epochs))
        if os.path.isfile(final_model):
            for epoch in range(1,self.num_epochs):
                old_g_path = os.path.join(self.model_path, 'G-%d.pkl' %(epoch))
                old_d_path = os.path.join(self.model_path, 'D-%d.pkl' %(epoch))
                os.system('rm %s' % (old_g_path))
                os.system('rm %s' % (old_d_path))

            folder = '%s/%s-w%s-%s-%s' % (self.model_path,
                                    self.lr, 
                                    self.w_loss_img, 
                                    self.w_loss_code,
                                    self.w_loss_dis)
            os.system('mkdir %s' % (folder))    
            os.system('mv %s/*.pkl %s' % (self.model_path, folder))

        else:
            print("ERROR: \"%s\" not exist" % (final_model))


    def train(self):

        start = time.time()
        total_step = len(self.data_loader)
        fixed_noise = self.to_variable(torch.randn(1, self.h_dim)) # for one image

        if self.resume:
            resume()

        for epoch in range(self.num_epochs):
            for i, data_pack in enumerate(self.data_loader): 
                x_real, _label = data_pack                
                x_real = self.to_variable(x_real, False)

                #===================== E -> G -> loss_img =====================#
                # Push the image x through encoder E to get a real code h           
                # Get two features out of encoder: 
                # comparator feature h1 (pool5) 
                # and h optimization feature (fc6)        

                h_real, h1_real = self.E(x_real)
                h_real = self.to_variable(h_real.data)
                x_fake = self.G(h_real)
                x_fake.retain_grad()

                criterion_1 = nn.MSELoss(True).cuda()
                loss_img = self.w_loss_img * criterion_1(x_fake, x_real)
                
                #=====================  E -> loss_code (h1, h) ================#

                h_fake, h1_fake = self.E(x_fake)

                criterion_h1 = nn.MSELoss(True).cuda()
                target_h1 = self.to_variable(h1_real.data, False)
                loss_h1 = self.w_loss_code * criterion_h1(h1_fake, target_h1)
                loss_code = loss_h1.clone() # caveat usage of clone

                if self.model is "noise":
                    criterion_h = nn.MSELoss(True).cuda()
                    target_h = self.to_variable(h_real.data, False)
                    loss_h  = self.w_loss_code * self.criterion_h(h_fake, target_h)
                    loss_code += loss_h.clone()
                    noise = self.to_variable(torch.randn(self.batch_size, self.h_dim), False)

                #===================== D -> loss_dis  =========================#
                            
                # Train D to recognize real images and real feat as real.
                # 0 for real, 1 for fake

                y = self.D(self.to_variable(x_real.data), 
                            self.to_variable(h_real.data))
                target_real = self.to_variable(torch.zeros(y.size()[0]).long(), False)
                criterion_dis = nn.CrossEntropyLoss().cuda()
                loss_d_real = self.w_loss_dis * criterion_dis(y, target_real)

                if self.train_D:
                    self.D_iter += 1
                    self.D_optimizer.zero_grad()                  
                    loss_d_real.backward(retain_graph=True)

                # @diff: tutorial: L2 loss instead of Binary cross entropy loss 
                # (this is optional for stable training)
                        
                y = self.D(self.to_variable(x_fake.data), 
                            self.to_variable(h_real.data)) 
                target_fake = self.to_variable(torch.ones(y.size()[0]).long(), False) 
                loss_d_fake = self.w_loss_dis * criterion_dis(y, target_fake)

                if self.train_D:
                    loss_d_fake.backward()
                    self.D_optimizer.step()

                # @todo(chuanzi): optimize to back together
                # https://github.com/carpedm20/DiscoGAN-pytorch/blob/master/trainer.py#L213

                loss_dis = loss_d_real + loss_d_fake
                
                #===================== Update G, with 3 losses=====================#
                # L_code from h1/h1+h, loss_code
                # L_img from recon, loss_img
                # L_adv = get bp grad from D for G, loss_adv

                # G maximizes the prob for D to make a mistake
                d_h_real = self.to_variable(h_real.data, False)
                y = self.D(x_fake, d_h_real)
                target_op= self.to_variable(torch.zeros(y.size()[0]).long(), False) 
                loss_adv = self.w_loss_dis * criterion_dis(y, target_op) 

                if self.train_G:
                    
                    self.G_iter += 1
                    self.G_optimizer.zero_grad()
                    
                    # backward accumulated gradient from D,E and loss_img
                    loss_adv.backward(retain_graph=True) # should come from three D sources 
                    # #@(chuanzi): x_fake.grad no data until loss_adv.backward()
                    # > because train loss_dis with only x_fake and x_real values
                    loss_code.backward(retain_graph=True) 
                    x_fake.backward(gradient = x_fake.grad.data, retain_graph=True)
                    loss_img.backward()

                    self.G_optimizer.step()

                
                if (i+1) % self.log_step == 0:

                    print('Epoch [%d/%d], Step[%d/%d], '
                          'Iter[%d/%d]: %.4f(s) '
                          'loss_code: %.4f, ' 
                          'loss_adv: %.4f, '
                          'loss_img: %.4f, '
                          'loss_dis: %.4f, '
                          'loss_d_real:loss_d_fake: %.4f:%.4f'
                            %(epoch+1, self.num_epochs, i+1, total_step, 
                            self.D_iter, self.max_iter, time.time()-start, 
                            loss_code.data[0], 
                            loss_adv.data[0], 
                            loss_img.data[0], 
                            loss_dis.data[0],
                            loss_d_real.data[0],loss_d_fake.data[0]))

                    self.train_mode(loss_dis, loss_adv)
                    start = time.time()
                # # save the sampled images

                if (i+1) % self.sample_step == 0:
                    fake_images = self.G(fixed_noise)
                    torchvision.utils.save_image(self.denorm(fake_images.data), 
                        os.path.join(self.sample_path,
                                    'fake_samples-%d-%d.png' %(epoch+1, i+1)))

                # if os.path.isfile(self.snapshot_at_iter_file):
                #     with open (snapshot_at_iter_file, "r") as myfile:
                #         snapshot_at_iter = int(myfile.read()) #@todo(chuanzi): usage?
            
            # snapshot for each epoch
            g_path = os.path.join(self.model_path, 'G-%d.pkl' %(epoch+1))
            d_path = os.path.join(self.model_path, 'D-%d.pkl' %(epoch+1))
            torch.save(self.G.state_dict(), g_path)
            torch.save(self.D.state_dict(), d_path)

        self.clean_file()