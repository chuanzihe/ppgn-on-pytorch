import models
import time

import torch
import torchvision
import os
from torch import optim
from torch.autograd import Variable
from ModelPPGN import EncPPGN
from ModelPPGN import GenPPGN
from ModelPPGN import DisPPGN


##################### tutorial solver
class SolverNoiseless():
    def __init__(self, config, data_loader):

        # tutorial settings
        self.G = None
        self.D = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.z_dim = config.z_dim
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.image_size = config.image_size
        self.data_loader = data_loader
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.sample_size = config.sample_size
        self.lr = config.lr
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.sample_path = config.sample_path
        self.model_path = config.model_path
        self.build_model()

        # PPGN default
        self.max_iter = 1300000 # maximum number of iterations
        self.display_every = 50 # show losses every so many iterations
        self.snapshot_every = 1000 # snapshot every so many iterations
        self.snapshot_folder = 'snapshots' # where to save the snapshots (and load from)
        
        # PPGN: we store a bunch of old generated images and feed them to 
        # the discriminator so that it does not overfit to the current data
        self.gen_history_size = 10000 # how many history images to store
        self.use_buffer_from = -1 # set to negative for not using the buffer
        self.gpu_id = 1
        self.feat_shape = (4096,)
        self.comparator_feat_shape = (256,6,6)
        self.im_size = (3,227,227)
        self.batch_size = 64
        self.snapshot_at_iter = -1
        self.snapshot_at_iter_file = 'snapshot_at_iter.txt'

        ## PPGN solver value
        self.display: 0
        # time_per_iter: 1
        # base_lr: 0.0002
        self.base_lr: 0.00005
        self.beta1: 0.9
        self.beta2: 0.999
        self.weight_decay: 0.0004
        self.lr_policy: "multistep"
        self.gamma: 0.5
        self.stepvalue: 300000
        self.stepvalue: 500000
        self.stepvalue: 700000
        self.stepvalue: 900000
        self.stepvalue: 1100000
        self.max_iter: 1300000 # maximum number of iterations @dup  
        # self.epoch = 1300000/batch


        # PPGN loss weight  
        self.W_Limg  = 2e-6 # img_recon_los G
        self.W_Lh1 = 0.01 # feat_recon_loss E
        self.W_Lgan = 100  # softmax loss D
        # @todo(chuanzi): how is loss applied to elsewhere

        # training flags
        self.train_D = True
        self.train_G = True
    
    def build_model(self):
        """Build G, D, E """
        self.G = EncPPGN()
        self.D = GenPPGN()
        self.E = DisPPGN()

        self.G_optimizer = optim.Adam(self.G.parameters(),
                                      self.lr, [self.beta1, self.beta2])
        self.D_optimizer = optim.Adam(self.D.parameters(),
                                      self.lr, [self.beta1, self.beta2])
        # @todo(chuanzi): E, compute but not trainable?
        self.E_optimizer = optim.Adam(self.E.parameters(),
                                      self.lr, [self.beta1, self.beta2])
        
        if torch.cuda.is_available():
            self.G.cuda()
            self.D.cuda()
            self.E.cuda()
        
    def to_variable(self, x):
        """Convert tensor to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)
    
    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data
    
    def reset_grad(self):
        """Zero the gradient buffers."""
        self.D.zero_grad()
        self.G.zero_grad()
    
    def denorm(self, x):
        """Convert range (-1, 1) to (0, 1)"""
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def train(self):
        """Train G, D, and E"""
        start = time.time()
        fixed_noise = self.to_variable(torch.randn(self.batch_size, self.z_dim))
        total_step = len(self.data_loader)
        for epoch in range(self.num_epochs):
            for i, img_real in enumerate(self.data_loader): # one iteration

                img_real = self.to_variable(img_real)
                batch_size = img_real.size(0)

                #===================== Train E, L_img =====================#
                # Push the image x through encoder E to get a real code h
                h1_in = torch.zeros((self.batch_size,) + 
                                comparator_feat_shape, dtype='float32')
                
                # Get two features out of encoder: comparator feature h1 (pool5) 
                # and h optimization feature (fc6)
                h_real, h1_real = self.E(img_real, h1_in)
                img_fake = self.G(h_real, img_real)
                loss_img = nn.MSELoss(img_fake - img_real).cuda()
                # @todo(chuanzi): Euclidean loss? l2 loss
                
                #===================== Train E, L_h1, L_h =====================#

                h_fake, h1_fake = self.E(img_fake, h1_real)
                loss_h1 = nn.MSELoss(h1_fake, h1_real).cuda() 
                # @ EuclideanLoss no size average?
                # tutorial loss: torch.mean((outputs - 1) ** 2) 

                if config.model is not "noiseless"
                    loss_h  = nn.MSELoss(h_fake, h_real).cuda()


                #===================== Train D, L_gan  =====================#

                noise = self.to_variable(torch.randn(batch_size, self.z_dim))
 
                
                # Train D to recognize real images and real feat as real.
                """ 0 for real, 1 for fake"""
                y = self.D(img_real, h_real)
                target = self.to_variable(torch.zeros((batch_size, 1, 1, 1))                             
                loss_d_real = nn.CrossEntropyLoss(y, target).cuda()  
                # @diff:softmax loss
                # tutorial: L2 loss instead of Binary cross entropy loss 
                # (this is optional for stable training)

                # Run D on the fake data                    
                y = self.D(img_fake, h_real)
                target = self.to_variable(torch.ones((batch_size, 1, 1, 1))) 
                loss_d_fake = nn.CrossEntropyLoss(y, target).cuda()

                loss_d = loss_d_real + loss_d_fake

                # Update D
                if self.train_D:
                  # Q: there are 2 backward passes before this update
                  # Does this update combine the gradients from these two passes?
                  # Assume; YES (gradient accumulation)
                  # @todo(chuanzi): combine and backward together?
                  # @@todo(chuanzi): increment_iter()?
                  # @diff: caffe: two backward pass for real and fake
                    self.D.zero_grad()                  
                    loss_d.backward()                 
                    self.D_optimizer.step() # @todo(chuanzi): D.apply_update()?
                
                #===================== Train G, with 3 losses=====================#
                # compute grad from D for G
                # G maximizes the prob for D to make a mistake
                # run D on generated data and opposite labels to get grad of G

                y = self.D(img_fake, h_real)
                target_ops= self.to_variable(torch.zeros((batch_size, 1, 1, 1)))
                loss_g_d_ops = nn.CrossEntropyLoss(y, target_ops) # loss_G
                loss_d = loss_d_real + loss_d_fake

                if self.train_G:
                    # @todo(chuanzi): 
                    # G.increment_iter()
                    self.G.zero_grad()
                    # loss_h1.backward() # bp from E
                    # loss_g_d_ops.backward() # backprop to related grad,  l->D

                    # @todo(chuanzi): how to apply chain rule? in models?
                    loss_g = loss_h1 + loss_g_d_ops + loss_img + loss_d
                    loss_g.backward()
                    self.G_optimizer.step()

    
                # print the log info
                self.iter += 1
                if (i+1) % self.log_step == 0:
                    print('Epoch [%d/%d], Step[%d/%d], '
                          '[%s] Iteration(caffe)[%d/%d]: %.4f seconds '
                          'loss_h1: %.4f, ' 
                          'loss_g_d_ops: %.4f'
                          'loss_img: %.4f'
                          'loss_d_real: %.4f'
                          'loss_d_fake: %.4f'
                          %(epoch+1, self.num_epochs, i+1, total_step,
                            time.strftime("%c"), self.iter, 
                            self.max_iter, time.time()-start, 
                            loss_h1.data[0], loss_g_d_ops.data[0], 
                            loss_img.data[0], loss_d.data[0]))
                    start. = time.time()      
                    if os.path.isfile(snapshot_at_iter_file):
                        with open (snapshot_at_iter_file, "r") as myfile:
                            snapshot_at_iter = int(myfile.read())


                # save the sampled images
                # @todo(chuanzi): no sampling of ppgn during training?
                if (i+1) % self.sample_step == 0:
                    fake_images = self.G(fixed_noise)
                    torchvision.utils.save_image(self.denorm(fake_images.data), 
                        os.path.join(self.sample_path,
                                     'fake_samples-%d-%d.png' %(epoch+1, i+1)))
            
            # snapshot for each epoch
            # diff: save for iterations
            g_path = os.path.join(self.model_path, 'G-%d.pkl' %(epoch+1))
            d_path = os.path.join(self.model_path, 'D-%d.pkl' %(epoch+1))
            torch.save(self.G.state_dict(), g_path)
            torch.save(self.D.state_dict(), d_path)
