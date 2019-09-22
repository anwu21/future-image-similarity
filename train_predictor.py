import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import itertools
import progressbar
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=60, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save trained models')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--dataset', default='lab_pose', help='predictor training data: lab_pose or gaz_pose')
parser.add_argument('--n_past', type=int, default=3, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=1, help='number of frames to predict during training')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
parser.add_argument('--gpu', default='0', help='gpu number')

opt = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu

if 'lab' in opt.dataset:
    opt.data_root = 'data/dataset_imitation/lab'
    opt.skip_factor = 3
    opt.n_past = 3
elif 'gaz' in opt.dataset:
    opt.data_root = 'data/dataset_imitation/sim/targets/target1'
    opt.skip_factor = 10
    opt.n_past = 5

if opt.model_dir != '':
    # load model and continue training from checkpoint
    saved_model = torch.load('%s/model.pth' % opt.model_dir)
        
    optimizer = opt.optimizer
    model_dir = opt.model_dir
    opt = saved_model['opt']
    opt.optimizer = optimizer
    opt.model_dir = model_dir
    opt.log_dir = '%s/continued' % opt.log_dir
else:
    name = 'model_predictor'

    opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)

os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor


# ---------------- load the models  ----------------

print(opt)

# ---------------- optimizer ----------------
opt.optimizer = optim.Adam

from models.model_predictor import gaussian_lstm as lstm_model
if opt.model_dir != '':
    posterior = saved_model['posterior']
    prior = saved_model['prior']
else:
    posterior = lstm_model(4*opt.g_dim, opt.z_dim, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size)
    prior = lstm_model(4*opt.g_dim, opt.z_dim, opt.rnn_size, opt.prior_rnn_layers, opt.batch_size)

    posterior.apply(utils.init_weights)
    prior.apply(utils.init_weights)

import models.model_predictor as model
       
if opt.model_dir != '':
    decoder = saved_model['decoder']
    encoder = saved_model['encoder']
else:
    encoder = model.encoder_conv(opt.g_dim, opt.channels)
    decoder = model.decoder_conv(opt.g_dim, opt.channels)
    encoder.apply(utils.init_weights)
    decoder.apply(utils.init_weights)

pose_network = model.pose_network()
conv_network = model.conv_network(16+opt.g_dim+int(opt.z_dim/4), opt.g_dim)
pose_network.apply(utils.init_weights)
conv_network.apply(utils.init_weights)

posterior_optimizer = opt.optimizer(posterior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
prior_optimizer = opt.optimizer(prior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_optimizer = opt.optimizer(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
decoder_optimizer = opt.optimizer(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
pose_network_optimizer = opt.optimizer(pose_network.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
conv_network_optimizer = opt.optimizer(conv_network.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# --------- loss functions ------------------------------------
l1_criterion = nn.SmoothL1Loss()
def kl_criterion(mu1, logvar1, mu2, logvar2):
    sigma1 = logvar1.mul(0.5).exp() 
    sigma2 = logvar2.mul(0.5).exp() 
    kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
    return kld.sum() / opt.batch_size

# --------- transfer to gpu ------------------------------------
posterior.cuda()
prior.cuda()
encoder.cuda()
decoder.cuda()
pose_network.cuda()
conv_network.cuda()

l1_criterion.cuda()

# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt)

train_loader = DataLoader(train_data,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=opt.data_threads,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)

def get_training_batch():
    while True:
        for sequence in train_loader:
            batch = utils.normalize_data(opt, dtype, sequence[0])
            yield batch, sequence[1], sequence[2]
training_batch_generator = get_training_batch()

def get_testing_batch():
    while True:
        for sequence in test_loader:
            batch = utils.normalize_data(opt, dtype, sequence[0])
            yield batch, sequence[1], sequence[2]
testing_batch_generator = get_testing_batch()

def plot(x, diff_pose, epoch):
    posterior.hidden = posterior.init_hidden()
    gen_seq = []
    gen_seq.append(x[0])
    for i in range(1, opt.n_past+opt.n_future):
        h_conv = encoder(x[i-1])
        h_target = encoder(x[i])[0]
        
        if opt.last_frame_skip or i < opt.n_past:	
            h_conv, skip = h_conv
        else:
            h_conv = h_conv[0]
        h = h_conv.view(-1, 4*opt.g_dim)

        h_conv = h_conv.detach()
        h_target = h_target.detach()
        z_t, _, _= posterior(h_target)
        z_t = z_t.view(-1, int(opt.z_dim/4), 2, 2)

        if i < opt.n_past:
            gen_seq.append(x[i])
        else:
            z_d = pose_network(Variable(diff_pose[i-1].cuda())).detach()
            h_pred = conv_network(torch.cat([h_conv, z_t, z_d], 1)).detach()
            x_pred = decoder([h_pred, skip]).detach()
            gen_seq.append(x_pred)

    to_plot = []
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        row = []
        for t in range(opt.n_past+opt.n_future):
            row.append(gen_seq[t][i]) 
        to_plot.append(row)
    fname = '%s/gen/rec_%d.png' % (opt.log_dir, epoch) 
    utils.save_tensors_image(fname, to_plot)

def train(x, diff_pose):
    posterior.zero_grad()
    prior.zero_grad()
    encoder.zero_grad()
    decoder.zero_grad()
    pose_network.zero_grad()
    conv_network.zero_grad()

    # initialize the hidden state.
    posterior.hidden = posterior.init_hidden()
    prior.hidden = prior.init_hidden()

    l1 = 0
    kld = 0
    for i in range(1, opt.n_past+opt.n_future):
        h_conv = encoder(x[i-1])
        h_target = encoder(x[i])[0]
        if opt.last_frame_skip or i < opt.n_past:	
            h_conv, skip = h_conv
        else:
            h_conv = h_conv[0]
        h = h_conv.view(-1, 4*opt.g_dim)
        
        z_t, mu, logvar = posterior(h_target)
        z_t = z_t.view(-1, int(opt.z_dim/4), 2, 2)
        _, mu_p, logvar_p = prior(h)
        
        z_d = pose_network(Variable(diff_pose[i-1].cuda()))
        h_pred = conv_network(torch.cat([h_conv, z_t, z_d], 1))
 
        x_pred = decoder([h_pred, skip])

        l1 += l1_criterion(x_pred, x[i])
        kld += kl_criterion(mu, logvar, mu_p, logvar_p)

    loss = l1 + kld*opt.beta
    loss.backward()

    posterior_optimizer.step()
    prior_optimizer.step()
    encoder_optimizer.step()
    decoder_optimizer.step()
    pose_network_optimizer.step()
    conv_network_optimizer.step()

    return l1.data.cpu().numpy()/(opt.n_past+opt.n_future), kld.data.cpu().numpy()/(opt.n_future+opt.n_past)

# --------- training loop ------------------------------------
for epoch in range(opt.niter):
    posterior.train()
    prior.train()
    encoder.train()
    decoder.train()
    pose_network.train()
    conv_network.train()
    
    epoch_l1 = 0
    epoch_kld = 0
    progress = progressbar.ProgressBar(max_value=opt.epoch_size).start()
    print()
    for i in range(opt.epoch_size):
        progress.update(i+1)
        x, diff_pose, _ = next(training_batch_generator)
        l1, kld = train(x, diff_pose)
        epoch_l1 += l1
        epoch_kld += kld

    progress.finish()
    utils.clear_progressbar()

    print('training: [%02d] l1 loss: %.5f | kld loss: %.5f (%d)' % (epoch, epoch_l1/opt.epoch_size, epoch_kld/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size))
        
    torch.save({
        'pose_network': pose_network,
        'conv_network': conv_network,
        'encoder': encoder,
        'decoder': decoder,
        'posterior': posterior,
        'prior': prior,
        'opt': opt},
        '%s/model.pth' % opt.log_dir)
    
    x, diff_pose, _ = next(testing_batch_generator)

    if epoch % 20 == 0:
        print('log dir: %s' % opt.log_dir)        
        encoder.eval()
        decoder.eval()
        posterior.eval()
        prior.eval()
        pose_network.eval()
        conv_network.eval()

        plot(x, diff_pose, epoch)
            
        

