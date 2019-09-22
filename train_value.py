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
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save trained models')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--niter', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--dataset', default='lab_value', help='critic training data: lab_value or gaz_value')
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
parser.add_argument('--num_cand', default=16, type=int, help='number of candidate poses for value function')

opt = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu

# load model
if 'lab' in opt.dataset:
    opt.model_dir = 'logs/lab_pose/model_predictor'
elif 'gaz' in opt.dataset:
    opt.model_dir = 'logs/gaz_pose/model_predictor'
saved_model = torch.load('%s/model.pth' % opt.model_dir)
model_dir = opt.model_dir
niter = opt.niter
epoch_size = opt.epoch_size
log_dir = opt.log_dir
num_cand = opt.num_cand
lr = opt.lr
dataset = opt.dataset

opt = saved_model['opt']
opt.model_dir = model_dir
opt.niter = niter
opt.epoch_size = epoch_size
opt.log_dir = log_dir
opt.num_cand = num_cand
opt.lr = lr
opt.dataset = dataset

name = 'model_critic'

opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)
os.makedirs(opt.log_dir, exist_ok=True)

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor

# ---------------- load the models  ----------------

print(opt)

prior = saved_model['prior']
encoder = saved_model['encoder']
decoder = saved_model['decoder']
pose_network = saved_model['pose_network']
conv_network = saved_model['conv_network']

from models.model_value import ModelValue
value_network = ModelValue()
value_network.apply(utils.init_weights)

# ---------------- optimizers ----------------
opt.optimizer = optim.Adam
value_network_optimizer = opt.optimizer(value_network.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# --------- loss functions -------------------
loss_criterion = nn.SmoothL1Loss()

# --------- transfer to gpu ------------------
prior.cuda()
encoder.cuda()
decoder.cuda()
pose_network.cuda()
conv_network.cuda()
value_network.cuda()

loss_criterion.cuda()

# --------- load a dataset -------------------
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

    
# --------- training funtions ------------------------------------
def train(x, diff_pose_rand, num_cand):
    #value_network.zero_grad()

    # initialize the hidden state.
    prior.hidden = prior.init_hidden()
    
    reward_loss = 0.0
    running_loss = 0.0
    
    h_conv = encoder(x[0])
    h_conv, skip = h_conv
    h = h_conv.view(-1, 4*opt.g_dim)
    z_t, _, _ = prior(h)
    z_t = z_t.view(-1, int(opt.z_dim/4), 2, 2)

    z_d_exp = pose_network(Variable(diff_pose_rand[0].cuda())).detach()
    h_pred_exp = conv_network(torch.cat([h_conv, z_t, z_d_exp], 1)).detach()
    x_pred_exp = decoder([h_pred_exp, skip]).detach()

    h_conv = encoder(x[0])
    h_conv, skip = h_conv
    
    for j in range(num_cand):
        value_network.zero_grad()
        z_d_rand = pose_network(Variable(diff_pose_rand[j].cuda()))
        h_pred_rand = conv_network(torch.cat([h_conv, z_t, z_d_rand], 1))
        x_pred_rand = decoder([h_pred_rand, skip])
        x_value_rand = value_network(x_pred_rand)
    
        reward_label = []
        for batch_idx in range(opt.batch_size):
            reward_label.append(loss_criterion(x_pred_rand[batch_idx], x_pred_exp[batch_idx].detach()).data)

        reward_label = torch.stack(reward_label)   
        reward_label = Variable(reward_label, requires_grad=False)

        reward_loss += loss_criterion(x_value_rand, reward_label)

    loss = reward_loss
    loss.backward()
##        loss.backward(retain_graph=True)
    
    value_network_optimizer.step()
    running_loss = reward_loss
    return running_loss.data.cpu().numpy(), x_value_rand.data.cpu().numpy()[0:3]

def test(x, diff_pose_rand, num_cand):   
    # initialize the hidden state.
    prior.hidden = prior.init_hidden()
    
    reward_loss = 0.0
    
    h_conv = encoder(x[0])
    h_conv, skip = h_conv
    h = h_conv.view(-1, 4*opt.g_dim)
    z_t, _, _ = prior(h)
    z_t = z_t.view(-1, int(opt.z_dim/4), 2, 2)

    z_d_exp = pose_network(Variable(diff_pose_rand[0].cuda())).detach()
    h_pred_exp = conv_network(torch.cat([h_conv, z_t, z_d_exp], 1)).detach()
    x_pred_exp = decoder([h_pred_exp, skip]).detach()

    for j in range(num_cand):
        z_d_rand = pose_network(Variable(diff_pose_rand[j].cuda()))
        h_pred_rand = conv_network(torch.cat([h_conv, z_t, z_d_rand], 1))
        x_pred_rand = decoder([h_pred_rand, skip])
        x_value_rand = value_network(x_pred_rand)

        reward_label = []
        for batch_idx in range(opt.batch_size):
            reward_label.append(loss_criterion(x_pred_rand[batch_idx], x_pred_exp[batch_idx].detach()).data)

        reward_label = torch.stack(reward_label)
        reward_label = Variable(reward_label, requires_grad=False)

        reward_loss += loss_criterion(x_value_rand, reward_label)
   
    return reward_loss.data.cpu().numpy(), x_value_rand.data.cpu().numpy()[0:3]

# --------- training loop ------------------------------------
best_loss = 100000.0

for epoch in range(opt.niter):
    
    ###### TRAIN ######
    epoch_err = 0
    
    value_network.train()
    prior.eval()
    pose_network.eval()
    conv_network.eval()
    encoder.eval()
    decoder.eval()
    
    progress = progressbar.ProgressBar(max_value=opt.epoch_size).start()
    for i in range(opt.epoch_size):
        progress.update(i+1)
        x, diff_pose, _ = next(training_batch_generator)
        err, value_pred = train(x, diff_pose, opt.num_cand)
        epoch_err += err
    progress.finish()
    utils.clear_progressbar()
    print(name)
    print('TRAIN: [%02d] loss: %.10f (%d)' % (epoch, epoch_err/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size))
    
    ###### TEST ######
    test_epoch_size = max(int(opt.epoch_size / 10.0), 1)
    epoch_err_test = 0
    
    value_network.eval()
    prior.eval()
    pose_network.eval()
    conv_network.eval()
    encoder.eval()
    decoder.eval()
    
    progress_test = progressbar.ProgressBar(max_value=test_epoch_size).start()
    for i in range(test_epoch_size):
        progress_test.update(i+1)
        x, diff_pose, _ = next(testing_batch_generator)
        err_test, value_pred = test(x, diff_pose, opt.num_cand)
        epoch_err_test += err_test

    progress_test.finish()
    utils.clear_progressbar()
    
    print('TEST: [%02d] loss: %.10f (%d)' % (epoch, epoch_err_test/test_epoch_size, epoch*test_epoch_size*opt.batch_size))

    with open('%s/log_train.txt' % opt.log_dir, 'a') as f:
        f.write('TRAIN: [%02d] loss: %.10f (%d)\n' % (epoch, epoch_err/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size))
    with open('%s/log_test.txt' % opt.log_dir, 'a') as f:
        f.write('TEST: [%02d] loss: %.10f (%d)\n' % (epoch, epoch_err_test/test_epoch_size, epoch*test_epoch_size*opt.batch_size))

    opt.lr = 0.99 * opt.lr

    if epoch_err_test < best_loss:
        best_loss = epoch_err_test
        torch.save({
            'value_network': value_network,
            'opt': opt},
            '%s/model_critic.pth' % opt.log_dir)


