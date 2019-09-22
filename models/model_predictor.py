import torch
import torch.nn as nn
from torch.autograd import Variable

class cnn_layer(nn.Module):
    def __init__(self, nin, nout):
        super(cnn_layer, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 3, 1, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True)
                )

    def forward(self, input):
        return self.main(input)

class encoder_conv(nn.Module):
    def __init__(self, dim, nc=1):
        super(encoder_conv, self).__init__()
        self.dim = dim
        # 64 x 64
        self.c1 = nn.Sequential(
                cnn_layer(nc, 64),
                cnn_layer(64, 64),
                )
        # 32 x 32
        self.c2 = nn.Sequential(
                cnn_layer(64, 128),
                cnn_layer(128, 128),
                )
        # 16 x 16 
        self.c3 = nn.Sequential(
                cnn_layer(128, 256),
                cnn_layer(256, 256),
                cnn_layer(256, 256),
                )
        # 8 x 8
        self.c4 = nn.Sequential(
                cnn_layer(256, 512),
                cnn_layer(512, 512),
                cnn_layer(512, 512),
                )
        # 4 x 4
        self.c5 = nn.Sequential(
                nn.Conv2d(512, dim, 3, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, input):
        h1 = self.c1(input) # 64 -> 32
        h2 = self.c2(self.mp(h1)) # 32 -> 16
        h3 = self.c3(self.mp(h2)) # 16 -> 8
        h4 = self.c4(self.mp(h3)) # 8 -> 4
        h5 = self.c5(self.mp(h4)) # 4 -> 2
        return h5.view(-1, self.dim, 2, 2), [h1, h2, h3, h4]

class decoder_conv(nn.Module):
    def __init__(self, dim, nc=1):
        super(decoder_conv, self).__init__()
        self.dim = dim
        # 1 x 1 -> 4 x 4
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 512, 3, 1, 0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # 8 x 8
        self.upc2 = nn.Sequential(
                cnn_layer(512*2, 512),
                cnn_layer(512, 512),
                cnn_layer(512, 256)
                )
        # 16 x 16
        self.upc3 = nn.Sequential(
                cnn_layer(256*2, 256),
                cnn_layer(256, 256),
                cnn_layer(256, 128)
                )
        # 32 x 32
        self.upc4 = nn.Sequential(
                cnn_layer(128*2, 128),
                cnn_layer(128, 64)
                )
        # 64 x 64
        self.upc5 = nn.Sequential(
                cnn_layer(64*2, 64),
                nn.ConvTranspose2d(64, nc, 3, 1, 1),
                nn.Sigmoid()
                )
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input):
        vec, skip = input 
        d1 = self.upc1(vec.view(-1, self.dim, 2, 2)) # 2 -> 4
        up1 = self.up(d1) # 4 -> 8
        d2 = self.upc2(torch.cat([up1, skip[3]], 1)) # 8 x 8
        up2 = self.up(d2) # 8 -> 16 
        d3 = self.upc3(torch.cat([up2, skip[2]], 1)) # 16 x 16 
        up3 = self.up(d3) # 8 -> 32 
        d4 = self.upc4(torch.cat([up3, skip[1]], 1)) # 32 x 32
        up4 = self.up(d4) # 32 -> 64
        output = self.upc5(torch.cat([up4, skip[0]], 1)) # 64 x 64
        return output

class lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.output = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                nn.Tanh())
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))
        return hidden

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]

        return self.output(h_in)
    
class gaussian_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(gaussian_lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))
        return hidden

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class conv_network(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(conv_network, self).__init__()

        self.pre_lstm = nn.Sequential(nn.Conv2d(in_dim, in_dim, 1),
                                        nn.Tanh(),
                                        nn.Conv2d(in_dim, out_dim, 1),
                                        nn.Tanh())

    def forward(self, z):
        h_pred = self.pre_lstm(z)
        return h_pred

class pose_network(nn.Module):
    def __init__(self):
        super(pose_network, self).__init__()

        self.theta_transform = nn.Sequential(nn.Linear(3, 2*2*16),
                                             nn.Tanh(),
                                             nn.Linear(2*2*16, 2*2*16),
                                             nn.Tanh())
    
    def forward(self, delta_theta):
        z_theta = self.theta_transform(delta_theta).view(-1, 16, 2, 2)
        return z_theta
