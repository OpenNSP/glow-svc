import torch
from torch import nn
from torch.nn import functional as F

import modules.commons as commons
from modules.commons import get_padding, init_weights
from modules.DSConv import (
    Depthwise_Separable_Conv1D,
    remove_weight_norm_modules,
    weight_norm_modules,
)

LRELU_SLOPE = 0.1

Conv1dModel = nn.Conv1d

def set_Conv1dModel(use_depthwise_conv):
    global Conv1dModel
    Conv1dModel = Depthwise_Separable_Conv1D if use_depthwise_conv else nn.Conv1d


class ActNorm(nn.Module):
  def __init__(self, channels, ddi=False, **kwargs):
    super().__init__()
    self.channels = channels
    self.initialized = not ddi

    self.logs = nn.Parameter(torch.zeros(1, channels, 1))
    self.bias = nn.Parameter(torch.zeros(1, channels, 1))

  def forward(self, x, x_mask=None, g=None, reverse=False, **kwargs):
    if x_mask is None:
      x_mask = torch.ones(x.size(0), 1, x.size(2)).to(device=x.device, dtype=x.dtype)
    x_len = torch.sum(x_mask, [1, 2])
    if not self.initialized:
      self.initialize(x, x_mask)
      self.initialized = True

    if reverse:
      z = (x - self.bias) * torch.exp(-self.logs) * x_mask
      logdet = None
      return z
    else:
      z = (self.bias + torch.exp(self.logs) * x) * x_mask
      logdet = torch.sum(self.logs) * x_len # [b]
      return z, logdet
    

  def store_inverse(self):
    pass

  def set_ddi(self, ddi):
    self.initialized = not ddi

  def initialize(self, x, x_mask):
    with torch.no_grad():
      denom = torch.sum(x_mask, [0, 2])
      m = torch.sum(x * x_mask, [0, 2]) / denom
      m_sq = torch.sum(x * x * x_mask, [0, 2]) / denom
      v = m_sq - (m ** 2)
      logs = 0.5 * torch.log(torch.clamp_min(v, 1e-6))

      bias_init = (-m * torch.exp(-logs)).view(*self.bias.shape).to(dtype=self.bias.dtype)
      logs_init = (-logs).view(*self.logs.shape).to(dtype=self.logs.dtype)

      self.bias.data.copy_(bias_init)
      self.logs.data.copy_(logs_init)
      
  
class InvConvNear(nn.Module):
  def __init__(self, channels, n_split=4, no_jacobian=False, **kwargs):
    super().__init__()
    assert(n_split % 2 == 0)
    self.channels = channels
    self.n_split = n_split
    self.no_jacobian = no_jacobian
    
    w_init = torch.qr(torch.FloatTensor(self.n_split, self.n_split).normal_())[0]
    if torch.det(w_init) < 0:
      w_init[:,0] = -1 * w_init[:,0]
    self.weight = nn.Parameter(w_init)

  def forward(self, x, x_mask=None, g=None, reverse=False, **kwargs):
    b, c, t = x.size()
    assert(c % self.n_split == 0)
    if x_mask is None:
      x_mask = 1
      x_len = torch.ones((b,), dtype=x.dtype, device=x.device) * t
    else:
      x_len = torch.sum(x_mask, [1, 2])

    x = x.view(b, 2, c // self.n_split, self.n_split // 2, t)
    x = x.permute(0, 1, 3, 2, 4).contiguous().view(b, self.n_split, c // self.n_split, t)

    if reverse:
      if hasattr(self, "weight_inv"):
        weight = self.weight_inv
      else:
        weight = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)
      logdet = None
    else:
      weight = self.weight
      if self.no_jacobian:
        logdet = 0
      else:
        logdet = torch.logdet(self.weight) * (c / self.n_split) * x_len # [b]

    weight = weight.view(self.n_split, self.n_split, 1, 1)
    z = F.conv2d(x, weight)

    z = z.view(b, 2, self.n_split // 2, c // self.n_split, t)
    z = z.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t) * x_mask
    if reverse:
      return z
    else:
      return z, logdet

  def store_inverse(self):
    self.weight_inv = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)

class LayerNorm(nn.Module):
  def __init__(self, channels, eps=1e-5):
    super().__init__()
    self.channels = channels
    self.eps = eps

    self.gamma = nn.Parameter(torch.ones(channels))
    self.beta = nn.Parameter(torch.zeros(channels))

  def forward(self, x):
    x = x.transpose(1, -1)
    x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
    return x.transpose(1, -1)

 
class ConvReluNorm(nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
    super().__init__()
    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.n_layers = n_layers
    self.p_dropout = p_dropout
    assert n_layers > 1, "Number of layers should be larger than 0."

    self.conv_layers = nn.ModuleList()
    self.norm_layers = nn.ModuleList()
    self.pre = Conv1dModel(in_channels, hidden_channels, kernel_size, padding=kernel_size//2)
    self.conv_layers.append(Conv1dModel(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2))
    self.norm_layers.append(LayerNorm(hidden_channels))
    self.relu_drop = nn.Sequential(
        nn.ReLU(),
        nn.Dropout(p_dropout))
    for _ in range(n_layers-1):
      self.conv_layers.append(Conv1dModel(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2))
      self.norm_layers.append(LayerNorm(hidden_channels))
    self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
    self.proj.weight.data.zero_()
    self.proj.bias.data.zero_()

  def forward(self, x):
    x = self.pre(x)
    x_org = x
    for i in range(self.n_layers):
      x = self.conv_layers[i](x)
      x = self.norm_layers[i](x)
      x = self.relu_drop(x)
    x = x_org + self.proj(x)
    return x


class WN(torch.nn.Module):
  def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
    super(WN, self).__init__()
    assert(kernel_size % 2 == 1)
    self.hidden_channels =hidden_channels
    self.kernel_size = kernel_size,
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels
    self.p_dropout = p_dropout

    self.in_layers = torch.nn.ModuleList()
    self.res_skip_layers = torch.nn.ModuleList()
    self.drop = nn.Dropout(p_dropout)

    if gin_channels != 0:
      cond_layer = torch.nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)
      self.cond_layer = weight_norm_modules(cond_layer, name='weight')

    for i in range(n_layers):
      dilation = dilation_rate ** i
      padding = int((kernel_size * dilation - dilation) / 2)
      in_layer = Conv1dModel(hidden_channels, 2*hidden_channels, kernel_size,
                                 dilation=dilation, padding=padding)
      in_layer = weight_norm_modules(in_layer, name='weight')
      self.in_layers.append(in_layer)

      # last one is not necessary
      if i < n_layers - 1:
        res_skip_channels = 2 * hidden_channels
      else:
        res_skip_channels = hidden_channels

      res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
      res_skip_layer = weight_norm_modules(res_skip_layer, name='weight')
      self.res_skip_layers.append(res_skip_layer)

  def forward(self, x, x_mask, g=None, **kwargs):
    output = torch.zeros_like(x)
    n_channels_tensor = torch.IntTensor([self.hidden_channels])

    if g is not None:
      g = self.cond_layer(g)

    for i in range(self.n_layers):
      x_in = self.in_layers[i](x)
      if g is not None:
        cond_offset = i * 2 * self.hidden_channels
        g_l = g[:,cond_offset:cond_offset+2*self.hidden_channels,:]
      else:
        g_l = torch.zeros_like(x_in)

      acts = commons.fused_add_tanh_sigmoid_multiply(
          x_in,
          g_l,
          n_channels_tensor)
      acts = self.drop(acts)

      res_skip_acts = self.res_skip_layers[i](acts)
      if i < self.n_layers - 1:
        res_acts = res_skip_acts[:,:self.hidden_channels,:]
        x = (x + res_acts) * x_mask
        output = output + res_skip_acts[:,self.hidden_channels:,:]
      else:
        output = output + res_skip_acts
    return output * x_mask

  def remove_weight_norm(self):
    if self.gin_channels != 0:
      remove_weight_norm_modules(self.cond_layer)
    for l in self.in_layers:
      remove_weight_norm_modules(l)
    for l in self.res_skip_layers:
      remove_weight_norm_modules(l)


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm_modules(l)
        for l in self.convs2:
            remove_weight_norm_modules(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm_modules(l)


class Log(nn.Module):
  def forward(self, x, x_mask, reverse=False, **kwargs):
    if not reverse:
      y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
      logdet = torch.sum(-y, [1, 2])
      return y, logdet
    else:
      x = torch.exp(x) * x_mask
      return x
    

class Flip(nn.Module):
  def forward(self, x, *args, reverse=False, **kwargs):
    x = torch.flip(x, [1])
    if not reverse:
      logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
      return x, logdet
    else:
      return x


class ElementwiseAffine(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.channels = channels
    self.m = nn.Parameter(torch.zeros(channels,1))
    self.logs = nn.Parameter(torch.zeros(channels,1))

  def forward(self, x, x_mask, reverse=False, **kwargs):
    if not reverse:
      y = self.m + torch.exp(self.logs) * x
      y = y * x_mask
      logdet = torch.sum(self.logs * x_mask, [1,2])
      return y, logdet
    else:
      x = (x - self.m) * torch.exp(-self.logs) * x_mask
      return x


class ResidualCouplingLayer(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      p_dropout=0,
      gin_channels=0,
      mean_only=False,
      wn_sharing_parameter=None
      ):
    assert channels % 2 == 0, "channels should be divisible by 2"
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.half_channels = channels // 2
    self.mean_only = mean_only

    self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
    self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels) if wn_sharing_parameter is None else wn_sharing_parameter
    self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
    self.post.weight.data.zero_()
    self.post.bias.data.zero_()

  def forward(self, x, x_mask, g=None, reverse=False):
    x0, x1 = torch.split(x, [self.half_channels]*2, 1)
    h = self.pre(x0) * x_mask
    h = self.enc(h, x_mask, g=g)
    stats = self.post(h) * x_mask
    if not self.mean_only:
      m, logs = torch.split(stats, [self.half_channels]*2, 1)
    else:
      m = stats
      logs = torch.zeros_like(m)

    if not reverse:
      x1 = m + x1 * torch.exp(logs) * x_mask
      x = torch.cat([x0, x1], 1)
      logdet = torch.sum(logs, [1,2])
      return x, logdet
    else:
      x1 = (x1 - m) * torch.exp(-logs) * x_mask
      x = torch.cat([x0, x1], 1)
      return x
