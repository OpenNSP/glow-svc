import torch
from torch import nn

import modules.attentions as attentions
import modules.commons as commons
import modules.modules as modules
import utils
from utils import f0_to_coarse


class ResidualCouplingBlock(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=4,
                 gin_channels=0,
                 share_parameter=False
                 ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        self.wn = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=gin_channels) if share_parameter else None

        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                                              gin_channels=gin_channels, mean_only=True, wn_sharing_parameter=self.wn))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            logdet_tot = 0
            for flow in self.flows:
                x, logdet = flow(x, x_mask, g=g, reverse=reverse)
                logdet_tot += logdet
        else:
            logdet_tot = None
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x, logdet_tot


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        # print(x.shape,x_lengths.shape)
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class TextEncoder(nn.Module):
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 n_layers,
                 gin_channels=0,
                 filter_channels=None,
                 n_heads=None,
                 p_dropout=None):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        self.f0_emb = nn.Embedding(256, hidden_channels)

        self.enc_ = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)

    def forward(self, x, x_mask, f0=None, noice_scale=1):
        x = x + self.f0_emb(f0).transpose(1, 2)
        x = self.enc_(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs) * noice_scale) * x_mask

        return z, m, logs, x_mask


class F0Decoder(nn.Module):
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 spk_channels=0):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.spk_channels = spk_channels

        self.prenet = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.decoder = attentions.FFT(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.f0_prenet = nn.Conv1d(1, hidden_channels, 3, padding=1)
        self.cond = nn.Conv1d(spk_channels, hidden_channels, 1)

    def forward(self, x, norm_f0, x_mask, spk_emb=None):
        x = torch.detach(x)
        if (spk_emb is not None):
            x = x + self.cond(spk_emb)
        x += self.f0_prenet(norm_f0)
        x = self.prenet(x) * x_mask
        x = self.decoder(x * x_mask, x_mask)
        x = self.proj(x) * x_mask
        return x


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(self,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 gin_channels,
                 ssl_dim,
                 n_speakers,
                 sampling_rate=44100,
                 vol_embedding=False,
                 vocoder_name = "nsf-hifigan",
                 use_depthwise_conv = False,
                 use_automatic_f0_prediction = True,
                 flow_share_parameter = False,
                 n_flow_layer = 4,
                 **kwargs):

        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.ssl_dim = ssl_dim
        self.vol_embedding = vol_embedding
        self.emb_g = nn.Embedding(n_speakers, gin_channels)
        self.use_depthwise_conv = use_depthwise_conv
        self.use_automatic_f0_prediction = use_automatic_f0_prediction
        if vol_embedding:
           self.emb_vol = nn.Linear(1, hidden_channels)

        self.pre = nn.Conv1d(ssl_dim, hidden_channels, kernel_size=5, padding=2)

        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout
        )

        modules.set_Conv1dModel(self.use_depthwise_conv)

        self.enc_q = Encoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, n_flow_layer, gin_channels=gin_channels, share_parameter= flow_share_parameter)
        if self.use_automatic_f0_prediction:
            self.f0_decoder = F0Decoder(
                1,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                spk_channels=gin_channels
            )
        self.emb_uv = nn.Embedding(2, hidden_channels)
        self.character_mix = False
        self.spk_emb = nn.Embedding(n_speakers, hidden_channels)

    def EnableCharacterMix(self, n_speakers_map, device):
        self.speaker_map = torch.zeros((n_speakers_map, 1, 1, self.gin_channels)).to(device)
        for i in range(n_speakers_map):
            self.speaker_map[i] = self.emb_g(torch.LongTensor([[i]]).to(device))
        self.speaker_map = self.speaker_map.unsqueeze(0).to(device)
        self.character_mix = True

    def forward(self, c, f0, uv, mel, g=None, c_lengths=None, vol = None):
        xg = self.spk_emb(g).transpose(1,2)
        g = self.emb_g(g).transpose(1, 2)

        # vol proj
        vol = self.emb_vol(vol[:,:,None]).transpose(1,2) if vol is not None and self.vol_embedding else 0

        # ssl prenet
        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1,2) + vol + xg
        
        # f0 predict
        if self.use_automatic_f0_prediction:
            lf0 = 2595. * torch.log10(1. + f0.unsqueeze(1) / 700.) / 500
            norm_lf0 = utils.normalize_f0(lf0, x_mask, uv)
            pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)
        else:
            lf0 = 0
            norm_lf0 = 0
            pred_lf0 = 0
        # encoder
        z_ptemp, m_p, logs_p, _ = self.enc_p(x, x_mask, f0=f0_to_coarse(f0))

        # flow
        z_p, logdet = self.flow(mel, x_mask, g=g)

        return x_mask, (z_p, m_p, logs_p), logdet, pred_lf0, norm_lf0, lf0

    @torch.no_grad()
    def infer(self, c, f0, uv, g=None, noice_scale=0.35, seed=52468, predict_f0=False, vol = None):

        if c.device == torch.device("cuda"):
            torch.cuda.manual_seed_all(seed)
        else:
            torch.manual_seed(seed)

        c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)

        if self.character_mix and len(g) > 1:   # [N, S]  *  [S, B, 1, H]
            g = g.reshape((g.shape[0], g.shape[1], 1, 1, 1))  # [N, S, B, 1, 1]
            g = g * self.speaker_map  # [N, S, B, 1, H]
            g = torch.sum(g, dim=1) # [N, 1, B, 1, H]
            g = g.transpose(0, -1).transpose(0, -2).squeeze(0) # [B, H, N]
        else:
            if g.dim() == 1:
                g = g.unsqueeze(0)
            xg = self.spk_emb(g).transpose(1, 2)
            g = self.emb_g(g).transpose(1, 2)
        
        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        # vol proj
        
        vol = self.emb_vol(vol[:,:,None]).transpose(1,2) if vol is not None and self.vol_embedding else 0

        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2) + vol + xg

        
        if self.use_automatic_f0_prediction and predict_f0:
            lf0 = 2595. * torch.log10(1. + f0.unsqueeze(1) / 700.) / 500
            norm_lf0 = utils.normalize_f0(lf0, x_mask, uv, random_scale=False)
            pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)
            f0 = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)
        
        z_p, m_p, logs_p, c_mask = self.enc_p(x, x_mask, f0=f0_to_coarse(f0), noice_scale=noice_scale)
        o,_ = self.flow(z_p, c_mask, g=g, reverse=True)

        return o,f0

