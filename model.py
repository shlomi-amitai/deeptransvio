import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs, lazy_property, clamp_probs
from extra_models import image_Inertial_Encoder
import torch.nn.functional as F
import os
from utils.utils import visualize_imu_sequence

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )

class PyramidalIMUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, levels=3):
        super(PyramidalIMUEncoder, self).__init__()
        self.levels = levels
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        for i in range(levels):
            self.convs.append(nn.Conv1d(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size=3, padding=1))
            self.pools.append(nn.AvgPool1d(kernel_size=2, stride=2))
        
        self.global_conv = nn.Conv1d(hidden_dim * levels, output_dim, kernel_size=1)
        
    def forward(self, x):
        # x: (batch_size, seq_len * 11, 6)
        features = []
        for i in range(self.levels):
            x = F.relu(self.convs[i](x))
            features.append(x)
            x = self.pools[i](x)
        
        # Upsample and concatenate features
        for i in range(1, self.levels):
            features[i] = F.interpolate(features[i], size=features[0].shape[-1], mode='linear', align_corners=False)
        
        x = torch.cat(features, dim=1)
        x = self.global_conv(x)
        
        return x  # (batch_size, output_dim, seq_len * 11)

# The inertial encoder for raw imu data
class Inertial_encoder(nn.Module):
    def __init__(self, opt):
        super(Inertial_encoder, self).__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout))
        self.proj = nn.Linear(256 * 1 * 11, opt.i_f_len)

    def forward(self, x):
        # x: (N, seq_len, 11, 6)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size * seq_len, x.size(2), x.size(3))    # x: (N x seq_len, 11, 6)
        x = self.encoder_conv(x.permute(0, 2, 1))                 # x: (N x seq_len, 64, 11)
        out = self.proj(x.view(x.shape[0], -1))                   # out: (N x seq_len, 256)
        return out.view(batch_size, seq_len, 256)


import torch.nn as nn

class CrossSequenceAttentionEncoder(nn.Module):
    def __init__(self, opt):
        super(CrossSequenceAttentionEncoder, self).__init__()

        self.sensor_dim = 6
        self.hidden_dim = 64
        self.num_heads = 4
        self.num_layers = 2
        self.max_time_steps = 11
        self.max_seq_len = opt.seq_len  # Assuming this is defined in opt

        # Embeddings
        self.sensor_embed = nn.Linear(self.sensor_dim, self.hidden_dim)
        self.time_step_embed = nn.Embedding(self.max_time_steps, self.hidden_dim)
        self.seq_position_embed = nn.Embedding(self.max_seq_len, self.hidden_dim)

        # Multi-head attention layers
        self.self_attention = nn.MultiheadAttention(self.hidden_dim, self.num_heads)
        self.cross_attention = nn.MultiheadAttention(self.hidden_dim, self.num_heads)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, 256)

    def forward(self, x):
        # x: (batch_size, seq_len, time_steps, sensors)
        batch_size, seq_len, time_steps, _ = x.size()
    
        # Reshape and embed sensor data
        x = x.view(batch_size * seq_len, time_steps, self.sensor_dim)
        x = self.sensor_embed(x)  # (batch_size * seq_len, time_steps, hidden_dim)
    
        # Add time step embeddings
        time_steps_indices = torch.arange(time_steps, device=x.device).unsqueeze(0).expand(batch_size * seq_len, -1)
        x = x + self.time_step_embed(time_steps_indices)
    
        # Add sequence position embeddings
        seq_positions = torch.arange(seq_len, device=x.device).repeat(batch_size)
        seq_position_embed = self.seq_position_embed(seq_positions)
        seq_position_embed = seq_position_embed.view(batch_size * seq_len, 1, self.hidden_dim)
        x = x + seq_position_embed
    
        # Self-attention within each sequence
        x = x.transpose(0, 1)  # (time_steps, batch_size * seq_len, hidden_dim)
        x = self.norm1(x + self.self_attention(x, x, x)[0])
    
        # Reshape for cross-sequence attention
        x = x.transpose(0, 1).view(batch_size, seq_len, time_steps, self.hidden_dim)
        x = x.transpose(1, 2).reshape(batch_size * time_steps, seq_len, self.hidden_dim)
    
        # Cross-sequence attention
        x = x.transpose(0, 1)  # (seq_len, batch_size * time_steps, hidden_dim)
        x = self.norm2(x + self.cross_attention(x, x, x)[0])
    
        # Feed-forward network
        x = self.norm3(x + self.ffn(x))
    
        # Global average pooling over sequences
        x = x.transpose(0, 1).mean(dim=1)  # (batch_size * time_steps, hidden_dim)
    
        # Reshape and project to output dimension
        x = x.view(batch_size, time_steps, self.hidden_dim)
        x = self.output_proj(x.mean(dim=1))  # (batch_size, 256)
    
        return x.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, 256)


class Inertial_temporal_encoder(nn.Module):
    def __init__(self, opt):
        super(Inertial_temporal_encoder, self).__init__()

        self.sensor_embed = nn.Linear(6, 32)
        self.time_embed = nn.Linear(11, 32)

        self.lstm = nn.LSTM(32, 128, num_layers=2, batch_first=True, bidirectional=True)

        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4)

        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(256)
        )

    def forward(self, x):
        # x: (N, seq_len, 11, 6)
        batch_size, seq_len, time_steps, sensors = x.size()

        # Embed sensor data
        x = x.view(batch_size * seq_len, time_steps, sensors)
        x = self.sensor_embed(x)  # (batch_size * seq_len, time_steps, 32)

        # Embed time information
        time_embed = self.time_embed(torch.eye(time_steps).to(x.device))
        x = x + time_embed.unsqueeze(0)  # Add time embeddings

        # Process with LSTM
        x, _ = self.lstm(x)  # (batch_size * seq_len, time_steps, 256)

        # Self-attention mechanism
        x = x.transpose(0, 1)  # (time_steps, batch_size * seq_len, 256)
        x, _ = self.attention(x, x, x)
        x = x.transpose(0, 1)  # (batch_size * seq_len, time_steps, 256)

        # Global average pooling over time
        x = x.mean(dim=1)  # (batch_size * seq_len, 256)

        # Final fully connected layers
        x = self.fc(x)

        return x.view(batch_size, seq_len, 256)


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        # CNN
        self.opt = opt
        self.conv1 = conv(True, 6, 64, kernel_size=7, stride=2, dropout=0.2)
        self.conv2 = conv(True, 64, 128, kernel_size=5, stride=2, dropout=0.2)
        self.conv3 = conv(True, 128, 256, kernel_size=5, stride=2, dropout=0.2)
        self.conv3_1 = conv(True, 256, 256, kernel_size=3, stride=1, dropout=0.2)
        self.conv4 = conv(True, 256, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv4_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv5 = conv(True, 512, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv5_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv6 = conv(True, 512, 1024, kernel_size=3, stride=2, dropout=0.5)
        # Comput the shape based on diff image size
        __tmp = Variable(torch.zeros(1, 6, opt.img_w, opt.img_h))
        __tmp = self.encode_image(__tmp)

        self.visual_head = nn.Linear(int(np.prod(__tmp.size())), opt.v_f_len)
        self.inertial_encoder = Inertial_encoder(opt)
        # self.inertial_encoder = Inertial_temporal_encoder(opt)
        # self.inertial_encoder = image_Inertial_Encoder(opt)
        # self.inertial_encoder = CrossSequenceAttentionEncoder(opt)

    def forward(self, img, imu):
        v = torch.cat((img[:, :-1], img[:, 1:]), dim=2)
        batch_size = v.size(0)
        seq_len = v.size(1)

        # image CNN
        v = v.view(batch_size * seq_len, v.size(2), v.size(3), v.size(4))
        v = self.encode_image(v)
        v = v.view(batch_size, seq_len, -1)  # (batch, seq_len, fv)
        v = self.visual_head(v)  # (batch, seq_len, 256)
        
        # IMU CNN
        imu = torch.cat([imu[:, i * 10:i * 10 + 11, :].unsqueeze(1) for i in range(seq_len)], dim=1)
        imu = self.inertial_encoder(imu) # 10 X sequences with seq_len =11
        return v, imu

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6

import torch
import torch.nn as nn
import torch.nn.functional as F

class Fusion_module(nn.Module):
    def __init__(self, opt):
        super(Fusion_module, self).__init__()
        self.fuse_method = opt.fuse_method
        self.i_f_len = opt.i_f_len
        self.v_f_len = opt.v_f_len
        self.f_len = self.i_f_len + self.v_f_len
        
        if self.fuse_method == 'soft':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, self.f_len))
        elif self.fuse_method == 'hard':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, 2 * self.f_len))
        elif self.fuse_method == 'enhanced':
            self.attention = nn.MultiheadAttention(embed_dim=self.f_len, num_heads=4)
            self.gate = nn.Sequential(
                nn.Linear(self.f_len * 2, self.f_len),
                nn.Sigmoid()
            )
            self.fusion_net = nn.Sequential(
                nn.Linear(self.f_len * 2, self.f_len),
                nn.ReLU(),
                nn.Linear(self.f_len, self.f_len)
            )

    def forward(self, v, i):
        if self.fuse_method == 'cat':
            return torch.cat((v, i), -1)
        elif self.fuse_method == 'soft':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            return feat_cat * weights
        elif self.fuse_method == 'hard':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            weights = weights.view(v.shape[0], v.shape[1], self.f_len, 2)
            mask = F.gumbel_softmax(weights, tau=1, hard=True, dim=-1)
            return feat_cat * mask[:, :, :, 0]
        elif self.fuse_method == 'enhanced':
            feat_cat = torch.cat((v, i), -1)
            
            # Apply self-attention
            attn_out, _ = self.attention(feat_cat.transpose(0, 1), feat_cat.transpose(0, 1), feat_cat.transpose(0, 1))
            attn_out = attn_out.transpose(0, 1)
            
            # Compute gating weights
            gate_weights = self.gate(torch.cat((feat_cat, attn_out), dim=-1))
            
            # Apply gating
            gated_features = feat_cat * gate_weights + attn_out * (1 - gate_weights)
            
            # Final fusion
            fused = self.fusion_net(torch.cat((gated_features, feat_cat), dim=-1))
            
            return fused


# The policy network module
class PolicyNet(nn.Module):
    def __init__(self, opt):
        super(PolicyNet, self).__init__()
        in_dim = opt.rnn_hidden_size + opt.i_f_len
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 2))

    def forward(self, x, temp):
        logits = self.net(x)
        hard_mask = F.gumbel_softmax(logits, tau=temp, hard=True, dim=-1)
        return logits, hard_mask

# The pose estimation network
class Pose_RNN(nn.Module):
    def __init__(self, opt):
        super(Pose_RNN, self).__init__()

        # The main RNN network
        f_len = opt.v_f_len + opt.i_f_len
        self.rnn = nn.LSTM(
            input_size=f_len,
            hidden_size=opt.rnn_hidden_size,
            num_layers=2,
            dropout=opt.rnn_dropout_between,
            batch_first=True)

        self.fuse = Fusion_module(opt)

        # The output networks
        self.rnn_drop_out = nn.Dropout(opt.rnn_dropout_out)
        self.regressor = nn.Sequential(
            nn.Linear(opt.rnn_hidden_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 6))

    def forward(self, fv, fv_alter, fi, prev=None):
        if prev is not None:
            prev = (prev[0].transpose(1, 0).contiguous(), prev[1].transpose(1, 0).contiguous())
        
        # Select between fv and fv_alter
        # v_in = fv * dec[:, :, :1] + fv_alter * dec[:, :, -1:] if fv_alter is not None else fv
        fused = self.fuse(fv, fi)
        
        out, hc = self.rnn(fused) if prev is None else self.rnn(fused, prev)
        out = self.rnn_drop_out(out)
        pose = self.regressor(out)

        hc = (hc[0].transpose(1, 0).contiguous(), hc[1].transpose(1, 0).contiguous())
        return pose, hc



# shlomia alternative pose transformer
import torch
import torch.nn as nn
import math


class HybridPoseNetwork(nn.Module):
    def __init__(self, opt):
        super(HybridPoseNetwork, self).__init__()

        f_len = opt.v_f_len + opt.i_f_len
        self.hidden_dim = opt.rnn_hidden_size
        self.num_heads = 8

        # Input projection
        self.input_proj = nn.Linear(f_len, self.hidden_dim)

        # Transformer for feature extraction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=opt.rnn_dropout_between,
            activation='relu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1  # Single transformer layer as we'll use LSTM after
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim, opt.rnn_dropout_between)

        # LSTM for sequential processing
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            dropout=opt.rnn_dropout_between,
            batch_first=True
        )

        # Keep original components
        self.fuse = Fusion_module(opt)
        self.rnn_drop_out = nn.Dropout(opt.rnn_dropout_out)
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 6)
        )

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, fv, fv_alter, fi, dec, prev=None):
        # Handle input fusion like original
        v_in = fv * dec[:, :, :1] + fv_alter * dec[:, :, -1:] if fv_alter is not None else fv
        fused = self.fuse(v_in, fi)

        # Project input
        x = self.input_proj(fused)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Generate transformer mask
        trans_mask = self._generate_square_subsequent_mask(x.size(1)).to(x.device)

        # Transformer processing for feature extraction
        trans_out = self.transformer(x, mask=trans_mask)

        # Handle LSTM states like original RNN
        if prev is not None:
            prev = (prev[0].transpose(1, 0).contiguous(),
                    prev[1].transpose(1, 0).contiguous())

        # LSTM processing for sequential modeling
        lstm_out, hc = self.lstm(trans_out) if prev is None else self.lstm(trans_out, prev)

        # Process output
        out = self.rnn_drop_out(lstm_out)
        pose = self.regressor(out)

        # Format hidden states like original RNN
        hc = (hc[0].transpose(1, 0).contiguous(),
              hc[1].transpose(1, 0).contiguous())

        return pose, hc


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class DeepVIO(nn.Module):
    def __init__(self, opt):
        super(DeepVIO, self).__init__()

        self.Feature_net = Encoder(opt)
        self.Pose_net = Pose_RNN(opt)
        # /self.Pose_net = HybridPoseNetwork(opt) # shlomia change
        # self.Policy_net = PolicyNet(opt)
        self.opt = opt
        initialization(self)

    def forward(self, img, imu, is_first=True, hc=None, temp=5, selection='gumbel-softmax', p=0.5):

        fv, fi = self.Feature_net(img, imu)
        batch_size = fv.shape[0]
        seq_len = fv.shape[1]

        poses, decisions, logits= [], [], []
        hidden = torch.zeros(batch_size, self.opt.rnn_hidden_size).to(fv.device) if hc is None else hc[0].contiguous()[:, -1, :]
        fv_alter = torch.zeros_like(fv) # zero padding in the paper, can be replaced by other 
        
        for i in range(seq_len):
            if i == 0 and is_first:
                # The first relative pose is estimated by both images and imu by default
                pose, hc = self.Pose_net(fv[:, i:i+1, :], None, fi[:, i:i+1, :], hc)
            else:
                pose, hc = self.Pose_net(fv[:, i:i + 1, :], fv_alter[:, i:i + 1, :], fi[:, i:i + 1, :], hc)
            poses.append(pose)
            hidden = hc[0].contiguous()[:, -1, :]

        poses = torch.cat(poses, dim=1)

        return poses, hc


def initialization(net):
    #Initilization
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(0)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    start, end = n//4, n//2
                    param.data[start:end].fill_(1.)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
