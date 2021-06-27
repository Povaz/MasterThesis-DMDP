import torch
import torch.nn as nn
import numpy as np
from utils.RECOEncoderCore import PositionalEncoding


class ReconstructionNetwork(nn.Module):

    def __init__(self, env, encoder_dim, encoder_heads, encoder_ff_hid, encoder_layers, dropout=0.1,
                 rescaling=False, normalized=False, batchnorm=False, mask=False,
                 transformer_decoder=False, append_pos_encoding=False, pendulum_state=False):
        super(ReconstructionNetwork, self).__init__()
        self.state_dim = env.state_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        if append_pos_encoding:
            self.encoder_dim = 2*encoder_dim
        else:
            self.encoder_dim = encoder_dim
        self.encoder_ff_hid = encoder_ff_hid
        self.encoder_layers = encoder_layers
        self.encoder_heads = encoder_heads

        # Boolean Options
        self.rescaling = rescaling
        self.normalized = normalized
        self.batchnorm = batchnorm
        self.transformer_decoder = transformer_decoder
        self.mask = mask
        self.input_mask = None
        self.pendulum_state = pendulum_state
        self.env_name = env.__class__.__name__
        if self.rescaling or self.normalized or self.batchnorm:
            self.low_state_value = env.state_space.low
            self.high_state_value = env.state_space.high
            self.low_action_value = env.action_space.low
            self.high_action_value = env.action_space.high
            self.action_scaling = torch.from_numpy(np.max((self.high_action_value, -self.low_action_value), 0)).reshape(1, 1, -1)
            self.state_scaling = torch.from_numpy(np.max((self.high_state_value, -self.low_state_value), 0)).reshape(1, 1, -1)
            

        # Inverted Pendulum Environment: State Reconstruction
        # WARNING: this option only works with Inverted Pendulum Environment and it is implemented here
        # in order to not modify the Environment Code
        # The purpose is to restore the original State of the Inverted Pendulum Environment, retrieve the actual
        # Angle of the Pendulum from Sin and Cos and use it as State for the Encoder.
        if self.env_name == 'PendulumDelayEnv' and self.pendulum_state:
            self.state_dim = 2
            self.state_scaling = torch.from_numpy(np.array([np.pi, env.max_speed])).reshape(1, 1, -1)

        # Input Mapping (Embedding)
        self.input_mapping = nn.Linear(self.state_dim + self.action_dim, encoder_dim, bias=True)

        # Positional Encoding & Normalization Layer
        self.positional_encoding = PositionalEncoding(encoder_dim, max_len=50, append=append_pos_encoding)
        if self.batchnorm:  
            self.batchnorm_layer = nn.BatchNorm1d(self.encoder_dim, eps=1e-5)
        else:
            self.norm_layer = nn.LayerNorm(self.encoder_dim, eps=1e-5)
        self.dropout = nn.Dropout(p=dropout)

        # Transformer Encoders
        encoder_layer = nn.TransformerEncoderLayer(self.encoder_dim, nhead=encoder_heads,
                                                   dim_feedforward=encoder_ff_hid,
                                                   dropout=dropout, activation='relu')
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=encoder_layers, norm=None)
        self.relu = nn.ReLU()

        # --------- Decoder ---------
        if self.transformer_decoder:
            self.hidden_layer = nn.Linear(self.state_dim, self.encoder_dim, bias=True)
            decoder_layer = nn.TransformerDecoderLayer(d_model=self.encoder_dim, nhead=encoder_heads,
                                                       dim_feedforward=self.encoder_ff_hid, dropout=dropout,
                                                       activation='relu')
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=encoder_layers, norm=None)
        else:
            self.hidden_layer = nn.Linear(self.encoder_dim, self.encoder_dim, bias=True)
        if self.normalized:
            self.hidden_norm_layer = nn.LayerNorm(self.encoder_dim)
        self.prediction_layer = nn.Linear(self.encoder_dim, self.state_dim, bias=True)

    @staticmethod
    def _generate_square_subsequent_mask(sz):
        mask = torch.tril(torch.ones(sz, sz))
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    @staticmethod
    def _retrieve_pendulum_state(state):
        angles = torch.where(state[:, 0, 1] > 0.0, torch.acos(state[:, 0, :1]), -torch.acos(state[:, 0, :1]))[:, 1]
        state = torch.cat((angles.unsqueeze(dim=-1), state[:, 0, 2].unsqueeze(dim=-1)), dim=-1)
        return state.unsqueeze(dim=1)

    @staticmethod
    def _restore_pendulum_state(pred):
        cos = torch.cos(pred[:, :, :1])
        sin = torch.sin(pred[:, :, 1:2])
        pred = torch.cat((cos, sin, pred[:, :, :1]), dim=-1)
        return pred

    def forward(self, x, decoder_input=None):
        # ---- INPUT SHAPING ---- #
        # Inverted Pendulum Environment: State Reconstruction
        if self.env_name == 'PendulumDelayEnv' and self.pendulum_state:
            self.state_dim = 3

        state = x[:, :self.state_dim].reshape(x.size(0), 1, self.state_dim)
        action = x[:, self.state_dim:].reshape(x.size(0), -1, self.action_dim)

        # Inverted Pendulum Environment: State Reconstruction
        if self.env_name == 'PendulumDelayEnv' and self.pendulum_state:
            state = self._retrieve_pendulum_state(state)
            self.state_dim = 2

        if self.rescaling or self.normalized or self.batchnorm:
            state = state / self.state_scaling
            action = action / self.action_scaling
        x = state.repeat(1, action.size(1), 1)
        x = torch.cat((x, action), dim=2)

        # ---- COMPUTATIONS ---- #
        # Mask of the input
        if self.mask:
            device = x.device
            if self.input_mask is None or self.input_mask.size(0) != x.size(1):
                self.input_mask = self._generate_square_subsequent_mask(x.size(1)).to(device)
        else:
            self.input_mask = None

        # Embedding & Positional Encoding
        x = self.input_mapping(x)
        x = self.relu(x)
        x = self.positional_encoding(x.transpose(0, 1))
        if not self.batchnorm:
            x = self.norm_layer(x)
        else:
            x = self.batchnorm_layer(x.transpose(1, 2))
            x = x.transpose(1, 2)
        
        # Encoder Self-Attention
        encoded_state = self.encoder(x, mask=self.input_mask)
        
        if not self.transformer_decoder:
            encoded_state = encoded_state.transpose(0, 1)
            # Output Prediction
            if self.normalized:
                encoded_state = self.hidden_layer(encoded_state)
                encoded_state = self.hidden_norm_layer(encoded_state)
                pred = self.relu(encoded_state)
            else:
                encoded_state = self.hidden_layer(encoded_state)
                pred = self.relu(encoded_state)
        else:
            # if condition in case of test
            if decoder_input is None:
                decoder_input = torch.zeros(x.size(0), x.size(1), self.state_dim)
                decoder_input[0, :, :] = state.transpose(0, 1)
                for i in range(x.size(0) - 1):
                    temp_decoder_input = self.hidden_layer(decoder_input)
                    temp_decoder_input = self.relu(temp_decoder_input)
                    pred_temp = self.decoder(encoded_state, temp_decoder_input, tgt_mask=self.input_mask,
                                             memory_mask=self.input_mask)
                    pred_temp = pred_temp.transpose(0, 1)
                    pred_temp = self.prediction_layer(pred_temp)
                    decoder_input[i + 1, :, :] = pred_temp[:, i, :]
            else: 
                decoder_input = decoder_input.reshape(encoded_state.size(1), -1, self.state_dim).transpose(0, 1)

            # For test and train
            decoder_input = self.hidden_layer(decoder_input)
            decoder_input = self.relu(decoder_input)
            pred = self.decoder(encoded_state, decoder_input, tgt_mask=self.input_mask, memory_mask=self.input_mask)
            pred = pred.transpose(0, 1)

        pred = self.prediction_layer(pred)

        # Inverted Pendulum Environment: State Reconstruction
        if self.env_name == 'PendulumDelayEnv' and self.pendulum_state:
            pred = self._restore_pendulum_state(pred)
            self.state_dim = 3

        if self.rescaling:
            pred = pred * self.state_scaling

        return pred
