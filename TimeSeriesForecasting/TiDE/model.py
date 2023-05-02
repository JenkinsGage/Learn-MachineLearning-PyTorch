import torch
import torch.nn as nn

class ResMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super(ResMLP, self).__init__()
        self.linear0 = nn.Linear(in_dim, out_dim)

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.LeakyReLU()
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        b1 = self.linear1(x)
        b1 = self.activation(b1)
        b1 = self.linear2(b1)
        b1 = self.dropout(b1)
        b2 = self.linear0(x)
        y = b1 + b2
        y = self.norm(y)
        return y


class TiDE(nn.Module):
    def __init__(self, lookback_steps: int, horizon_steps: int, static_attributes_dim: int, dynamic_covariates_dim: int, dynamic_covariates_projection_dim: int, hidden_dim: int, num_encoder_layers: int, num_decoder_layers: int, decoder_output_dim: int, temporal_decoder_hidden_dim: int, dropout: float = 0.1):
        super(TiDE, self).__init__()
        self.lookback_steps = lookback_steps
        self.horizon_steps = horizon_steps
        self.static_attributes_dim = static_attributes_dim
        self.dynamic_covariates_dim = dynamic_covariates_dim
        self.dynamic_covariates_projection_dim = dynamic_covariates_projection_dim
        self.hidden_dim = hidden_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.decoder_output_dim = decoder_output_dim
        self.temporal_decoder_hidden_dim = temporal_decoder_hidden_dim
        self.dropout = dropout

        self.dynamic_covariates_feature_projection = ResMLP(in_dim=self.dynamic_covariates_dim, hidden_dim=self.hidden_dim, out_dim=self.dynamic_covariates_projection_dim)
        self.encoder_input_dim = lookback_steps + static_attributes_dim + (lookback_steps + horizon_steps) * dynamic_covariates_projection_dim
        self.encoders = nn.Sequential(*[ResMLP(in_dim=self.encoder_input_dim if _ == 0 else self.hidden_dim, hidden_dim=self.hidden_dim, out_dim=self.hidden_dim, dropout=self.dropout) for _ in range(self.num_encoder_layers)])
        self.decoder_output_features = self.horizon_steps * self.decoder_output_dim
        self.decoders = nn.Sequential(*[ResMLP(in_dim=self.hidden_dim if _==0 else self.decoder_output_features, hidden_dim=self.hidden_dim, out_dim=self.decoder_output_features, dropout=self.dropout) for _ in range(self.num_decoder_layers)])
        self.temporal_decoder_input_dim = self.decoder_output_dim + self.dynamic_covariates_projection_dim
        self.temporal_decoder = ResMLP(in_dim=self.temporal_decoder_input_dim, hidden_dim=self.temporal_decoder_hidden_dim, out_dim=1, dropout=self.dropout)
        self.global_residual_connection = nn.Linear(self.lookback_steps, self.horizon_steps, bias=False)

    def forward(self, lookback: torch.Tensor, static_attributes: torch.Tensor, dynamic_covariates: torch.Tensor):
        dynamic_covariates_projection = self.dynamic_covariates_feature_projection(dynamic_covariates)
        encoder_input = torch.cat([lookback, static_attributes, torch.flatten(dynamic_covariates_projection, start_dim=1)], dim=-1)
        embedding = self.encoders(encoder_input)
        g = self.decoders(embedding)
        D = g.view(-1, self.horizon_steps, self.decoder_output_dim)
        temporal_decoder_input = torch.cat([D, dynamic_covariates_projection[:, -self.horizon_steps:, ]], dim=-1)
        y = self.temporal_decoder(temporal_decoder_input).squeeze(-1)
        y += self.global_residual_connection(lookback)
        return y
    
class LogNorm(nn.Module):
    def __init__(self, eps=1e-9):
        super(LogNorm, self).__init__()
        self.eps = eps
        self.mean = 0
        self.std = 1
        
    def forward(self, x):
        x = torch.log(1+x)
        self.mean = torch.mean(x, dim=-1, keepdim=True)
        self.std = torch.std(x, dim=-1, keepdim=True)
        return (x-self.mean)/(self.std+self.eps)
    
    def inverse(self, x):
        return torch.exp(x*self.std+self.mean)-1
    
    def normalize(self, x):
        x = torch.log(1+x)
        return (x-self.mean)/(self.std+self.eps)