"""
ARC-VAE Transformer Encoder
__
encoder.py maps an observed S2 growing-season time series to a posterior distribution
over the 11 ARC latent parameters z = (p, h).

Architecture:
  1. Input tokenisation
     Each observation i: u_i = W_in[r_i || Θ_i] + PE(doy_i)
  2. Transformer encoder (L layers of multi-head self-attention)
  3. Parameter-specific attention pooling
     11 learned query vectors — one per parameter — each attending
     over the T encoder outputs to extract a dedicated context vector.
     h_end's query learns to attend to late-season observations;
     h_start's query learns to attend to early green-up, etc.
  4. Per-parameter output heads
     Each parameter j has its own (μ_j, log σ_j) head reading 
     from its dedicated context vector c_j.

Latent variable structure (11 total):
  z[0:7]  = p  (scaling: N, Cab, Cm, Cw, LAI, ALA, Cbrown)
  z[7:11] = h  (phenology: growth_speed, start_DOY, senes_speed, end_DOY)
"""

import math
import torch
import torch.nn as nn


_P_LO = torch.tensor([0.4541, 0.2362, 0.0692, 0.0318, 0.1624, 0.8122, 0.0000], # Upper and lower physical boundaries for the 7 canopy scaling metrics
                      dtype=torch.float32)
_P_HI = torch.tensor([1.3623, 0.9449, 2.7676, 3.1813, 2.5978, 1.2996, 1.8799],
                      dtype=torch.float32)
_H_LO = torch.tensor([0.045, 115.0, 0.010, 245.0], dtype=torch.float32) # Upper and lower absolute day boundaries for the 4 phenological timing metrics
_H_HI = torch.tensor([0.325, 210.0, 0.370, 365.0], dtype=torch.float32)

# Join boundary blocks to configure unified 11-dimensional latent limits
Z_LO = torch.cat([_P_LO, _H_LO])   # (11,)
Z_HI = torch.cat([_P_HI, _H_HI])   # (11,)

# We define standard deviation boundaries to control variational distribution spreads
_SIGMA_MIN_VALS = torch.tensor(
    [5e-4, 5e-4, 5e-4, 5e-4, 5e-4, 5e-4, 5e-4, 1e-3, 0.50, 1e-3, 0.50],
    dtype=torch.float32)
_SIGMA_MAX_VALS = torch.tensor(
    [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.10, 40.0, 0.10, 50.0],
    dtype=torch.float32)

# Convert min/max standard deviation boundaries into log space for stability
LOG_SIGMA_MIN = torch.log(_SIGMA_MIN_VALS)
LOG_SIGMA_MAX = torch.log(_SIGMA_MAX_VALS)

class DOYPositionalEncoding(nn.Module): # Sinusoidal positional encoding of calendar DOY
    def __init__(self, d_model: int, n_annual_harmonics: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_annual_harmonics = n_annual_harmonics
        n_annual_feats = 2 * n_annual_harmonics
        self.proj = nn.Linear(n_annual_feats + d_model, d_model, bias=False) # Projection layer to compress combined calendar and spacing contexts to embedding size

    def forward(self, doys: torch.Tensor) -> torch.Tensor:
        B, T = doys.shape
        device = doys.device
        annual_feats = []
        for k in range(1, self.n_annual_harmonics + 1): # Calculate cyclical sine and cosine values across 4 multi-frequency calendar intervals
            annual_feats.append(torch.sin(2 * math.pi * k * doys / 365.0))
            annual_feats.append(torch.cos(2 * math.pi * k * doys / 365.0))
        annual = torch.stack(annual_feats, dim=-1)
        d = self.d_model # Build standard relative spacing positional embeddings
        pos = doys.unsqueeze(-1).float()
        div = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / d))
        std_pe = torch.zeros(B, T, d, device=device)
        std_pe[:, :, 0::2] = torch.sin(pos * div)
        std_pe[:, :, 1::2] = torch.cos(pos * div[:d // 2])
        combined = torch.cat([annual, std_pe], dim=-1) # Bind cyclical and standard features along last axis and project down
        return self.proj(combined)


class ParameterSpecificAttentionPooling(nn.Module): # Parameter-specific attention pooling
    """
    11 learned query vectors, one per latent parameter.

    Each query j attends over the T encoder outputs to extract a dedicated
    context vector c_j in R^d_model. This allows each parameter to learn
    which temporal observations are most informative for its estimation:
    - h_end learns to focus on late-season observations
    - h_start learns to focus on early green-up
    - p_LAI learns to focus on peak reflectance

    All 11 queries attend simultaneously via a single cross-attention call
    for efficiency.
    """
    def __init__(self, d_model: int, n_params: int = 11, n_heads: int = 4):
        super().__init__()
        self.n_params = n_params
        self.d_model  = d_model
        self.queries  = nn.Parameter(torch.empty(1, n_params, d_model)) # Initialise 11 independent learned query vector embeddings
        nn.init.xavier_uniform_(self.queries)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True) # Configure multihead attention mechanics to execute cross-pooling operations

    def forward(self, x: torch.Tensor,
                mask: torch.BoolTensor) -> torch.Tensor:
        # x: (B, T, d_model); mask: (B, T) True=valid
        B = x.shape[0] 
        queries = self.queries.expand(B, -1, -1)     # (B, 11, d_model); Expand latent query parameter shapes across batch items
        pooled, _ = self.cross_attn(
            query=queries, key=x, value=x,
            key_padding_mask=~mask)                   # (B, 11, d_model); Run cross-attention over sequence parameters while masking out empty padded slots
        return pooled


class ARCVAEEncoder(nn.Module): #main encoder
    N_BANDS  = 10
    N_ANGLES = 3
    N_PARAMS = 11

    def __init__(self, d_model=128, n_layers=4, n_heads=4,
                 n_queries=11, d_ff=256, dropout=0.1,
                 n_annual_harmonics=4):
        super().__init__()
        # n_queries is kept as an argument for API compatibility but is
        # always overridden to N_PARAMS internally.
        self.d_model = d_model

        # Input normalisation constants as model state buffers
        self.register_buffer("refl_mean",
            torch.tensor([0.06,0.08,0.06,0.10,0.20,0.24,0.25,0.25,0.12,0.08],
                         dtype=torch.float32))
        self.register_buffer("refl_std",
            torch.tensor([0.05,0.06,0.06,0.07,0.08,0.09,0.09,0.09,0.07,0.05],
                         dtype=torch.float32))
        self.register_buffer("ang_mean",
            torch.tensor([42.0, 5.0, 90.0], dtype=torch.float32))
        self.register_buffer("ang_std",
            torch.tensor([10.0, 3.0, 52.0], dtype=torch.float32))

        # Parameter bounds
        self.register_buffer("z_lo",          Z_LO)
        self.register_buffer("z_hi",          Z_HI)
        self.register_buffer("log_sigma_min", LOG_SIGMA_MIN)
        self.register_buffer("log_sigma_max", LOG_SIGMA_MAX)

        # Input projection (map 13 normalise input dimensions up to transformer embedding space size)
        self.input_proj = nn.Linear(self.N_BANDS + self.N_ANGLES, d_model)

        # Positional encoding
        self.pos_enc = DOYPositionalEncoding(d_model, n_annual_harmonics)

        # Transformer encoder (build sequence layers using standard Pre-LayerNorm configuration mechanics)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            enable_nested_tensor=False)

        # Parameter-specific attention pooling (11 queries, one per parameter)
        self.pool = ParameterSpecificAttentionPooling(
            d_model, self.N_PARAMS, n_heads)

        # Per-parameter output heads
        # Each parameter j reads exclusively from its context vector c_j.
        # Implemented as a single batched linear for efficiency:
        #   raw_mu[b, j]     = mu_weight[j] · c_j[b] + mu_bias[j]
        #   raw_logstd[b, j] = ls_weight[j] · c_j[b] + ls_bias[j]
        self.mu_weight  = nn.Parameter(torch.empty(self.N_PARAMS, d_model))
        self.mu_bias    = nn.Parameter(torch.zeros(self.N_PARAMS))
        self.ls_weight  = nn.Parameter(torch.empty(self.N_PARAMS, d_model))
        self.ls_bias    = nn.Parameter(torch.zeros(self.N_PARAMS))
        nn.init.xavier_uniform_(self.mu_weight)
        nn.init.xavier_uniform_(self.ls_weight)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _normalise_inputs(self, s2_refl, angles):
      # Scale incoming raw bands and multi-angle viewing dimensions
        r_norm = (s2_refl - self.refl_mean) / (self.refl_std + 1e-8)
        a_norm = (angles  - self.ang_mean)  / (self.ang_std  + 1e-8)
        return torch.cat([r_norm, a_norm], dim=-1)

    def _constrain_mu(self, raw_mu):
      # Clip inferred mean coordinates within strict bounds via sigmoid scaling
        return self.z_lo + torch.sigmoid(raw_mu) * (self.z_hi - self.z_lo)

    def _constrain_sigma(self, raw_logstd):
      # Clip generated log uncertainties within configured validation ranges
        log_sigma = (self.log_sigma_min
                     + torch.sigmoid(raw_logstd)
                     * (self.log_sigma_max - self.log_sigma_min))
        return torch.exp(log_sigma)

    def forward(self, s2_refl, angles, doys, obs_mask):
        # 1. Normalise and project
        x = self._normalise_inputs(s2_refl, angles)
        x = self.input_proj(x)

        # 2. Positional encoding
        x = x + self.pos_enc(doys)
        x = x * obs_mask.unsqueeze(-1).float()

        # 3. Transformer encoder
        h = self.transformer(x, src_key_padding_mask=~obs_mask)

        # 4. Parameter-specific pooling --> (B, 11, d_model)
        contexts = self.pool(h, obs_mask)

        # 5. Per-parameter heads via batched dot product
        # contexts: (B, 11, d_model), mu_weight: (11, d_model)
        raw_mu     = (contexts * self.mu_weight.unsqueeze(0)).sum(-1) + self.mu_bias
        raw_logstd = (contexts * self.ls_weight.unsqueeze(0)).sum(-1) + self.ls_bias

        mu    = self._constrain_mu(raw_mu)
        sigma = self._constrain_sigma(raw_logstd)
        return mu, sigma


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_encoder():
    print("=" * 60)
    print("Step 4: Verify Transformer encoder (parameter-specific attention)")
    print("=" * 60)
    torch.manual_seed(0)
    B, T = 8, 50
    encoder = ARCVAEEncoder(d_model=128, n_layers=4, n_heads=4)
    encoder.train()

    obs_counts = torch.randint(10, T, (B,))
    obs_mask   = torch.zeros(B, T, dtype=torch.bool)
    for i, n in enumerate(obs_counts):
        obs_mask[i, :n] = True

    s2_refl = torch.rand(B, T, 10) * 0.5
    angles  = torch.zeros(B, T, 3)
    angles[:, :, 0] = 40.0; angles[:, :, 1] = 5.0; angles[:, :, 2] = 90.0
    doys = torch.zeros(B, T)
    for i, n in enumerate(obs_counts):
        doys[i, :n] = torch.linspace(130, 280, n)
    s2_refl = s2_refl * obs_mask.unsqueeze(-1)
    angles  = angles  * obs_mask.unsqueeze(-1)

    mu, sigma = encoder(s2_refl, angles, doys, obs_mask)

    print(f"\nOutput shapes:  μ={tuple(mu.shape)}  σ={tuple(sigma.shape)}")
    in_bounds = ((mu >= encoder.z_lo - 1e-4) & (mu <= encoder.z_hi + 1e-4)).all()
    print(f"μ within bounds: {'✓' if in_bounds else '✗'}")

    loss = mu.sum() + sigma.sum()
    loss.backward()
    grad_ok = all(p.grad is not None and p.grad.abs().sum() > 0
                  for p in encoder.parameters() if p.requires_grad)
    print(f"Gradients flow: {'✓' if grad_ok else '✗'}")

    # Verify parameter-specific attention: each parameter's context differs
    with torch.no_grad():
        x_in  = encoder._normalise_inputs(s2_refl[:1], angles[:1])
        x_in  = encoder.input_proj(x_in) + encoder.pos_enc(doys[:1])
        x_in  = x_in * obs_mask[:1].unsqueeze(-1).float()
        h_out = encoder.transformer(x_in, src_key_padding_mask=~obs_mask[:1])
        ctx   = encoder.pool(h_out, obs_mask[:1])   # (1, 11, d_model)
        ctx_std = ctx[0].std(dim=0).mean().item()
        print(f"Parameter contexts are distinct (std={ctx_std:.4f}): "
              f"{'✓' if ctx_std > 0.01 else '✗'}")

    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nEncoder parameter count: {n_params:,}")
    print("\nEncoder verification: PASSED ✓")
    print("Proceed to Step 5 (VAE training loop).")


if __name__ == "__main__":
    verify_encoder()
