"""PreDiff model and diffusion components."""

import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps.unsqueeze(-1).float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ---------------------------------------------------------------------------
# Beta schedules
# ---------------------------------------------------------------------------

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        if num_diffusion_timesteps < 24:
            # Subsample from the standard 1000-step linear schedule's alpha_bar curve
            betas_ref = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
            return _betas_from_alpha_bar_subsample(betas_ref, num_diffusion_timesteps)
        else:
            scale = 1000 / num_diffusion_timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return _betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == 'sqrt':
        return _betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1 - np.sqrt(t + 0.0001),
        )
    elif schedule_name == 'trunc_lin':
        if num_diffusion_timesteps < 24:
            betas_ref = np.linspace(0.0001 + 0.01, 0.02 + 0.01, 1000, dtype=np.float64)
            return _betas_from_alpha_bar_subsample(betas_ref, num_diffusion_timesteps)
        else:
            scale = 1000 / num_diffusion_timesteps
            beta_start = scale * 0.0001 + 0.01
            beta_end = scale * 0.02 + 0.01
            if beta_end > 1:
                beta_end = scale * 0.001 + 0.01
            return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == 'trunc_cos':
        return _betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: max(math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2 - 0.01, 1e-8),
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def _betas_from_alpha_bar_subsample(betas_ref, num_diffusion_timesteps):
    """Subsample a reference beta schedule by interpolating its alpha_bar curve."""
    alphas = 1.0 - betas_ref
    alpha_bar = np.cumprod(alphas)
    alpha_bar = np.concatenate([[1.0], alpha_bar])
    indices = np.linspace(0, len(betas_ref), num_diffusion_timesteps + 1)
    alpha_bar_sub = np.interp(indices, np.arange(len(alpha_bar)), alpha_bar)
    betas = 1.0 - (alpha_bar_sub[1:] / alpha_bar_sub[:-1])
    return np.clip(betas, 0, 0.999)


def _betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / max(alpha_bar(t1), 1e-10), max_beta))
    return np.clip(np.array(betas), 1e-8, max_beta)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


# ---------------------------------------------------------------------------
# Uniform timestep sampler
# ---------------------------------------------------------------------------

class UniformSampler:
    def __init__(self, diffusion):
        self._weights = np.ones([diffusion.num_timesteps])

    def sample(self, batch_size, device):
        probabilities = self._weights / np.sum(self._weights)
        indices_np = np.random.choice(len(probabilities), size=(batch_size,), p=probabilities)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(probabilities) * probabilities[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


# ---------------------------------------------------------------------------
# Gaussian Diffusion
# ---------------------------------------------------------------------------

class GaussianDiffusion:
    def __init__(self, *, betas):
        betas = np.array(betas, dtype=np.float64)
        assert len(betas.shape) == 1
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def _scale_timesteps(self, t):
        return t.float() * (1000.0 / self.num_timesteps)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        return posterior_mean, posterior_variance

    def p_mean_variance(self, model, x, t, condition, all_item_emb):
        assert t.shape == (x.size(0),)
        model_output = model(x, self._scale_timesteps(t), condition, all_item_emb)

        model_mean, model_variance = self.q_posterior_mean_variance(x_start=model_output, x_t=x, t=t)
        return {"mean": model_mean, "variance": model_variance}

    def p_sample(self, model, x, t, condition, all_item_emb):
        out = self.p_mean_variance(model, x, t, condition, all_item_emb)
        noise = th.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        return out["mean"] + nonzero_mask * th.sqrt(out["variance"]) * noise

    def p_sample_loop(self, model, shape, condition, all_item_emb, noise=None):
        device = next(model.parameters()).device
        if noise is not None:
            sample_x = noise
        else:
            sample_x = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        with th.no_grad():
            for i in indices:
                t = th.tensor([i] * shape[0], device=device)
                sample_x = self.p_sample(model, sample_x, t, condition, all_item_emb)
        encoder_logits = th.matmul(condition, all_item_emb.t())
        return sample_x + encoder_logits


# ---------------------------------------------------------------------------
# PreDiff Encoder
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_attention_heads=4, dropout_prob=0.5):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_attention_heads,
                                               dropout=dropout_prob, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)

    def forward(self, x):
        att_out, _ = self.attention(x, x, x)
        x = x + self.dropout1(att_out)
        x = self.norm1(x)
        ffn_out = self.ffn(x)
        x = x + self.dropout2(ffn_out)
        x = self.norm2(x)
        return x


class Encoder(nn.Module):
    """Encode history items into a single embedding."""

    def __init__(self, hidden_dim, num_encoder_layers, encoder_seq_len,
                 item_size, num_attention_heads=4, dropout_prob=0.5):
        super().__init__()
        self.encoder_seq_len = encoder_seq_len
        self.item_size = item_size + 1  # +1 for padding

        self.position_emb = nn.Embedding(encoder_seq_len, hidden_dim)
        self.item_emb = nn.Embedding(self.item_size, hidden_dim)
        self.encoder = nn.ModuleList([
            EncoderLayer(hidden_dim, num_attention_heads, dropout_prob)
            for _ in range(num_encoder_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight.data, a=0.01, nonlinearity="leaky_relu")
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_()

    def get_all_item_emb(self):
        return self.item_emb.weight

    def forward(self, history):
        padding_mask = th.ne(history, 0)
        last_position = th.sum(padding_mask, dim=1).clamp(min=1).long() - 1

        emb_pos = self.position_emb(th.arange(self.encoder_seq_len, device=history.device))
        history_emb = self.item_emb(history)
        history_emb = th.where(padding_mask.unsqueeze(-1), history_emb, th.zeros_like(history_emb))
        encoder_inputs = history_emb + emb_pos.unsqueeze(0)

        x = self.dropout(self.norm(encoder_inputs))
        for layer in self.encoder:
            x = layer(x)

        batch_indices = th.arange(x.size(0), device=history.device)
        return x[batch_indices, last_position]


# ---------------------------------------------------------------------------
# Decoder (DeepDecoderDDPM)
# ---------------------------------------------------------------------------

class DeepDecoderDDPM(nn.Module):
    def __init__(self, hidden_dim, num_layers=4, dropout_prob=0.5,
                 dist_dropout=0.8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dist_dropout = dist_dropout
        self.time_emb = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.item_linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        if num_layers == 0:  # 3d -> d
            self.direct_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        else: # 3d -> 2d -> [res_blocks] -> ds
            self.input_proj = nn.Linear(hidden_dim * 3, hidden_dim * 2)
            self.res_blocks = nn.ModuleList()
            for _ in range(num_layers):
                self.res_blocks.append(nn.Sequential(
                    nn.LayerNorm(hidden_dim * 2),
                    nn.Linear(hidden_dim * 2, hidden_dim * 4), nn.GELU(),
                    nn.Dropout(dropout_prob),
                    nn.Linear(hidden_dim * 4, hidden_dim * 2), nn.Dropout(dropout_prob),
                ))
            self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight.data, a=0.01, nonlinearity="leaky_relu")
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def dist_projection(self, noise, all_item_emb):
        """Project N-dim distribution to D-dim embedding: E[emb | x_t] = softmax(topk_sample(x_t)) @ E.
        """
        all_item_emb_processed = self.item_linear(all_item_emb)
        
        if self.dist_dropout > 0.0 and self.training:
            B, N = noise.shape
            K = max(1, int(N * (1.0 - self.dist_dropout)))
            # Score-based sampling: sample K items proportional to x_t distribution
            sample = -th.log(-th.log(th.rand(B, N, device=noise.device) + 1e-20) + 1e-20)
            topk_idx = (noise + sample).topk(K, dim=-1).indices  # [B, K]
            mask = th.ones(B, N, device=noise.device, dtype=th.bool)
            mask.scatter_(1, topk_idx, False)
            noise_masked = noise.masked_fill(mask, float('-inf'))  # [B, N]
            score = F.softmax(noise_masked, dim=-1)  # [B, N], masked items get 0
            weighted_v = th.matmul(score, all_item_emb_processed)  # [B, D]
        else:
            score = F.softmax(noise, dim=-1)  # [B, N]
            weighted_v = th.matmul(score, all_item_emb_processed)
        weighted_v = self.dropout(weighted_v)
        return weighted_v

    def _mlp_forward(self, inp):
        """Shared MLP: input -> hidden -> output embedding."""
        if self.num_layers == 0:
            hidden = self.direct_proj(inp)
        else:
            hidden = self.input_proj(inp)
            for block in self.res_blocks:
                hidden = hidden + block(hidden)
            hidden = self.output_proj(hidden)
        return hidden

    def forward(self, noise, timesteps, condition, all_item_emb):
        timestep_emb = self.time_emb(timestep_embedding(timesteps, self.hidden_dim))
        weighted_v = self.dist_projection(noise, all_item_emb)
        inp = th.cat([weighted_v, timestep_emb, condition], dim=-1)
        h = self._mlp_forward(inp)
        return th.matmul(h, all_item_emb.t())


# ---------------------------------------------------------------------------
# Soft target
# ---------------------------------------------------------------------------

def compute_soft_target(target, target_id, all_item_emb, corr_alpha=0.8, corr_tau=0.15):
    if corr_alpha >= 1.0:
        return target
    target_emb = all_item_emb[target_id]
    target_emb_norm = F.normalize(target_emb, dim=-1)
    all_emb_norm = F.normalize(all_item_emb, dim=-1)
    sim = th.matmul(target_emb_norm, all_emb_norm.t())
    soft_sim = F.softmax(sim / corr_tau, dim=-1)
    return corr_alpha * target + (1.0 - corr_alpha) * soft_sim


# ---------------------------------------------------------------------------
# PreDiff main model
# ---------------------------------------------------------------------------

class PreDiff(nn.Module):
    """PreDiff Model."""

    def __init__(self, item_size, hidden_dim, encoder_seq_len,
                 num_encoder_layers=2, num_attention_heads=2, dropout_prob=0.4,
                 diffusion_steps=32, noise_schedule='linear',
                 decoder_layers=4,
                 corr_alpha=0.8, corr_tau=0.15,
                 dist_dropout=0.8):
        super().__init__()
        self.corr_alpha = corr_alpha
        self.corr_tau = corr_tau

        self.encoder = Encoder(hidden_dim, num_encoder_layers, encoder_seq_len,
                               item_size, num_attention_heads, dropout_prob)
        self.decoder = DeepDecoderDDPM(hidden_dim, decoder_layers,
                                       dropout_prob,
                                       dist_dropout=dist_dropout)
        self.diffusion = GaussianDiffusion(betas=get_named_beta_schedule(noise_schedule, diffusion_steps))
        self.schedule_sampler = UniformSampler(self.diffusion)

    def compute_loss(self, batch):
        device = next(self.parameters()).device
        bsz = batch['condition'].shape[0]
        t, weights = self.schedule_sampler.sample(bsz, device)

        history = batch['condition'].to(device)
        condition = self.encoder(history)
        target_id = batch['target_id'].to(device).squeeze()
        all_item_emb = self.encoder.get_all_item_emb()

        encoder_logits = th.matmul(condition, all_item_emb.t())

        # Build dense target on-the-fly from target_id (sparse → dense)
        target = th.zeros(bsz, all_item_emb.shape[0], device=device)
        target.scatter_(1, target_id.unsqueeze(1), 1.0)
        soft_target = compute_soft_target(target, target_id, all_item_emb,
                                          self.corr_alpha, self.corr_tau)

        x_t = self.diffusion.q_sample(soft_target, t, noise=th.randn_like(soft_target))

        decoder_logits = self.decoder(x_t, self.diffusion._scale_timesteps(t), condition, all_item_emb.detach())

        fused_logits = encoder_logits + decoder_logits

        log_pred = F.log_softmax(fused_logits, dim=-1)
        kl_loss = F.kl_div(log_pred, soft_target, reduction='none').sum(dim=-1)

        loss = (kl_loss * weights).mean()

        return {'loss': loss}

    def compute_scores(self, batch):
        device = next(self.parameters()).device
        history = batch['condition'].to(device)
        mask = batch['mask'].to(device) == 0.0
        condition = self.encoder(history)
        all_item_emb = self.encoder.get_all_item_emb()

        sample_shape = mask.shape
        x_noised = th.randn(sample_shape, dtype=th.float32, device=device)

        sample = self.diffusion.p_sample_loop(
            self.decoder, sample_shape, condition, all_item_emb,
            noise=x_noised
        )

        sample[mask] = -th.inf
        return sample