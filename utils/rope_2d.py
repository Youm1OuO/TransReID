import torch

def generate_2d_rope(h, w, dim, base=10000):
    assert dim % 4 == 0, "head_dim 必须是 4 的倍数"
    d_2 = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, d_2, 2).float() / d_2))
    
    grid_y = torch.arange(h).float()
    grid_x = torch.arange(w).float()
    
    freqs_y = torch.einsum('i,j->ij', grid_y, inv_freq)
    freqs_x = torch.einsum('i,j->ij', grid_x, inv_freq)
    
    freqs_y = freqs_y.unsqueeze(1).expand(-1, w, -1).reshape(h * w, -1)
    freqs_x = freqs_x.unsqueeze(0).expand(h, -1, -1).reshape(h * w, -1)
    
    freqs = torch.cat([freqs_y, freqs_x], dim=-1)
    emb = torch.cat([freqs, freqs], dim=-1) 
    return emb.cos(), emb.sin()

def apply_rotary_pos_emb(x, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0).to(x.device)
    sin = sin.unsqueeze(0).unsqueeze(0).to(x.device)
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    x_rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + x_rot * sin