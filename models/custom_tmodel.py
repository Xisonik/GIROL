import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, mask=None):
        # x: (batch, seq_len, hidden_size)
        batch_size, seq_len, _ = x.shape
        
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention, V)
        out = out.contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return self.out_linear(out)

class TransformerActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, seq_len=10, **kwargs):
        super().__init__(observation_space, action_space, device)
        GaussianMixin.__init__(self, **kwargs)
        
        self.img_dim = observation_space["img"].shape[0]
        self.num_observations = self.img_dim
        self.seq_len = seq_len
        
        # Позиционное кодирование
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_len, 512))
        
        self.encoder = nn.Sequential(
            nn.Linear(self.num_observations, 512),
            nn.LayerNorm(512),
            nn.ELU()
        )
        
        self.attention = MultiHeadAttention(512, 8)
        self.layer_norm = nn.LayerNorm(512)
        
        self.decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, self.num_actions),
            nn.Tanh()
        )
        
        # Буфер для хранения истории
        self.observation_history = None
        
    def reset_history(self, batch_size=1):
        self.observation_history = torch.zeros(batch_size, self.seq_len, self.num_observations).to(self.device)
    
    def update_history(self, observation):
        # Сдвигаем историю и добавляем новое наблюдение
        if self.observation_history is None:
            self.reset_history(observation.size(0))
        
        self.observation_history = torch.cat([
            self.observation_history[:, 1:], 
            observation.unsqueeze(1)
        ], dim=1)
    
    def compute(self, inputs, role):
        inputs_unflatten = unflatten_tensorized_space(self.observation_space, inputs["states"])
        current_obs = inputs_unflatten["img"]
        
        # Обновляем историю
        self.update_history(current_obs)
        
        # Кодируем всю историю
        encoded_history = self.encoder(self.observation_history) + self.pos_encoding
        
        # Применяем self-attention
        attended = self.attention(encoded_history)
        attended = self.layer_norm(attended + encoded_history)  # residual connection
        
        # Берем только последний вектор (текущий временной шаг)
        context = attended[:, -1, :]
        
        actions = self.decoder(context)
        return actions, self.log_std_parameter, {}