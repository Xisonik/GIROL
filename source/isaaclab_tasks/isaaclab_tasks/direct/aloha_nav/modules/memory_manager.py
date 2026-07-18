import torch
from typing import Optional

class MemoryManager:
    def __init__(self, num_envs, embedding_size, action_size, device, history_length = 13,
                 dtype=torch.float32):
        self.num_envs = num_envs
        self.embedding_size = embedding_size
        self.action_size = action_size
        self.history_length = history_length
        self.device = device
        self.dtype = dtype

        self.embedding_history = torch.zeros(
            (num_envs, history_length, embedding_size), device=device, dtype=dtype
        )
        self.action_history = torch.zeros(
            (num_envs, history_length, action_size), device=device, dtype=dtype
        )

        # per-env флаг наличия истории
        self.initialized = torch.zeros((num_envs,), device=device, dtype=torch.bool)

        # нулевое действие
        self.zero_action = torch.zeros((action_size,), device=device, dtype=dtype)

    @torch.no_grad()
    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Сбрасывает историю для env_ids (или для всех)."""
        if env_ids is None:
            self.embedding_history.zero_()
            self.action_history.zero_()
            self.initialized.fill_(False)
            return

        env_ids = env_ids.to(self.device, dtype=torch.long)
        self.embedding_history[env_ids] = 0
        self.action_history[env_ids] = 0
        self.initialized[env_ids] = False

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, actions: torch.Tensor):
        """
        Алгоритм:
        1) найти неинициализированные env (initialized=False)
        2) для них: продублировать текущий embedding на всю историю
           и заполнить actions нулями на всю историю
        3) затем сделать общий тензорный push-front для ВСЕХ env:
           history[:, 0] = текущие данные, history сдвигается вправо
        """
        embeddings = embeddings.to(self.device, dtype=self.dtype)
        actions = actions.to(self.device, dtype=self.dtype)

        # 1) найти env без истории
        new_mask = ~self.initialized  # [N]

        # 2) инициализировать только их
        if new_mask.any():
            # embeddings: дублируем по всей истории
            self.embedding_history[new_mask] = embeddings[new_mask].unsqueeze(1).expand(-1, self.history_length, -1)

            # actions: по всей истории нули
            self.action_history[new_mask] = self.zero_action.view(1, 1, -1).expand(
                int(new_mask.sum().item()), self.history_length, -1
            )

            self.initialized[new_mask] = True

        # 3) общий push-front для всех env (одним тензорным обновлением)
        self.embedding_history = torch.roll(self.embedding_history, shifts=1, dims=1)
        self.embedding_history[:, 0] = embeddings

        self.action_history = torch.roll(self.action_history, shifts=1, dims=1)
        self.action_history[:, 0] = actions

    @torch.no_grad()
    def get_observations(self, m: int = 4, k: int = 4, aggregate_actions: str = 'sum') -> torch.Tensor:
        """
        Возвращает [N, m*(E + A)] = concat(emb_0, act_0_to_k, emb_k, act_k_to_2k, ...)
        
        Args:
            m: количество сэмплируемых точек
            k: шаг между точками
            aggregate_actions: способ агрегации действий ('sum', 'mean', 'last')
        """
        # Индексы для эмбеддингов
        emb_indices = torch.arange(0, m * k, k, device=self.device, dtype=torch.long)
        emb_indices = torch.clamp(emb_indices, 0, self.history_length - 1)
        
        # Получаем эмбеддинги [N, m, E]
        sel_emb = self.embedding_history[:, emb_indices]
        
        # Получаем агрегированные действия для каждого окна
        aggregated_actions = []
        
        for i in range(m):
            start_idx = i * k
            end_idx = min((i + 1) * k, self.history_length)
            
            window_actions = self.action_history[:, start_idx:end_idx]  # [N, window_len, A]
            
            if aggregate_actions == 'sum':
                window_actions = window_actions.sum(dim=1)  # [N, A]
            
            aggregated_actions.append(window_actions)
        
        # Стек действий [N, m, A]
        sel_act = torch.stack(aggregated_actions, dim=1)
        
        # Конкатенируем и решейпим
        return torch.cat([sel_emb, sel_act], dim=-1).reshape(self.num_envs, -1)