class TwoTowerProjectionHead(nn.Module):
    """
    Two-Tower Projection Head with multi-tower customer fusion and bounded temperature.
    Supports single or multiple customer towers (e.g. Long-term + Mid-term).
    """
    def __init__(
        self,
        gemma_dim: int = 768,
        projection_dim: int = 128,
        dropout: float = 0.1,
        num_customer_towers: int = 1,
    ):
        super().__init__()
        self.num_customer_towers = num_customer_towers

        # Independent customer projection towers
        self.user_towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(gemma_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, projection_dim),
            )
            for _ in range(num_customer_towers)
        ])

        # Gated fusion when combining multiple customer towers
        if num_customer_towers > 1:
            self.fusion_gate = nn.Sequential(
                nn.Linear(projection_dim * num_customer_towers, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_customer_towers),
            )

        # Product Projection Tower
        self.item_fc = nn.Sequential(
            nn.Linear(gemma_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, projection_dim),
        )

        # Learnable log-temperature parameter initialized to ln(0.07) ~ -2.659
        self.log_temperature = nn.Parameter(torch.tensor(-2.65926))

    def get_temperature(self) -> torch.Tensor:
        return self.log_temperature.clamp(min=-4.6, max=-0.7).exp()

    def project_user(self, user_vectors: torch.Tensor) -> torch.Tensor:
        """Projects single or stacked customer vectors into a normalized projection_dim vector."""
        user_vectors = user_vectors.float()
        if self.num_customer_towers == 1:
            if user_vectors.ndim == 3:
                user_vectors = user_vectors[:, 0, :]
            return F.normalize(self.user_towers[0](user_vectors), p=2, dim=1)

        present = (user_vectors.abs().sum(dim=-1) > 0)
        present[:, 0] = True  # Long-term is always primary

        tower_embs = [F.normalize(self.user_towers[t](user_vectors[:, t, :]), p=2, dim=1) for t in range(self.num_customer_towers)]
        concat_emb = torch.cat(tower_embs, dim=1)
        gate_logits = self.fusion_gate(concat_emb)
        gate_logits = gate_logits.masked_fill(~present, -1e4)
        weights = F.softmax(gate_logits, dim=1)

        stacked = torch.stack(tower_embs, dim=1)
        weighted_sum = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return F.normalize(weighted_sum, p=2, dim=1)

    def project_item(self, item_vectors: torch.Tensor) -> torch.Tensor:
        """Projects product vectors into a normalized 128-dim vector."""
        return F.normalize(self.item_fc(item_vectors.float()), p=2, dim=1)

    def forward(
        self,
        user_vectors: torch.Tensor,
        item_vectors: torch.Tensor,
        mode: str = "matrix",
    ) -> Any:
        logq = None
        if item_vectors.shape[-1] == self.item_fc[0].in_features + 1:
            logq = item_vectors[:, -1]
            item_vectors = item_vectors[:, :-1]

        u_emb = self.project_user(user_vectors)
        i_emb = self.project_item(item_vectors)
        temp = self.get_temperature()

        if mode == "matrix":
            logits = torch.matmul(u_emb, i_emb.T) / temp
            return logits, logq
        elif mode == "pairwise":
            return torch.sum(u_emb * i_emb, dim=1) / temp
