from torch import nn, cat
from .ResBlock import ResBlock
from .MoveEmbedder import MoveEmbedder
from .custom_encodings import ENCODING_CONSTANTS

class ResNet(nn.Module):
    def __init__(self, action_size, num_resBlocks, num_hidden = 256):
        super().__init__()

        self.n_moves = (ENCODING_CONSTANTS.MAX_MOVES_PER_PKM+1) * ENCODING_CONSTANTS.MAX_ACT_PKM_PER_TEAM
        self.field_dim = ENCODING_CONSTANTS.STATE - ENCODING_CONSTANTS.MOVE * self.n_moves

        self.move_embedder = MoveEmbedder()

        self.startBlock = nn.Sequential(
            nn.Linear(self.field_dim + self.n_moves * ENCODING_CONSTANTS.EMBEDDED_MOVE, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        
        self.policyHead = nn.Sequential(
            nn.Linear(num_hidden, num_hidden // 2),
            nn.BatchNorm1d(num_hidden // 2),
            nn.ReLU(),
            nn.Linear(num_hidden // 2, action_size)
        )
        
        self.valueHead = nn.Sequential(
            nn.Linear(num_hidden, num_hidden // 2),
            nn.BatchNorm1d(num_hidden // 2),
            nn.ReLU(),
            nn.Linear(num_hidden // 2, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # embed moves
        batch_size = x.size(0)
        field_state = x[:, :self.field_dim]
        moves_raw = x[:, self.field_dim:]
        moves_raw = moves_raw.reshape(-1, self.n_moves, ENCODING_CONSTANTS.MOVE)
        moves_emb = self.move_embedder(moves_raw)
        moves_emb = moves_emb.reshape(batch_size, -1)

        # concatenated field and embedded move features
        fused = cat([field_state, moves_emb], dim=1)

        fused = self.startBlock(fused)
        for resBlock in self.backBone:
            fused = resBlock(fused)
        policy = self.policyHead(fused)
        value = self.valueHead(fused)
        return policy, value