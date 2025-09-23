import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, global_mean_pool

class ASGRA(nn.Module):
    '''
    GATv2 for graph-level indoor Scene Classification
    '''
    def __init__(
        self,
        num_tokens: int = 151,
        num_relations: int = 51,
        num_classes: int = 8,
        emb_dim: int = 256,
        bbox_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        assert hidden_dim % heads == 0

        self.tok_emb = nn.Embedding(num_tokens, emb_dim)
        self.bbox_proj = nn.Linear(4, bbox_dim)
        in_dim = emb_dim + bbox_dim

        self.rel_emb = nn.Embedding(num_relations, hidden_dim // heads)

        GATv2Layer = GATv2Conv(
            in_dim,
            hidden_dim // heads,
            heads=heads,
            edge_dim=hidden_dim // heads,
            dropout=dropout,
        )
        
        self.convs = nn.ModuleList()

        # first layer: 288 → hidden_dim
        self.convs.append(
            GATv2Conv(
                in_dim,
                hidden_dim // heads,
                heads=heads,
                edge_dim=hidden_dim // heads,
                dropout=dropout,
            )
        )

        # remaining layers: hidden_dim → hidden_dim
        for _ in range(1, num_layers):
            self.convs.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    edge_dim=hidden_dim // heads,
                    dropout=dropout,
                )
            )
    

        assert self.convs[0].in_channels == in_dim, \
       f"GAT expects {self.convs[0].in_channels} but node dim is {in_dim}"

        self.dropout = nn.Dropout(dropout)

        # Graph-level readout + classifier
        self.pool = global_mean_pool
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        """
        data.x          – [N,5] (col0=int token-id, col1:4=float bbox)
        data.edge_index – [2,E]
        data.edge_attr  – [E]   (int relation-id)
        data.batch      – batch vector (provided by PyG’s DataLoader)
        """
        tok_id = data.x[:, 0].long()
        bbox   = data.x[:, 1:].float()          
        if bbox.size(1) > 4:                   
            bbox = bbox[:, :4]
        elif bbox.size(1) < 4:                  
            pad  = bbox.new_zeros(bbox.size(0), 4 - bbox.size(1))
            bbox = torch.cat([bbox, pad], dim=1)

        assert bbox.size(1) == 4, "bbox feature must be 4-D after sanitising" 

        x = torch.cat(
            [self.tok_emb(tok_id), self.bbox_proj(bbox)], dim=-1     # [N, in_dim]
        )

        assert x.size(-1) == self.convs[0].in_channels, \
            f"feature dim {x.size(-1)} ≠ GAT expects {self.convs[0].in_channels}"

        edge_attr = self.rel_emb(data.edge_attr)                     # [E, edge_dim]

        for conv in self.convs:
            x = conv(x, data.edge_index, edge_attr).relu()
            x = self.dropout(x)
        
        g = self.pool(x, data.batch)
        return self.mlp(g)
    