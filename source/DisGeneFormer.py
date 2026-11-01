import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv


class AttentionAggregation(nn.Module):
    def __init__(self, hidden_dim, num_layers=3, num_heads=4):
        super(AttentionAggregation, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)

        # Add a learnable weighted sum to avoid too much dampening from exploding gradients
        self.alpha = nn.Parameter(torch.ones(num_layers) / num_layers)   # shape [L]

    def forward(self, layer_outputs):
        """
        layer_outputs is a list of [N, hidden_dim] from each layer.
        We'll stack them into [num_layers, N, hidden_dim],
        then apply multihead attention across the 'num_layers' dimension.
        """
        # layer_outputs = torch.stack(layer_outputs, dim=0)  # [L, N, D]
        # attn_output, _ = self.attention(layer_outputs, layer_outputs, layer_outputs)

        # # Use learnable weighted sum instead of mean to avoid too much dampening
        # L = layer_outputs.size(0)  # number of layers
        # self.alpha = nn.Parameter(torch.ones(L)/L)
        # attn_output = (self.alpha.softmax(0).view(L,1,1) * attn_output).sum(dim=0)

        # # Checks for embeddings and attention output
        # # Return average over the layer dimension => [N, D]

        layer_outputs = torch.stack(layer_outputs, dim=0)    # [L, N, D]
        attn_out, _ = self.attention(layer_outputs,
                                     layer_outputs,
                                     layer_outputs)          # same shape

        # weighted sum over the layer dimension
        w = self.alpha.softmax(0).view(-1, 1, 1).to(attn_out.device)     # [L,1,1]
        out = (w * attn_out).sum(dim=0)          # [N, D]

        return out


class GlobalGraphTransformer(nn.Module):
    def __init__(self, hidden_dim, n_heads=4, num_layers=2, dropout=0.1):
        super(GlobalGraphTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, node_embs):
        # node_embs: [B, seq_len, hidden_dim], often B=1
        out = self.transformer_encoder(node_embs)
        return out


class DisGeneFormer(nn.Module):
    def __init__(
            self,
            gene_feature_dim,
            disease_feature_dim,
            fc_hidden_dim=2048,
            gene_net_hidden_dim=512,
            disease_net_hidden_dim=512,
            mode='DGP',
            ablate=None,
            task='classification',
            transformer_num_heads=4,
            transformer_num_layers=2,
            transformer_dropout=0.1
    ):
        super(DisGeneFormer, self).__init__()
        self.mode = mode
        self.task = task

        # Ablation switches (config: model.ablate.<name>: true). Absent/False = keep.
        self.ablate = dict(ablate or {})
        self.use_global_transformer = not self.ablate.get('global_transformer', False)

        # Regression predicts one continuous score; classification predicts 2 logits.
        self.out_dim = 1 if task == 'regression' else 2

        # GNN layers for genes
        self.gene_conv_0 = GATConv(gene_feature_dim, gene_net_hidden_dim, heads=4)
        self.gene_conv_1 = GATConv(gene_net_hidden_dim * 4, gene_net_hidden_dim, heads=4)
        self.gene_conv_2 = GATConv(gene_net_hidden_dim * 4, gene_net_hidden_dim, heads=1, dropout=0.2)

        self.gene_proj_0 = Linear(gene_net_hidden_dim * 4, gene_net_hidden_dim)
        self.gene_proj_1 = Linear(gene_net_hidden_dim * 4, gene_net_hidden_dim)

        self.bn_gene_0 = nn.BatchNorm1d(gene_net_hidden_dim * 4)
        self.bn_gene_1 = nn.BatchNorm1d(gene_net_hidden_dim * 4)
        self.bn_gene_2 = nn.BatchNorm1d(gene_net_hidden_dim)

        self.gene_attention_agg = AttentionAggregation(gene_net_hidden_dim, num_heads=4)

        # GNN layers for diseases
        self.disease_conv_0 = GATConv(disease_feature_dim, disease_net_hidden_dim, heads=4)
        self.disease_conv_1 = GATConv(disease_net_hidden_dim * 4, disease_net_hidden_dim, heads=4)
        self.disease_conv_2 = GATConv(disease_net_hidden_dim * 4, disease_net_hidden_dim, heads=1)

        self.disease_proj_0 = Linear(disease_net_hidden_dim * 4, disease_net_hidden_dim)
        self.disease_proj_1 = Linear(disease_net_hidden_dim * 4, disease_net_hidden_dim)

        self.bn_disease_0 = nn.BatchNorm1d(disease_net_hidden_dim * 4)
        self.bn_disease_1 = nn.BatchNorm1d(disease_net_hidden_dim * 4)
        self.bn_disease_2 = nn.BatchNorm1d(disease_net_hidden_dim)

        self.disease_attention_agg = AttentionAggregation(disease_net_hidden_dim, num_heads=4)

        # Global transformer aggregator over all nodes
        assert gene_net_hidden_dim == disease_net_hidden_dim, \
            "For a single transformer aggregator, gene_net_hidden_dim must match disease_net_hidden_dim."

        self.hidden_dim = gene_net_hidden_dim  # == disease_net_hidden_dim
        # Built even when ablated so checkpoints stay compatible; just not applied.
        self.global_transformer = GlobalGraphTransformer(
            hidden_dim=self.hidden_dim,
            n_heads=transformer_num_heads,
            num_layers=transformer_num_layers,
            dropout=transformer_dropout
        )

        # Final head: 2 logits for classification, 1 score for regression.
        self.final_fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, fc_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc_hidden_dim // 2, self.out_dim)
        )

        # If there is a gene classification mode separately:
        fc_gene_classification_hidden_dim = fc_hidden_dim
        self.lin_gc1 = Linear(self.hidden_dim, fc_gene_classification_hidden_dim)
        self.lin_gc2 = Linear(fc_gene_classification_hidden_dim, fc_gene_classification_hidden_dim // 2)
        self.lin_gc3 = Linear(fc_gene_classification_hidden_dim // 2, fc_gene_classification_hidden_dim // 4)
        self.lin_gc4 = Linear(fc_gene_classification_hidden_dim // 4, 2)

    @torch.no_grad()
    def encode_nodes(self, gene_net_data, disease_net_data):
        """
        Run the full backbone once (GNN → Transformer) and return the final
        embedding of **every** gene node and every disease node.

        Returns
        -------
        gene_emb  : torch.Tensor  [N_gene,    hidden_dim]
        dis_emb   : torch.Tensor  [N_disease, hidden_dim]
        """
        self.eval()

        # Gene path
        gx0 = self.gene_conv_0(gene_net_data.x, gene_net_data.edge_index)
        gx0 = self.bn_gene_0(F.leaky_relu(gx0))
        gx0p = self.gene_proj_0(gx0)

        gx1 = self.gene_conv_1(gx0,          gene_net_data.edge_index)
        gx1 = self.bn_gene_1(F.leaky_relu(gx1))
        gx1p = self.gene_proj_1(gx1)

        gx2 = self.gene_conv_2(gx1,          gene_net_data.edge_index)
        gx2 = self.bn_gene_2(F.leaky_relu(gx2))

        g_out = self.gene_attention_agg([gx0p, gx1p, gx2])      # [N_g, D]

        # Disease path
        dx0 = self.disease_conv_0(disease_net_data.x, disease_net_data.edge_index)
        dx0 = self.bn_disease_0(F.leaky_relu(dx0))
        dx0p = self.disease_proj_0(dx0)

        dx1 = self.disease_conv_1(dx0,          disease_net_data.edge_index)
        dx1 = self.bn_disease_1(F.leaky_relu(dx1))
        dx1p = self.disease_proj_1(dx1)

        dx2 = self.disease_conv_2(dx1,          disease_net_data.edge_index)
        dx2 = self.bn_disease_2(F.leaky_relu(dx2))

        d_out = self.disease_attention_agg([dx0p, dx1p, dx2])   # [N_d, D]

        # Global transformer across gene and disease graphs (skipped if ablated)
        num_g = g_out.size(0)
        combined = torch.cat([g_out, d_out], dim=0).unsqueeze(0)       # [1, N_g+N_d, D]
        if self.use_global_transformer:
            combined = self.global_transformer(combined)              # same shape
        combined = combined.squeeze(0)                                 # [N_g+N_d, D]

        gene_emb     = combined[:num_g]          # [N_g, D]
        disease_emb  = combined[num_g:]          # [N_d, D]
        return gene_emb, disease_emb


    def forward(self, gene_net_data, disease_net_data, batch_idx):
        """
        gene_net_data, disease_net_data: each has (x, edge_index).
        batch_idx: [batch_size, 2], each row = (gene_idx, disease_idx).

        For a global aggregator, we have to be careful about indexing:
          - gene_x_out has shape [num_gene_nodes, hidden_dim]
          - disease_x_out has shape [num_disease_nodes, hidden_dim]
        We'll create a single [num_gene_nodes + num_disease_nodes, hidden_dim] to feed the transformer.
        Then we have to pick out the gene or disease node for each sample from the correct portion.
        """
        # GNN for Genes
        gene_x, gene_edge_index = gene_net_data.x, gene_net_data.edge_index
        gx0 = self.gene_conv_0(gene_x, gene_edge_index)
        gx0 = self.bn_gene_0(F.leaky_relu(gx0))
        gx0_proj = self.gene_proj_0(gx0)

        gx1 = self.gene_conv_1(gx0, gene_edge_index)
        gx1 = self.bn_gene_1(F.leaky_relu(gx1))
        gx1_proj = self.gene_proj_1(gx1)

        gx2 = self.gene_conv_2(gx1, gene_edge_index)
        gx2 = self.bn_gene_2(F.leaky_relu(gx2))

        # Aggregation across layers
        gene_layer_outputs = [gx0_proj, gx1_proj, gx2]
        gene_x_out = self.gene_attention_agg(gene_layer_outputs)  # shape [num_gene_nodes, hidden_dim]

        # GNN for Diseases
        disease_x, disease_edge_index = disease_net_data.x, disease_net_data.edge_index
        dx0 = self.disease_conv_0(disease_x, disease_edge_index)
        dx0 = self.bn_disease_0(F.leaky_relu(dx0))
        dx0_proj = self.disease_proj_0(dx0)

        dx1 = self.disease_conv_1(dx0, disease_edge_index)
        dx1 = self.bn_disease_1(F.leaky_relu(dx1))
        dx1_proj = self.disease_proj_1(dx1)

        dx2 = self.disease_conv_2(dx1, disease_edge_index)
        dx2 = self.bn_disease_2(F.leaky_relu(dx2))

        # Aggregation across layers
        disease_layer_outputs = [dx0_proj, dx1_proj, dx2]
        disease_x_out = self.disease_attention_agg(disease_layer_outputs)  # shape [num_disease_nodes, hidden_dim]

        # Global Transformer 
        num_gene_nodes = gene_x_out.size(0)
        num_disease_nodes = disease_x_out.size(0)

        combined_emb = torch.cat([gene_x_out, disease_x_out], dim=0)  # [N_g + N_d, hidden_dim]
        combined_emb = combined_emb.unsqueeze(0)                     # => [1, (N_g + N_d), hidden_dim]

        # Apply the global transformer unless it is ablated (then pass through).
        if self.use_global_transformer:
            combined_emb = self.global_transformer(combined_emb)      # => [1, (N_g + N_d), hidden_dim]
        combined_emb_trans = combined_emb.squeeze(0)                  # => [N_g + N_d, hidden_dim]

        updated_gene_emb = combined_emb_trans[:num_gene_nodes, :]     # [N_g, hidden_dim]
        updated_disease_emb = combined_emb_trans[num_gene_nodes:, :]  # [N_d, hidden_dim]

        # Gather the relevant node embeddings for the (gene_idx, disease_idx) pairs
        x_gene = updated_gene_emb[batch_idx[:, 0]]
        x_disease = updated_disease_emb[batch_idx[:, 1]]

        if self.mode == 'DGP':
            pair_emb = torch.cat([x_gene, x_disease], dim=-1)  # [B, 2*hidden_dim]
            logits = self.final_fc(pair_emb)                   # => [B, 2]
            return logits

        else:
            # Gene classification mode
            x = F.dropout(x_gene, p=0.5, training=self.training)
            x = F.leaky_relu(self.lin_gc1(x))
            x = F.dropout(x, p=0.4, training=self.training)
            x = F.leaky_relu(self.lin_gc2(x))
            x = F.dropout(x, p=0.2, training=self.training)
            x = F.leaky_relu(self.lin_gc3(x))
            x = self.lin_gc4(x)
            return x
