from utils import *
from graphormer.layers import *
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(
            self,
            hops,
            output_dim,
            input_dim,
            num_drug,
            num_meta,
            graphformer_layers,
            num_heads,
            hidden_dim,
            ffn_dim,
            dropout_rate,
            attention_dropout_rate,
    ):
        super().__init__()
        self.seq_len = hops + 1
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.graphformer_layers = graphformer_layers
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.num_drug = num_drug
        self.num_meta = num_meta

        self.att_embeddings_nope = nn.Linear(self.input_dim, self.hidden_dim)
        self.att_embeddings_nope_1 = nn.Linear(64, self.hidden_dim)
        encoders = [
            EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.attention_dropout_rate, self.num_heads)
            for _ in range(self.graphformer_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, int(self.hidden_dim / 2))
        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)
        self.Linear1 = nn.Linear(int(self.hidden_dim / 2), self.output_dim)
        self.scaling = nn.Parameter(torch.ones(1) * 0.5)
        self.semantic_attention = Attention(self.hidden_dim, self.dropout_rate)
        self.mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 3)
        )
        self.apply(lambda module: init_params(module, n_layers=self.graphformer_layers))

    def MetaPath_Encoding(self, processed_features):
        if processed_features.shape[2] == 80:
            tensor = self.att_embeddings_nope(processed_features)
        else:
            tensor = self.att_embeddings_nope_1(processed_features)
        # transformer encoder
        for enc_layer in self.layers:
            tensor = enc_layer(tensor)
        output = self.final_ln(tensor)
        target = output[:, 0, :].unsqueeze(1).repeat(1, self.seq_len - 1, 1)
        split_tensor = torch.split(output, [1, self.seq_len - 1], dim=1)
        node_tensor = split_tensor[0]
        neighbor_tensor = split_tensor[1]
        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))
        layer_atten = F.softmax(layer_atten, dim=1)
        neighbor_tensor = neighbor_tensor * layer_atten
        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)
        x_former = (node_tensor + neighbor_tensor).squeeze()
        return x_former

    def forward(self, Dr_Down_M_adj, Dr_Up_M_adj, Dr_Di_hops, Dr_G_hops, M_Di_hops, M_G_hops, train_edge_index, val_edge_index, hops):

        Dr_Di = self.MetaPath_Encoding(Dr_Di_hops)
        Dr_G = self.MetaPath_Encoding(Dr_G_hops)
        M_Di = self.MetaPath_Encoding(M_Di_hops)
        M_G = self.MetaPath_Encoding(M_G_hops)

        Drug_semantic_embeddings = torch.cat((Dr_Di[0:self.num_drug, :], Dr_G[0:self.num_drug, :]), dim=1)
        Meta_semantic_embeddings = torch.cat((M_Di[0:self.num_meta, :], M_G[0:self.num_meta, :]), dim=1)

        Drug_Meta_1 = torch.cat((Drug_semantic_embeddings, Meta_semantic_embeddings), dim=0)

        Dr_Down_M_input = Drug_Meta_1
        Dr_Up_M_input = Drug_Meta_1

        Dr_Down_M_hops_1 = re_features(Dr_Down_M_adj, Dr_Down_M_input, hops).to(device)
        Dr_Up_M_hops_1 = re_features(Dr_Up_M_adj, Dr_Up_M_input, hops).to(device)

        Dr_Up_M = self.MetaPath_Encoding(Dr_Down_M_hops_1)
        Dr_Down_M = self.MetaPath_Encoding(Dr_Up_M_hops_1)

        Drug_embeddings = torch.stack((Dr_Up_M[0:self.num_drug, :], Dr_Down_M[0:self.num_drug, :]), dim=0)
        Drug_weighted_embeddings = self.semantic_attention(Drug_embeddings)

        Meta_embeddings = torch.stack((Dr_Up_M[self.num_drug:, :], Dr_Down_M[self.num_drug:, :]), dim=0)
        Meta_weighted_embeddings = self.semantic_attention(Meta_embeddings)

        Drug_embeddings = Drug_weighted_embeddings
        Meta_embeddings = Meta_weighted_embeddings

        train_val_edge_index = torch.cat([train_edge_index, val_edge_index], 1)

        Drug_Meta = []
        for i in range(train_val_edge_index.size(1)):
            first_element = train_val_edge_index[0, i]
            second_element = train_val_edge_index[1, i]
            combined_row = torch.cat((Drug_embeddings[first_element, :], Meta_embeddings[second_element, :]), dim=0)
            Drug_Meta.append(combined_row)
        Drug_Meta_features = torch.stack(Drug_Meta)
        embedings = self.mlp(Drug_Meta_features)
        return embedings





