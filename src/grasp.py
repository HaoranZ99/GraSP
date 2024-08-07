import torch
import torch.nn.functional as F

from layers import AttentionModule, TensorNetworkModule

from torch_geometric.nn import GCNConv, GINConv, ResGatedGraphConv, pool, norm

from torch_geometric.nn.inits import reset



class GraSP(torch.nn.Module):
    """
    GraSP
    """

    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(GraSP, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()
        self.reset_parameters()
        self.scale_init()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metric = args.metric

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.pre = torch.nn.Linear(self.number_labels, self.args.hidden_dim)
        self.use_pe = self.args.use_pe
        if self.use_pe:
            hidden_dim = self.args.hidden_dim * 2
        else:
            hidden_dim = self.args.hidden_dim
        if self.args.gnn_operator == "gcn":
            self.convolution = lambda: GCNConv(hidden_dim, hidden_dim)
        elif self.args.gnn_operator == "rggc":
            self.convolution = lambda: ResGatedGraphConv(hidden_dim, hidden_dim, act=torch.nn.Sigmoid())
        elif self.args.gnn_operator == "gin":
            nn = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )
            self.convolution = lambda: GINConv(nn, train_eps=True)
        else:
            raise NotImplementedError("Unknown GNN-Operator.")


        self.attention = AttentionModule(self.args)

        self.conv_layers = torch.nn.ModuleList()
        for _ in range(self.args.k):
            self.conv_layers.append(self.convolution())

        self.tensor_network = TensorNetworkModule(self.args)
        self.scoring_layer = torch.nn.Sequential(
            torch.nn.Linear(self.args.tensor_neurons, self.args.tensor_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(self.args.tensor_neurons, 1)
        )

        # Positional encoding.
        self.use_pe = self.args.use_pe
        if self.use_pe:
            input_size = self.args.pe_dim
            output_size = self.args.hidden_dim

            hidden_size = input_size * 2
            layers = []

            while hidden_size < output_size:
                layers.append(torch.nn.Linear(input_size, hidden_size))
                layers.append(torch.nn.ReLU())
                input_size = hidden_size
                hidden_size *= 2

            layers.append(torch.nn.Linear(input_size, output_size))
            self.embedding_pos_enc = torch.nn.Sequential(*layers)

        hidden_dim = self.args.hidden_dim * (self.args.k + 1)
        if self.use_pe:
            hidden_dim = hidden_dim * 2
        self.post = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // self.args.reduction),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // self.args.reduction, self.args.hidden_dim)
        )

        self.alpha = torch.nn.Parameter(torch.Tensor(1))


        self.graphnorm = norm.GraphSizeNorm()


    def reset_parameters(self):
        reset(self.pre)
        reset(self.post)
        reset(self.scoring_layer)
        if self.use_pe:
            reset(self.embedding_pos_enc)

    def scale_init(self):
        torch.nn.init.constant_(self.alpha, 0)
        self.delta = torch.nn.Parameter(torch.Tensor(1))
        torch.nn.init.constant_(self.delta, 1)
        if self.args.pool == 'multi':
            dims = self.args.hidden_dim * (self.args.k + 1)
            if self.use_pe:
                dims *= 2
            self.mu = torch.nn.Parameter(torch.Tensor(dims))
            torch.nn.init.constant_(self.mu, 0.5)

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Abstract feature matrix.
        """
        x = features * (1 + self.delta)
        for i in range(self.args.k - 1):
            features_prev = features
            features = self.conv_layers[i](features, edge_index)
            features = F.relu(features)
            features = features + features_prev
            features = F.dropout(features, p=self.args.dropout, training=self.training)
            x = torch.cat((x, features), dim=1)
        features = self.conv_layers[self.args.k - 1](features, edge_index)
        x = torch.cat((x, features), dim=1)  # [1127, 256]
        return x

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """
        edge_index_1 = data["g1"].edge_index.to(self.device)
        edge_index_2 = data["g2"].edge_index.to(self.device)
        features_1 = data["g1"].x.to(self.device)
        features_2 = data["g2"].x.to(self.device)
        batch_1 = (
            data["g1"].batch.to(self.device)
            if hasattr(data["g1"], "batch")
            else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes).to(self.device)
        )
        batch_2 = (
            data["g2"].batch.to(self.device)
            if hasattr(data["g2"], "batch")
            else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes).to(self.device)
        )

        features_1 = self.pre(features_1)
        features_2 = self.pre(features_2)

        features_1 = self.graphnorm(features_1, batch_1)
        features_2 = self.graphnorm(features_2, batch_2)

        if self.use_pe:
            pos_features_1 = self.embedding_pos_enc(data["g1"].pos_enc.to(self.device).float())
            pos_features_2 = self.embedding_pos_enc(data["g2"].pos_enc.to(self.device).float())
            features_1 = torch.cat((features_1, pos_features_1), dim=1)
            features_2 = torch.cat((features_2, pos_features_2), dim=1)

        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        pooled_attention_1 = self.attention(abstract_features_1, batch_1)
        pooled_attention_2 = self.attention(abstract_features_2, batch_2)
        pooled_sum_1 = pool.global_add_pool(abstract_features_1, batch_1)
        pooled_sum_2 = pool.global_add_pool(abstract_features_2, batch_2)
        if self.args.pool == 'multi':
            pooled_features_1 = self.mu * pooled_attention_1 + (1 - self.mu) * pooled_sum_1  # 256 * 576
            pooled_features_2 = self.mu * pooled_attention_2 + (1 - self.mu) * pooled_sum_2
        elif self.args.pool == 'att':
            pooled_features_1 = pooled_attention_1
            pooled_features_2 = pooled_attention_2
        elif self.args.pool == 'sum':
            pooled_features_1 = pooled_sum_1
            pooled_features_2 = pooled_sum_2
        else:
            raise NotImplementedError("Unknown pooling method.")

        gx = self.post(pooled_features_1)
        hx = self.post(pooled_features_2)

        scores = self.tensor_network(gx, hx)

        score = self.scoring_layer(scores).view(-1)

        l2_score = torch.norm(gx - hx, dim=-1)

        if self.args.sim_dist == 'sim':
            score = torch.sigmoid(score)
            l2_score = torch.sigmoid(-l2_score).view(-1)

        if self.args.sim_dist == 'sim':
            score = self.alpha * score + (1 - self.alpha) * l2_score
        else:
            if self.metric == 'ged':
                score = -self.alpha * score + (1 - self.alpha) * l2_score
            else:
                score = self.alpha * score - (1 - self.alpha) * l2_score
        return score