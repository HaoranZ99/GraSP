import math
import copy
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import CyclicLR  # Clr.
from tqdm import tqdm, trange
from scipy.stats import spearmanr, kendalltau

from utils import (
    calculate_ranking_correlation,
    calculate_prec_at_k,
    calculate_prec_at_k_dist,
    normalize_sim_score_batch,
    normalize_mcs_batch,
)
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj, degree
from torch_geometric.transforms import OneHotDegree

from mydataset import MyGEDDataset
from grasp import GraSP

class Trainer(object):
    """
    GraSP model trainer.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.best_state_dict = None  # Store the best model.
        self.process_dataset()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_model()
        self.max_lr = args.learning_rate  # clr.
        self.min_lr = self.max_lr / 10
        self.step_size = 100
        self.metric = args.metric
        if self.metric != "ged" and self.metric != 'mcs':
            raise NotImplementedError("Unknown metric.")

    def setup_model(self):
        """
        Creating a model.
        """
        self.model = GraSP(self.args, self.number_of_labels)
        self.model = self.model.to(self.device)

    def save(self):
        """
        Saving model.
        """
        torch.save(self.model.state_dict(), self.args.save)
        print(f"Model is saved under {self.args.save}.")

    def load(self):
        """
        Loading model.
        """
        self.model.load_state_dict(torch.load(self.args.load, map_location=self.device))
        print(f"Model is loaded from {self.args.load}.")

    def process_dataset(self):
        """
        Downloading and processing dataset.
        """
        print("\nPreparing dataset.\n")

        self.trainval_graphs = MyGEDDataset(self.args, train=True)
        self.testing_graphs = MyGEDDataset(self.args, train=False)

        if self.args.metric == "ged":
            self.ged_matrix = self.trainval_graphs.ged
            self.nged_matrix = self.trainval_graphs.norm_ged
            self.real_data_size = self.ged_matrix.size(0)
        else:
            self.mcs_matrix = self.trainval_graphs.mcs
            self.nmcs_matrix = self.trainval_graphs.norm_mcs
            self.real_data_size = self.mcs_matrix.size(0)

        train_size = len(self.trainval_graphs) - len(self.testing_graphs)
        self.training_graphs = self.trainval_graphs[0:train_size]
        self.validation_graphs = self.trainval_graphs[train_size:]

        if self.trainval_graphs[0].x is None:
            max_degree = 0
            for g in (
                    self.training_graphs
                    + self.validation_graphs
                    + self.testing_graphs
            ):
                if g.edge_index.size(1) > 0:
                    max_degree = max(
                        max_degree, int(degree(g.edge_index[0]).max().item())
                    )
            one_hot_degree = OneHotDegree(max_degree, cat=False)
            self.training_graphs.transform = one_hot_degree
            self.trainval_graphs.transform = one_hot_degree
            self.validation_graphs.transform = one_hot_degree
            self.testing_graphs.transform = one_hot_degree

        self.number_of_labels = self.trainval_graphs.num_features

    def create_batches(self):
        """
        Creating batches from the training graph list.
        :return batches: Zipped loaders as list.
        """

        source_loader = DataLoader(
            self.training_graphs.shuffle(),
            batch_size=self.args.batch_size,
        )
        target_loader = DataLoader(
            self.training_graphs.shuffle(),
            batch_size=self.args.batch_size,
        )

        return list(zip(source_loader, target_loader))

    def transform(self, data):
        """
        Getting ged for graph pair and grouping with data into dictionary.
        :param data: Graph pair.
        :return new_data: Dictionary with data.
        """
        new_data = dict()

        new_data["g1"] = data[0]
        new_data["g2"] = data[1]

        if self.args.metric == "ged":
            ged = self.ged_matrix[
                data[0]["i"].reshape(-1).tolist(), data[1]["i"].reshape(-1).tolist()
            ].tolist()
            new_data["target"] = (torch.tensor(ged).float())

            nged = self.nged_matrix[
                data[0]["i"].reshape(-1).tolist(), data[1]["i"].reshape(-1).tolist()
            ].tolist()
            new_data["sim_target"] = (
                torch.from_numpy(np.exp([(-el) for el in nged])).view(-1).float()
            )
        else:
            mcs = self.mcs_matrix[
                data[0]["i"].reshape(-1).tolist(), data[1]["i"].reshape(-1).tolist()
            ].tolist()
            new_data["target"] = torch.tensor(mcs).float()

            nmcs = self.nmcs_matrix[
                data[0]["i"].reshape(-1).tolist(), data[1]["i"].reshape(-1).tolist()
            ].tolist()
            new_data["sim_target"] = (
                torch.from_numpy(np.array([(el) for el in nmcs])).view(-1).float()
            )

        return new_data

    def process_batch(self, data):
        """
        Forward pass with a data.
        :param data: Data that is essentially pair of batches, for source and target graphs.
        :return loss: Loss on the data.
        """
        self.optimizer.zero_grad()
        data = self.transform(data)
        target = data["target"].to(self.device) if self.args.sim_dist == 'dist' else data["sim_target"].to(self.device)
        prediction = self.model(data)
        loss = F.mse_loss(prediction, target, reduction="sum")
        loss.backward()
        self.optimizer.step()
        self.clr_scheduler.step()  # Update the learning rate.
        return loss.item()

    def fit(self):
        """
        Training a model.
        """
        print("\nModel training.\n")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.min_lr,
            weight_decay=self.args.weight_decay,
        )
        self.clr_scheduler = \
            CyclicLR(self.optimizer, base_lr=self.min_lr, max_lr=self.max_lr,
                     step_size_up=self.step_size * 2, mode='triangular', cycle_momentum=False)  # clr.
        self.model.train()

        best_val_loss = float('inf')

        epochs = trange(self.args.epochs, leave=True, desc="Epoch")
        loss_list = []
        for epoch in epochs:
            batches = self.create_batches()

            main_index = 0
            loss_sum = 0
            for index, batch_pair in tqdm(
                    enumerate(batches), total=len(batches), desc="Batches", leave=False
            ):
                loss_score = self.process_batch(batch_pair)
                main_index = main_index + batch_pair[0].num_graphs
                loss_sum = loss_sum + loss_score
            loss = loss_sum / main_index
            epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
            loss_list.append(loss)

            # Validation.
            if self.args.val_ratio == 0:
                validation_graphs = self.testing_graphs
            else:
                validation_graphs = self.validation_graphs
            if epoch > (1 - self.args.val_percentage) * self.args.epochs and \
                    epoch % self.args.val_every == 0:
                self.model.train(False)
                t = tqdm(
                    total=len(validation_graphs) * len(self.trainval_graphs),
                    position=2,
                    leave=False,
                    desc="Validation",
                )
                scores = torch.empty((len(validation_graphs), len(self.trainval_graphs)))

                for i, g in enumerate(validation_graphs):
                    source_batch = Batch.from_data_list([g] * len(self.trainval_graphs))
                    target_batch = Batch.from_data_list(
                        self.trainval_graphs.shuffle()
                    )
                    data = self.transform((source_batch, target_batch))
                    target = data["target"] if self.args.sim_dist == 'dist' else data["sim_target"]
                    prediction = self.model(data)
                    prediction = prediction.cpu()

                    scores[i] = F.mse_loss(
                        prediction, target, reduction="none"
                    ).detach()
                    t.update(len(self.trainval_graphs))

                t.close()
                loss = scores.mean().item()
                if loss <= best_val_loss:
                    best_val_loss = loss
                    self.best_state_dict = copy.deepcopy(self.model.state_dict())
                self.model.train(True)

    def score(self):
        """
        Scoring.
        """
        print("\n\nModel evaluation.\n")
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            print("Model is loaded from best model.")
        self.model.eval()

        scores = np.empty((len(self.testing_graphs), len(self.trainval_graphs)))
        scores_mae = np.empty((len(self.testing_graphs), len(self.trainval_graphs)))
        ground_truth = np.empty((len(self.testing_graphs), len(self.trainval_graphs)))
        ground_truth_origin = np.empty((len(self.testing_graphs), len(self.trainval_graphs)))
        prediction_mat = np.empty((len(self.testing_graphs), len(self.trainval_graphs)))

        rho_list = []
        tau_list = []
        prec_at_10_list = []
        prec_at_20_list = []

        t = tqdm(total=len(self.testing_graphs) * len(self.trainval_graphs))

        for i, g in enumerate(self.testing_graphs):
            source_batch = Batch.from_data_list([g] * len(self.trainval_graphs))
            target_batch = Batch.from_data_list(self.trainval_graphs)

            data = self.transform((source_batch, target_batch))
            target = data["target"] if self.args.sim_dist == 'dist' else data["sim_target"]
            sim_target = data["sim_target"]
            ground_truth[i] = target
            ground_truth_origin[i] = data["target"]
            prediction = self.model(data)
            prediction = prediction.cpu()
            prediction_mat[i] = prediction.detach().numpy()

            # Loss sim.
            scores[i] = (
                F.mse_loss(prediction, target, reduction="none").detach().numpy()
            )
            scores_mae[i] = (
                torch.nn.L1Loss(reduction="none")(prediction, target).detach().numpy()
            )

            # # Loss sim.
            # if self.args.sim_dist == 'sim':
            #     scores[i] = (
            #         F.mse_loss(prediction, target, reduction="none").detach().numpy()
            #     )
            # else:
            #     if self.args.metric == 'ged':
            #         scores[i] = (
            #             F.mse_loss
            #             (
            #                 normalize_sim_score_batch(g.num_nodes,
            #                                           torch.tensor(
            #                                               [el.num_nodes for el in self.trainval_graphs]).float(),
            #                                           prediction.detach()),
            #                 sim_target, reduction="none"
            #             )
            #         )
            #     else:
            #         scores[i] = (
            #             F.mse_loss
            #             (
            #                 normalize_mcs_batch(g.num_nodes,
            #                                     torch.tensor([el.num_nodes for el in self.trainval_graphs]).float(),
            #                                     prediction.detach()),
            #                 sim_target, reduction="none"
            #             )
            #         )

            if not np.all(ground_truth[i] == ground_truth[i][0]) \
                    and not np.all(prediction_mat[i] == prediction_mat[i][0]):
                rho_list.append(
                    calculate_ranking_correlation(
                        spearmanr, prediction_mat[i], ground_truth[i]
                    )
                )
                tau_list.append(
                    calculate_ranking_correlation(
                        kendalltau, prediction_mat[i], ground_truth[i]
                    )
                )

            if self.args.metric == 'ged' and self.args.sim_dist == 'dist':
                prec_at_10_list.append(
                    calculate_prec_at_k_dist(10, prediction_mat[i], ground_truth[i])
                )
                prec_at_20_list.append(
                    calculate_prec_at_k_dist(20, prediction_mat[i], ground_truth[i])
                )
            else:
                prec_at_10_list.append(
                    calculate_prec_at_k(10, prediction_mat[i], ground_truth[i])
                )
                prec_at_20_list.append(
                    calculate_prec_at_k(20, prediction_mat[i], ground_truth[i])
                )

            t.update(len(self.trainval_graphs))

        self.rho = np.mean(rho_list).item()
        self.tau = np.mean(tau_list).item()
        self.prec_at_10 = np.mean(prec_at_10_list).item()
        self.prec_at_20 = np.mean(prec_at_20_list).item()
        self.model_error = np.mean(scores).item()
        self.model_mae = np.mean(scores_mae).item()
        self.print_evaluation()
        return prediction_mat

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        # print("\nmse(10^-3): " + str(round(self.model_error * 1000, 5)) + ".")
        print("\nrmse: " + str(round(math.sqrt(self.model_error), 5)) + ".")
        print("mae: " + str(round(self.model_mae, 5)) + ".")
        print("Spearman's rho: " + str(round(self.rho, 5)) + ".")
        print("Kendall's tau: " + str(round(self.tau, 5)) + ".")
        print("p@10: " + str(round(self.prec_at_10, 5)) + ".")
        print("p@20: " + str(round(self.prec_at_20, 5)) + ".")