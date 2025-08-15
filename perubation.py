import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import os
import traceback
import random
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import time

# Paths to data files
ppi_path = "/mnt/c/Desktop/PPI/Biogrid.txt"
drug_path = "/mnt/c/Desktop/PPI/ChG-Miner_miner-chem-gene.tsv"
mutation_path = "/mnt/c/Desktop/PPI/S4191.txt"
# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def extract_common_subgraph(edge_index, protein_to_idx, drug_targets, mutated_proteins, node_features, perturbation_scores, pagerank_scores):
    common_proteins = drug_targets.intersection(mutated_proteins)
    common_indices = [protein_to_idx[p] for p in common_proteins if p in protein_to_idx]
    common_set = set(common_indices)

    mask = [(i, j) for i, j in zip(edge_index[0], edge_index[1]) if i.item() in common_set and j.item() in common_set]
    if not mask:
        raise ValueError("No common edges found between mutated and drug target proteins.")

    sub_edge_index = torch.tensor(mask, dtype=torch.long).t().contiguous()

    # Reindex
    old_to_new = {old: new for new, old in enumerate(sorted(common_set))}
    remapped_edges = [[old_to_new[i.item()], old_to_new[j.item()]] for i, j in sub_edge_index.t()]
    remapped_edge_index = torch.tensor(remapped_edges, dtype=torch.long).t().contiguous()

    node_mask = torch.tensor([i in common_set for i in range(len(protein_to_idx))])
    sub_features = node_features[node_mask]
    sub_pert_scores = perturbation_scores[node_mask]
    sub_pagerank = pagerank_scores[node_mask]

    return remapped_edge_index, sub_features, sub_pert_scores, sub_pagerank, old_to_new


def calculate_pagerank(edge_index, num_nodes, alpha=0.85, max_iterations=100, tol=1e-6):
    print("Calculating PageRank scores...")
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edge_list = edge_index.t().numpy()
    for src, dst in edge_list:
        G.add_edge(int(src), int(dst))
    pagerank_dict = nx.pagerank(G, alpha=alpha, max_iter=max_iterations, tol=tol)
    pagerank_scores = torch.zeros(num_nodes, dtype=torch.float)
    for node, score in pagerank_dict.items():
        pagerank_scores[node] = score
    if pagerank_scores.max() > 0:
        pagerank_scores = pagerank_scores / pagerank_scores.max()
    print(f"PageRank calculation complete. Min: {pagerank_scores.min():.6f}, Max: {pagerank_scores.max():.6f}")
    return pagerank_scores


def load_real_ppi_data(ppi_path, drug_path, mutation_path):
    """Load real PPI data and generate perturbation scores based on drug targets and mutations"""
    print("🔍 Loading real datasets...")

    print("Loading BioGRID interactions...")
    ppi_df = pd.read_csv(ppi_path, sep='\t', comment='#', header=None, low_memory=False)
    col_A, col_B = 0, 1
    proteins_A = ppi_df[col_A].astype(str).str.split(":", expand=True)[1]
    proteins_B = ppi_df[col_B].astype(str).str.split(":", expand=True)[1]
    all_proteins = pd.concat([proteins_A, proteins_B]).unique()

    protein_to_idx = {protein: idx for idx, protein in enumerate(sorted(all_proteins))}
    idx_to_protein = {v: k for k, v in protein_to_idx.items()}
    num_nodes = len(protein_to_idx)
    print(f"Number of unique proteins: {num_nodes}")

    edges = []
    for a, b in zip(proteins_A, proteins_B):
        if a in protein_to_idx and b in protein_to_idx:
            edges.append([protein_to_idx[a], protein_to_idx[b]])
            edges.append([protein_to_idx[b], protein_to_idx[a]])
    edge_index = torch.tensor(edges, dtype=torch.long).t()

    print("Generating node features...")
    one_hot_features = torch.eye(num_nodes)
    pagerank_scores = calculate_pagerank(edge_index, num_nodes)
    node_features = torch.cat([one_hot_features, pagerank_scores.unsqueeze(1)], dim=1)

    print("Loading drug targets...")
    drug_df = pd.read_csv(drug_path, sep='\t')
    drug_targets = set(drug_df.iloc[:, 1].astype(str).str.upper())

    print("Loading mutations...")
    with open(mutation_path, 'r') as f:
        mutated_proteins = set(line.strip().upper() for line in f if line.strip())

    print("Generating perturbation scores...")
    perturbation_scores = np.zeros(num_nodes)
    for protein, idx in protein_to_idx.items():
        if protein.upper() in drug_targets:
            perturbation_scores[idx] += 0.7
        if protein.upper() in mutated_proteins:
            perturbation_scores[idx] += 1.0
    if perturbation_scores.max() > 0:
        perturbation_scores = perturbation_scores / perturbation_scores.max()
    perturbation_scores = torch.tensor(perturbation_scores, dtype=torch.float)

    print("✅ Real network loaded successfully.")
    # RETURN 8 ITEMS (added 3 extras)
    return (edge_index,                 # 1
            node_features,              # 2
            list(protein_to_idx.keys()),# 3
            perturbation_scores,        # 4
            pagerank_scores,            # 5
            protein_to_idx,             # 6 (NEW)
            drug_targets,               # 7 (NEW)
            mutated_proteins)

# ----------------------------------------------------------------------------
# (All other functions: prepare_edge_data, PPI_GNN_Optimized class, training
#  utilities, etc. are UNCHANGED and are omitted here for brevity)
# ----------------------------------------------------------------------------
def prepare_edge_data(edge_index: torch.Tensor, num_nodes: int, test_ratio=0.15, val_ratio=0.15):
    """Return train/val/test dicts with balanced pos/neg edges."""
    pos_edges = set((min(i, j), max(i, j)) for i, j in edge_index.t().tolist())
    pos_edges = list(pos_edges)
    pos_edges = torch.tensor(pos_edges, dtype=torch.long).t()

    ne = pos_edges.size(1)
    idx = list(range(ne))
    random.shuffle(idx)
    n_test = int(ne * test_ratio)
    n_val = int(ne * val_ratio)
    n_train = ne - n_test - n_val

    train_pos = pos_edges[:, idx[:n_train]]
    val_pos = pos_edges[:, idx[n_train : n_train + n_val]]
    test_pos = pos_edges[:, idx[n_train + n_val :]]

    # negative sampling
    neg_set = set()
    while len(neg_set) < ne:
        u = np.random.randint(0, num_nodes, size=ne)
        v = np.random.randint(0, num_nodes, size=ne)
        for i, j in zip(u, v):
            if i != j:
                e = (min(int(i), int(j)), max(int(i), int(j)))
                if e not in pos_edges.t().tolist():
                    neg_set.add(e)
            if len(neg_set) >= ne:
                break
    neg_edges = torch.tensor(list(neg_set), dtype=torch.long).t()
    neg_idx = list(range(ne))
    random.shuffle(neg_idx)

    train_neg = neg_edges[:, neg_idx[:n_train]]
    val_neg = neg_edges[:, neg_idx[n_train : n_train + n_val]]
    test_neg = neg_edges[:, neg_idx[n_train + n_val :]]

    def pack(pos, neg):
        return {
            "edges": torch.cat([pos, neg], dim=1),
            "labels": torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))]),
        }

    return pack(train_pos, train_neg), pack(val_pos, val_neg), pack(test_pos, test_neg)


# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
class PPIGAT(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128, heads1: int = 4, heads2: int = 2, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden, heads=heads1, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden * heads1)
        self.conv2 = GATv2Conv(hidden * heads1, hidden, heads=heads2, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden * heads2)
        self.proj = nn.Linear(hidden * heads2 * 2 + 2 + 1, 2)  # src|dst|pr_src|pr_dst|pert_effect

    def encode(self, x, ei):
        h = F.dropout(F.relu(self.bn1(self.conv1(x, ei))), p=0.2, training=self.training)
        h = F.dropout(F.relu(self.bn2(self.conv2(h, ei))), p=0.2, training=self.training)
        return h
    def forward(self, x, ei, pred_edges, pert, pr):
        z = self.encode(x, ei)
        src, dst = pred_edges
        h = torch.cat([z[src], z[dst], pr[src].unsqueeze(1), pr[dst].unsqueeze(1)], dim=1)
        pert_eff = (pert[src] * pert[dst]).unsqueeze(1)
        h = torch.cat([h, pert_eff], dim=1)
        return self.proj(h)
    

    def train_epoch(model, opt, graph_data, train, pert, pr):
     model.train()
     opt.zero_grad()
     out = model(graph_data.x, graph_data.edge_index, train["edges"], pert, pr)
     loss = F.cross_entropy(out, train["labels"].long())
     loss.backward()
     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
     opt.step()
    
    return loss.item()


def evaluate(model, graph_data, data, pert, pr):
    model.eval()
    with torch.no_grad():
        logits = model(graph_data.x, graph_data.edge_index, data["edges"], pert, pr)
        probs = F.softmax(logits, dim=1)[:, 1]
        preds = (probs > 0.5).long()
    return {
        "accuracy": accuracy_score(data["labels"], preds),
        "precision": precision_score(data["labels"], preds, zero_division=0),
        "recall": recall_score(data["labels"], preds, zero_division=0),
        "f1": f1_score(data["labels"], preds, zero_division=0),
        "auc": roc_auc_score(data["labels"], probs),
    }, probs
   

def main():
    try:
        print("Starting PPI network analysis with perturbation and PageRank...")
        start_time = time.time()

        # ---- 1. Load full graph ----
        (edge_index,
         node_features,
         protein_names,
         perturbation_scores,
         pagerank_scores,
         protein_to_idx,
         drug_targets,
         mutated_proteins) = load_real_ppi_data(
                                ppi_path, drug_path, mutation_path)

        # ---- 2. Build sub‑graph (intersection of mutated & drug‑target proteins) ----
        sub_edge_index, sub_feats, sub_pert, sub_pr, old2new = \
            extract_common_subgraph(edge_index,
                                    protein_to_idx,
                                    set(drug_targets),
                                    set(mutated_proteins),
                                    node_features,
                                    perturbation_scores,
                                    pagerank_scores)
        # Optional visual sanity check
        # G_sub = nx.Graph(); G_sub.add_edges_from(sub_edge_index.t().tolist())
        # plt.figure(figsize=(8,6)); nx.draw_networkx(G_sub, node_size=20, with_labels=False)
        # plt.title("Subgraph of Common Proteins"); plt.show()

        # ---- 3. Replace tensors with sub‑graph tensors ----
        edge_index          = sub_edge_index
        node_features       = sub_feats
        perturbation_scores = sub_pert
        pagerank_scores     = sub_pr
        protein_names       = [protein_names[old] for old in sorted(old2new)]


        # ---- 4. Everything from here remains identical ----
        train_data, val_data, test_data = prepare_edge_data(
            edge_index, num_nodes=node_features.size(0), test_ratio=0.1, val_ratio=0.1
        )

        # Create PyG data object
        graph_data = Data(
            x=node_features,
            edge_index=edge_index
        )

        graph_data = Data(x=node_features, edge_index=edge_index)
        in_channels = node_features.size(1)
        hidden_channels = 64
        model = PPI_GNN_Optimized(in_channels, hidden_channels, heads=8, dropout=0.2, use_pagerank=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

        epochs = 100
        best_val_f1 = 0
        best_state = None
        patience = 15
        counter = 0
        print("\nTraining started...")
        for epoch in range(epochs):
            loss = train_epoch_optimized(model, optimizer, graph_data, train_data,
                                         perturbation_scores, pagerank_scores, scheduler)
            normal_val_metrics, perturbed_val_metrics, _, _ = evaluate_optimized(
                model, graph_data, val_data, perturbation_scores, pagerank_scores)
            if perturbed_val_metrics['f1'] > best_val_f1:
                best_val_f1 = perturbed_val_metrics['f1']
                best_state = model.state_dict().copy()
                counter = 0
            else:
                counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}"); break
            if epoch % 5 == 0 or epoch == epochs-1:
                print(f"Epoch {epoch:3d}: Loss {loss:.4f}, Val F1 {normal_val_metrics['f1']:.4f}/{perturbed_val_metrics['f1']:.4f}")

        if best_state is not None:
            model.load_state_dict(best_state)
        print("\nEvaluating on test set...")
        normal_metrics, perturbed_metrics, normal_probs, perturbed_probs = evaluate_optimized(
            model, graph_data, test_data, perturbation_scores, pagerank_scores)
        print("\nNormal prediction metrics:")
        for k, v in normal_metrics.items():
            print(f"  {k}: {v:.4f}")
        print("\nPerturbed prediction metrics:")
        for k, v in perturbed_metrics.items():
            print(f"  {k}: {v:.4f}")
        edge_pert = torch.tensor([perturbation_scores[s].item()*perturbation_scores[d].item() for s, d in test_data['edges'].t()])
        edge_pr   = torch.tensor([(pagerank_scores[s].item()+pagerank_scores[d].item())/2 for s, d in test_data['edges'].t()])
        compare_predictions(test_data, normal_probs, perturbed_probs,
                            perturbation_scores, pagerank_scores, protein_names, top_n=20)
        plot_comparison(test_data, normal_probs, perturbed_probs, edge_pert, edge_pr)
        torch.save({'model_state_dict': model.state_dict()}, 'ppi_gnn_model.pth')
        end_time = time.time(); print(f"\nExecution completed in {end_time-start_time:.2f} s")

    except Exception as e:
        print(f"Error during execution: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
