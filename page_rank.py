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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import time

# Paths to data files
ppi_path = "/mnt/c/Desktop/PPI/Biogrid.txt"
drug_path = "/mnt/c/Desktop/PPI/ChG-Miner_miner-chem-gene.tsv"
mutation_path = "/mnt/c/Desktop/PPI/S4191.txt"
# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def calculate_pagerank(edge_index, num_nodes, alpha=0.85, max_iterations=100, tol=1e-6):
    """
    Calculate PageRank for each node in the graph
    
    Args:
        edge_index: Edge indices tensor of shape [2, num_edges]
        num_nodes: Number of nodes in the graph
        alpha: Damping factor (default: 0.85)
        max_iterations: Maximum number of iterations (default: 100)
        tol: Convergence tolerance (default: 1e-6)
        
    Returns:
        PageRank scores tensor of shape [num_nodes]
    """
    print("Calculating PageRank scores...")
    
    # Convert to NetworkX graph for efficient PageRank calculation
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    edge_list = edge_index.t().numpy()
    for src, dst in edge_list:
        G.add_edge(int(src), int(dst))
    
    # Calculate PageRank
    pagerank_dict = nx.pagerank(G, alpha=alpha, max_iter=max_iterations, tol=tol)
    
    # Convert to tensor
    pagerank_scores = torch.zeros(num_nodes, dtype=torch.float)
    for node, score in pagerank_dict.items():
        pagerank_scores[node] = score
    
    # Normalize to [0, 1] range
    if pagerank_scores.max() > 0:
        pagerank_scores = pagerank_scores / pagerank_scores.max()
    
    print(f"PageRank calculation complete. Min: {pagerank_scores.min():.6f}, Max: {pagerank_scores.max():.6f}")
    return pagerank_scores

def load_real_ppi_data(ppi_path, drug_path, mutation_path):
    """Load real PPI data and generate perturbation scores based on drug targets and mutations"""
    print("🔍 Loading real datasets...")

    # 1. Load PPI edges from BioGRID (MITAB format)
    print("Loading BioGRID interactions...")
    ppi_df = pd.read_csv(ppi_path, sep='\t', comment='#', header=None, low_memory=False)
    col_A, col_B = 0, 1  # Columns with protein identifiers
    proteins_A = ppi_df[col_A].astype(str).str.split(":", expand=True)[1]
    proteins_B = ppi_df[col_B].astype(str).str.split(":", expand=True)[1]
    all_proteins = pd.concat([proteins_A, proteins_B]).unique()
    
    # Map protein IDs to node indices
    protein_to_idx = {protein: idx for idx, protein in enumerate(sorted(all_proteins))}
    idx_to_protein = {v: k for k, v in protein_to_idx.items()}
    num_nodes = len(protein_to_idx)
    print(f"Number of unique proteins: {num_nodes}")
    
    # Build edge index
    edges = []
    for a, b in zip(proteins_A, proteins_B):
        if a in protein_to_idx and b in protein_to_idx:
            edges.append([protein_to_idx[a], protein_to_idx[b]])
            edges.append([protein_to_idx[b], protein_to_idx[a]])  # Bidirectional

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    # Create one-hot node features
    print("Generating node features...")
    one_hot_features = torch.eye(num_nodes)
    
    # Calculate PageRank for nodes
    pagerank_scores = calculate_pagerank(edge_index, num_nodes)
    
    # Combine one-hot features with PageRank scores
    # Add PageRank as an additional feature column
    node_features = torch.cat([one_hot_features, pagerank_scores.unsqueeze(1)], dim=1)
    
    # Load drug-target interactions
    print("Loading drug targets...")
    drug_df = pd.read_csv(drug_path, sep='\t')
    drug_effects = {}  # Initialize the dictionary
    
    # Process drug effects (assuming inhibition by default since we don't have mechanism data)
    for _, row in drug_df.iterrows():
        protein = str(row.iloc[1]).upper()  # Assuming protein ID is in second column
        drug_effects[protein] = -1  # Default to inhibition effect
    
    # Load mutation effects
    print("Loading mutations...")
    mutation_df = pd.read_csv(mutation_path, sep='\t')
    mutation_ddg = {}
    for _, row in mutation_df.iterrows():
        protein = str(row['Partner1']).upper()
        try:
            ddg = float(row['DDGexp'])
            mutation_ddg[protein] = ddg
        except Exception:
            continue

    # Generate perturbation scores
    print("Generating perturbation scores...")
    perturbation_scores = np.zeros(num_nodes)
    for protein, idx in protein_to_idx.items():
            protein_upper = protein.upper()
            if protein_upper in drug_effects:
                # Scale drug effects
                perturbation_scores[idx] += drug_effects[protein_upper] * 0.5
            if protein_upper in mutation_ddg:
                # Scale mutation effects based on DDGexp magnitude
                ddg = mutation_ddg[protein_upper]
                perturbation_scores[idx] += -np.tanh(ddg)  # Use tanh for bounded scaling
        
        # Normalize without losing sign
    max_abs = np.abs(perturbation_scores).max()
    if max_abs > 0:
        perturbation_scores = perturbation_scores / max_abs
    perturbation_scores = torch.tensor(perturbation_scores, dtype=torch.float)

    print("✅ Real network loaded successfully.")
    return edge_index, node_features, list(protein_to_idx.keys()), perturbation_scores, pagerank_scores

def prepare_edge_data(edge_index, num_nodes, test_ratio=0.1, val_ratio=0.1):
    """Prepare edge data for training with real data only"""
    # First create edge_set
    edge_set = set()
    for i in range(edge_index.size(1)):
        src, dst = int(edge_index[0, i].item()), int(edge_index[1, i].item())
        edge_set.add((min(src, dst), max(src, dst)))
    
    # Now calculate num_neg_edges_needed
    num_neg_edges_needed = len(edge_set) * 2
    
    unique_edges = torch.tensor(list(edge_set), dtype=torch.long).t()
    
    # Rest of the function remains the same
    # Split positive edges
    num_edges = unique_edges.size(1)
    indices = list(range(num_edges))
    random.shuffle(indices)
    
    test_size = int(num_edges * test_ratio)
    val_size = int(num_edges * val_ratio)
    train_size = num_edges - test_size - val_size
    
    train_pos_edges = unique_edges[:, indices[:train_size]]
    val_pos_edges = unique_edges[:, indices[train_size:train_size+val_size]]
    test_pos_edges = unique_edges[:, indices[train_size+val_size:]]
    
    # Generate negative edges
    negative_edges = []
    pbar = tqdm(total=num_neg_edges_needed, desc="Generating negative edges")
    
    trials = 0
    max_trials = num_neg_edges_needed * 10
    
    while len(negative_edges) < num_neg_edges_needed and trials < max_trials:
        batch_size = min(10000, num_neg_edges_needed - len(negative_edges))
        src_nodes = np.random.randint(0, num_nodes, size=batch_size)
        dst_nodes = np.random.randint(0, num_nodes, size=batch_size)
        
        for src, dst in zip(src_nodes, dst_nodes):
            if src != dst:
                edge = (min(int(src), int(dst)), max(int(src), int(dst)))
                if edge not in edge_set and edge not in negative_edges:
                    negative_edges.append(edge)
                    pbar.update(1)
                    if len(negative_edges) >= num_neg_edges_needed:
                        break
            trials += 1
    
    pbar.close()
    print(f"Generated {len(negative_edges)} negative edges")
    
    # Rest of your existing code...
    
    # Split negative edges
    neg_edges = torch.tensor(negative_edges, dtype=torch.long).t()
    neg_indices = list(range(neg_edges.size(1)))
    random.shuffle(neg_indices)
    
    train_neg_size = int(len(neg_indices) * (train_size / num_edges))
    val_neg_size = int(len(neg_indices) * (val_size / num_edges))
    
    train_neg_edges = neg_edges[:, neg_indices[:train_neg_size]]
    val_neg_edges = neg_edges[:, neg_indices[train_neg_size:train_neg_size+val_neg_size]]
    test_neg_edges = neg_edges[:, neg_indices[train_neg_size+val_neg_size:]]
    
    # Create labels and combine
    train_data = {
        'edges': torch.cat([train_pos_edges, train_neg_edges], dim=1),
        'labels': torch.cat([
            torch.ones(train_pos_edges.size(1)), 
            torch.zeros(train_neg_edges.size(1))
        ])
    }
    
    val_data = {
        'edges': torch.cat([val_pos_edges, val_neg_edges], dim=1),
        'labels': torch.cat([
            torch.ones(val_pos_edges.size(1)), 
            torch.zeros(val_neg_edges.size(1))
        ])
    }
    
    test_data = {
        'edges': torch.cat([test_pos_edges, test_neg_edges], dim=1),
        'labels': torch.cat([
            torch.ones(test_pos_edges.size(1)),
            torch.zeros(test_neg_edges.size(1))
        ])
    }
    
    print(f"Train edges: {train_data['edges'].size(1)}, Val edges: {val_data['edges'].size(1)}, Test edges: {test_data['edges'].size(1)}")
    return train_data, val_data, test_data

class PPI_GNN_Optimized(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=8, dropout=0.3, use_pagerank=True):
        super().__init__()
        self.use_pagerank = use_pagerank
        
        # Increase model capacity
        self.conv1 = GATv2Conv(in_channels, hidden_channels * 2, heads=heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_channels * 2 * heads)
        
        self.conv2 = GATv2Conv(hidden_channels * 2 * heads, hidden_channels, heads=4, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 4)
        
        self.conv3 = GATv2Conv(hidden_channels * 4, hidden_channels, heads=1, dropout=dropout)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        self.res_proj = nn.Linear(hidden_channels * 2 * heads, hidden_channels)

        edge_input_size = hidden_channels * 2
        if use_pagerank:
            edge_input_size += 2  # source/target PageRank

        self.edge_predictor = nn.Sequential(
            nn.Linear(edge_input_size, hidden_channels * 4),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 4, hidden_channels * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, 2)
        )

        perturbed_input_size = edge_input_size + 1
        self.perturbed_predictor = nn.Sequential(
            nn.Linear(perturbed_input_size, hidden_channels * 4),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 4, hidden_channels * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, 2)
        )

    def encode(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1, 0.1)
        x1 = F.dropout(x1, p=0.3, training=self.training)

        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.leaky_relu(x2, 0.1)
        x2 = F.dropout(x2, p=0.3, training=self.training)

        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.leaky_relu(x3, 0.1)

        # Project x1 to match x3's shape before adding
        x1_proj = self.res_proj(x1)
        return x3 + x1_proj

    def decode(self, z, edge_index, pagerank_scores=None):
        src, dst = edge_index
        edge_features = torch.cat([z[src], z[dst]], dim=1)
        if self.use_pagerank and pagerank_scores is not None:
            src_pr = pagerank_scores[src].unsqueeze(1)
            dst_pr = pagerank_scores[dst].unsqueeze(1)
            edge_features = torch.cat([edge_features, src_pr, dst_pr], dim=1)
        return self.edge_predictor(edge_features)

    def decode_perturbed(self, z, edge_index, perturbation_scores, pagerank_scores=None):
        src, dst = edge_index
        edge_features = torch.cat([z[src], z[dst]], dim=1)
        if self.use_pagerank and pagerank_scores is not None:
            src_pr = pagerank_scores[src].unsqueeze(1)
            dst_pr = pagerank_scores[dst].unsqueeze(1)
            edge_features = torch.cat([edge_features, src_pr, dst_pr], dim=1)
        pert_effect = (perturbation_scores[src] * perturbation_scores[dst]).unsqueeze(1)
        edge_features = torch.cat([edge_features, pert_effect], dim=1)
        return self.perturbed_predictor(edge_features)

    def forward(self, x, edge_index, pred_edges, perturbation_scores=None, pagerank_scores=None):
        z = self.encode(x, edge_index)
        normal_pred = self.decode(z, pred_edges, pagerank_scores)
        perturbed_pred = self.decode_perturbed(z, pred_edges, perturbation_scores, pagerank_scores) if perturbation_scores is not None else normal_pred
        return normal_pred, perturbed_pred

def train_epoch_optimized(model, optimizer, graph_data, train_data, perturbation_scores=None, pagerank_scores=None, scheduler=None):
    """Train for one epoch with improved techniques"""
    model.train()
    optimizer.zero_grad()
    
    # Get normal and perturbed predictions
    normal_pred, perturbed_pred = model(
        graph_data.x, 
        graph_data.edge_index, 
        train_data['edges'],
        perturbation_scores,
        pagerank_scores
    )
    
    # Ensure labels are long integers for bincount
    labels = train_data['labels'].long()  # Convert to long integer type
    
    # Calculate class weights to handle imbalance
    class_counts = torch.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts.float())
    
    # Weighted loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    normal_loss = criterion(normal_pred, labels)
    perturbed_loss = criterion(perturbed_pred, labels)
    
    # Combined loss with emphasis on perturbed predictions
    normal_weight = 0.3
    perturbed_weight = 0.7
    loss = normal_weight * normal_loss + perturbed_weight * perturbed_loss
    
    # Backpropagation
    loss.backward()
    
    # Gradient clipping to prevent explosion
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Step learning rate scheduler if provided
    if scheduler is not None:
        scheduler.step()
    
    return loss.item()

def evaluate_optimized(model, graph_data, data, perturbation_scores=None, pagerank_scores=None):
    model.eval()
    with torch.no_grad():
        normal_pred, perturbed_pred = model(
            graph_data.x, 
            graph_data.edge_index, 
            data['edges'],
            perturbation_scores,
            pagerank_scores
        )
        
        normal_probs = F.softmax(normal_pred, dim=1)[:, 1]
        perturbed_probs = F.softmax(perturbed_pred, dim=1)[:, 1]
        
        normal_preds = (normal_probs > 0.5).long()
        perturbed_preds = (perturbed_probs > 0.5).long()
        
        normal_metrics = {
            'accuracy': accuracy_score(data['labels'], normal_preds),
            'auc': roc_auc_score(data['labels'], normal_probs),
            'precision': precision_score(data['labels'], normal_preds, zero_division=0),
            'recall': recall_score(data['labels'], normal_preds, zero_division=0),
            'f1': f1_score(data['labels'], normal_preds, zero_division=0),
            'specificity': recall_score(data['labels'], normal_preds, pos_label=0, zero_division=0),
            'interacted_recall': recall_score(data['labels'], normal_preds, pos_label=1, zero_division=0)
        }
        
        perturbed_metrics = {
            'accuracy': accuracy_score(data['labels'], perturbed_preds),
            'auc': roc_auc_score(data['labels'], perturbed_probs),
            'precision': precision_score(data['labels'], perturbed_preds, zero_division=0),
            'recall': recall_score(data['labels'], perturbed_preds, zero_division=0),
            'f1': f1_score(data['labels'], perturbed_preds, zero_division=0),
            'specificity': recall_score(data['labels'], perturbed_preds, pos_label=0, zero_division=0),
            'interacted_recall': recall_score(data['labels'], perturbed_preds, pos_label=1, zero_division=0)
        }
        
    return normal_metrics, perturbed_metrics, normal_probs, perturbed_probs

def compare_predictions(test_data, normal_probs, perturbed_probs, perturbation_scores, pagerank_scores, node_names, top_n=20):
    prob_diffs = perturbed_probs - normal_probs
    abs_diffs = torch.abs(prob_diffs)
    sorted_indices = torch.argsort(abs_diffs, descending=True)
    
    edge_perturbations = []
    edge_pageranks = []
    pert_signs = []
    for i in range(test_data['edges'].size(1)):
        src, dst = test_data['edges'][0, i], test_data['edges'][1, i]
        edge_pert = perturbation_scores[src].item() * perturbation_scores[dst].item()
        edge_perturbations.append(edge_pert)
        pert_signs.append("negative" if edge_pert < 0 else "positive" if edge_pert > 0 else "neutral")
        edge_pageranks.append((pagerank_scores[src].item() + pagerank_scores[dst].item()) / 2)
    
    edge_perturbations = torch.tensor(edge_perturbations)
    edge_pageranks = torch.tensor(edge_pageranks)
    
    print(f"\nTop {top_n} edges with largest prediction differences:")
    print("-----------------------------------------------------------------------------------------------------------------")
    print("| {:^10} | {:^10} | {:^8} | {:^8} | {:^8} | {:^10} | {:^10} | {:^8} | {:^10} |".format(
        "Source", "Target", "Normal", "Perturbed", "Diff", "Edge Pert", "Edge PR", "True", "PertType"
    ))
    print("-----------------------------------------------------------------------------------------------------------------")
    
    neg_count = 0
    pos_count = 0
    neu_count = 0
    for i in range(min(top_n, len(sorted_indices))):
        idx = sorted_indices[i].item()
        src_idx = test_data['edges'][0, idx].item()
        dst_idx = test_data['edges'][1, idx].item()
        
        src_name = node_names[src_idx][-10:] if len(node_names[src_idx]) > 10 else node_names[src_idx]
        dst_name = node_names[dst_idx][-10:] if len(node_names[dst_idx]) > 10 else node_names[dst_idx]
        pert_type = pert_signs[idx]
        if pert_type == "negative":
            neg_count += 1
        elif pert_type == "positive":
            pos_count += 1
        else:
            neu_count += 1
        
        print("| {:^10} | {:^10} | {:.6f} | {:.6f} | {:+.6f} | {:.6f} | {:.6f} | {:^8} | {:^10} |".format(
            src_name, dst_name,
            normal_probs[idx].item(),
            perturbed_probs[idx].item(),
            prob_diffs[idx].item(),
            edge_perturbations[idx].item(),
            edge_pageranks[idx].item(),
            test_data['labels'][idx].item(),
            pert_type
        ))
    
    print("-----------------------------------------------------------------------------------------------------------------")
    print(f"Summary: {pos_count} positive, {neg_count} negative, {neu_count} neutral perturbations in top {top_n} edges.")

    # ...rest of your code for correlations...
    # Calculate correlations
    pert_corr = np.corrcoef(edge_perturbations.numpy(), abs_diffs.numpy())[0, 1]
    pr_corr = np.corrcoef(edge_pageranks.numpy(), abs_diffs.numpy())[0, 1]
    
    print(f"\nCorrelation between edge perturbation and prediction difference: {pert_corr:.4f}")
    print(f"Correlation between edge PageRank and prediction difference: {pr_corr:.4f}")

# Update the plot_comparison function to handle numerical stability issues
def plot_comparison(test_data, normal_probs, perturbed_probs, edge_perturbations, edge_pageranks):
    """
    Plot comparison between normal and perturbed predictions with improved numerical stability
    
    Args:
        test_data: Test data containing edges and labels
        normal_probs: Probability predictions without perturbation
        perturbed_probs: Probability predictions with perturbation
        edge_perturbations: Perturbation values for each edge
        edge_pageranks: PageRank values for each edge
    """
    # Convert to numpy for plotting
    normal_np = normal_probs.numpy()
    perturbed_np = perturbed_probs.numpy()
    true_labels = test_data['labels'].numpy()
    edge_pert_np = edge_perturbations.numpy()
    edge_pr_np = edge_pageranks.numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Scatter plot of normal vs perturbed predictions
    axes[0, 0].scatter(normal_np, perturbed_np, alpha=0.3, c=true_labels, cmap='coolwarm')
    axes[0, 0].set_xlabel('Normal Prediction Probability')
    axes[0, 0].set_ylabel('Perturbed Prediction Probability')
    axes[0, 0].set_title('Normal vs Perturbed Predictions')
    axes[0, 0].plot([0, 1], [0, 1], 'k--')  # Diagonal line
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Histogram of prediction differences
    diff = perturbed_np - normal_np
    axes[0, 1].hist(diff, bins=50, alpha=0.7)
    axes[0, 1].set_xlabel('Prediction Difference (Perturbed - Normal)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Prediction Differences')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Edge perturbation vs prediction difference with improved stability
    # Check for NaN or infinite values in edge_pert_np
    valid_pert_mask = np.isfinite(edge_pert_np) & np.isfinite(np.abs(diff))
    if np.sum(valid_pert_mask) > 0:  # Only plot if we have valid data
        axes[0, 2].scatter(edge_pert_np[valid_pert_mask], np.abs(diff)[valid_pert_mask], alpha=0.3)
        axes[0, 2].set_xlabel('Edge Perturbation Score')
        axes[0, 2].set_ylabel('Absolute Prediction Difference')
        axes[0, 2].set_title('Edge Perturbation vs Prediction Difference')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add trend line with robust error handling
        try:
            if np.sum(valid_pert_mask) > 10:  # Need a reasonable number of points
                x_valid = edge_pert_np[valid_pert_mask]
                y_valid = np.abs(diff)[valid_pert_mask]
                
                # Filter to remove outliers if needed
                # Use only non-zero variance data
                if np.var(x_valid) > 1e-10 and np.var(y_valid) > 1e-10:
                    z = np.polyfit(x_valid, y_valid, 1)
                    p = np.poly1d(z)
                    x_sorted = np.sort(x_valid)
                    axes[0, 2].plot(x_sorted, p(x_sorted), "r--", alpha=0.7)
                    
                    # Add correlation info to plot
                    corr = np.corrcoef(x_valid, y_valid)[0, 1]
                    if np.isfinite(corr):  # Only display if correlation is valid
                        axes[0, 2].text(0.05, 0.95, f"Correlation: {corr:.4f}", 
                                      transform=axes[0, 2].transAxes, fontsize=12,
                                      verticalalignment='top')
        except Exception as e:
            print(f"Could not fit trend line for perturbation: {e}")
            # Continue with the plot without the trend line
    else:
        axes[0, 2].text(0.5, 0.5, "Insufficient valid data for plotting", 
                      ha='center', va='center', transform=axes[0, 2].transAxes)
    
    # 4. ROC curves
    from sklearn.metrics import roc_curve
    
    fpr_normal, tpr_normal, _ = roc_curve(true_labels, normal_np)
    fpr_perturbed, tpr_perturbed, _ = roc_curve(true_labels, perturbed_np)
    
    axes[1, 0].plot(fpr_normal, tpr_normal, label='Normal')
    axes[1, 0].plot(fpr_perturbed, tpr_perturbed, label='Perturbed')
    axes[1, 0].plot([0, 1], [0, 1], 'k--')
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curve Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. PageRank vs prediction difference with improved stability
    valid_pr_mask = np.isfinite(edge_pr_np) & np.isfinite(np.abs(diff))
    if np.sum(valid_pr_mask) > 0:
        axes[1, 1].scatter(edge_pr_np[valid_pr_mask], np.abs(diff)[valid_pr_mask], alpha=0.3)
        axes[1, 1].set_xlabel('Edge PageRank Score')
        axes[1, 1].set_ylabel('Absolute Prediction Difference')
        axes[1, 1].set_title('Edge PageRank vs Prediction Difference')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add trend line for PageRank with improved stability
        try:
            if np.sum(valid_pr_mask) > 10:
                x_valid = edge_pr_np[valid_pr_mask]
                y_valid = np.abs(diff)[valid_pr_mask]
                
                if np.var(x_valid) > 1e-10 and np.var(y_valid) > 1e-10:
                    z = np.polyfit(x_valid, y_valid, 1)
                    p = np.poly1d(z)
                    x_sorted = np.sort(x_valid)
                    axes[1, 1].plot(x_sorted, p(x_sorted), "r--", alpha=0.7)
                    
                    # Add correlation to plot
                    corr = np.corrcoef(x_valid, y_valid)[0, 1]
                    if np.isfinite(corr):
                        axes[1, 1].text(0.05, 0.95, f"Correlation: {corr:.4f}", 
                                      transform=axes[1, 1].transAxes, fontsize=12,
                                      verticalalignment='top')
        except Exception as e:
            print(f"Could not fit trend line for PageRank: {e}")
    else:
        axes[1, 1].text(0.5, 0.5, "Insufficient valid data for plotting", 
                      ha='center', va='center', transform=axes[1, 1].transAxes)
    
    # 6. PageRank vs Perturbation correlation with improved stability
    valid_both_mask = np.isfinite(edge_pr_np) & np.isfinite(edge_pert_np)
    if np.sum(valid_both_mask) > 0:
        axes[1, 2].scatter(edge_pr_np[valid_both_mask], edge_pert_np[valid_both_mask], alpha=0.3)
        axes[1, 2].set_xlabel('Edge PageRank Score')
        axes[1, 2].set_ylabel('Edge Perturbation Score')
        axes[1, 2].set_title('PageRank vs Perturbation Score')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add trend line with improved stability
        try:
            if np.sum(valid_both_mask) > 10:
                x_valid = edge_pr_np[valid_both_mask]
                y_valid = edge_pert_np[valid_both_mask]
                
                if np.var(x_valid) > 1e-10 and np.var(y_valid) > 1e-10:
                    z = np.polyfit(x_valid, y_valid, 1)
                    p = np.poly1d(z)
                    x_sorted = np.sort(x_valid)
                    axes[1, 2].plot(x_sorted, p(x_sorted), "r--", alpha=0.7)
                    
                    # Calculate correlation and add to plot
                    corr = np.corrcoef(x_valid, y_valid)[0, 1]
                    if np.isfinite(corr):
                        axes[1, 2].text(0.05, 0.95, f"Correlation: {corr:.4f}", 
                                      transform=axes[1, 2].transAxes, fontsize=12,
                                      verticalalignment='top')
        except Exception as e:
            print(f"Could not fit trend line for PageRank vs Perturbation: {e}")
    else:
        axes[1, 2].text(0.5, 0.5, "Insufficient valid data for plotting", 
                      ha='center', va='center', transform=axes[1, 2].transAxes)
    
    # Add overall title and adjust layout
    plt.tight_layout()
    plt.suptitle('Comparison of Normal vs Perturbed PPI Predictions with PageRank', fontsize=16, y=1.02)
    
    # Save figure
    plt.savefig('ppi_perturbation_pagerank_comparison.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'ppi_perturbation_pagerank_comparison.png'")

# Update the compare_predictions function to handle numerical stability issues
def compare_predictions(test_data, normal_probs, perturbed_probs, perturbation_scores, pagerank_scores, node_names, top_n=20):
    prob_diffs = perturbed_probs - normal_probs
    abs_diffs = torch.abs(prob_diffs)
    sorted_indices = torch.argsort(abs_diffs, descending=True)
    
    edge_perturbations = []
    edge_pageranks = []
    for i in range(test_data['edges'].size(1)):
        src, dst = test_data['edges'][0, i], test_data['edges'][1, i]
        edge_perturbations.append(perturbation_scores[src].item() * perturbation_scores[dst].item())
        # Calculate combined PageRank score for edge
        edge_pageranks.append((pagerank_scores[src].item() + pagerank_scores[dst].item()) / 2)
    
    edge_perturbations = torch.tensor(edge_perturbations)
    edge_pageranks = torch.tensor(edge_pageranks)
    
    print(f"\nTop {top_n} edges with largest prediction differences:")
    print("-----------------------------------------------------------------------------------------------------")
    print("| {:^10} | {:^10} | {:^8} | {:^8} | {:^8} | {:^10} | {:^10} | {:^8} |".format(
        "Source", "Target", "Normal", "Perturbed", "Diff", "Edge Pert", "Edge PR", "True"
    ))
    print("-----------------------------------------------------------------------------------------------------")
    
    for i in range(min(top_n, len(sorted_indices))):
        idx = sorted_indices[i].item()
        src_idx = test_data['edges'][0, idx].item()
        dst_idx = test_data['edges'][1, idx].item()
        
        src_name = node_names[src_idx][-10:] if len(node_names[src_idx]) > 10 else node_names[src_idx]
        dst_name = node_names[dst_idx][-10:] if len(node_names[dst_idx]) > 10 else node_names[dst_idx]
        
        print("| {:^10} | {:^10} | {:.6f} | {:.6f} | {:+.6f} | {:.6f} | {:.6f} | {:^8} |".format(
            src_name, dst_name,
            normal_probs[idx].item(),
            perturbed_probs[idx].item(),
            prob_diffs[idx].item(),
            edge_perturbations[idx].item(),
            edge_pageranks[idx].item(),
            test_data['labels'][idx].item()
        ))
    
    print("-----------------------------------------------------------------------------------------------------")
    
    # Calculate correlations with improved error handling
    # Convert to numpy arrays
    edge_pert_np = edge_perturbations.numpy()
    abs_diffs_np = abs_diffs.numpy()
    edge_pr_np = edge_pageranks.numpy()
    
    # Filter out non-finite values
    valid_pert_mask = np.isfinite(edge_pert_np) & np.isfinite(abs_diffs_np)
    valid_pr_mask = np.isfinite(edge_pr_np) & np.isfinite(abs_diffs_np)
    
    # Only calculate correlations if we have enough valid data
    if np.sum(valid_pert_mask) > 1 and np.var(edge_pert_np[valid_pert_mask]) > 0 and np.var(abs_diffs_np[valid_pert_mask]) > 0:
        pert_corr = np.corrcoef(edge_pert_np[valid_pert_mask], abs_diffs_np[valid_pert_mask])[0, 1]
        print(f"\nCorrelation between edge perturbation and prediction difference: {pert_corr:.4f}")
    else:
        print("\nInsufficient or zero-variance data to calculate perturbation correlation.")
    
    if np.sum(valid_pr_mask) > 1 and np.var(edge_pr_np[valid_pr_mask]) > 0 and np.var(abs_diffs_np[valid_pr_mask]) > 0:
        pr_corr = np.corrcoef(edge_pr_np[valid_pr_mask], abs_diffs_np[valid_pr_mask])[0, 1]
        print(f"Correlation between edge PageRank and prediction difference: {pr_corr:.4f}")
    else:
        print("Insufficient or zero-variance data to calculate PageRank correlation.")
def main():
    try:
        print("Starting PPI network analysis with perturbation and PageRank...")
        start_time = time.time()
        
        # Load real data
        edge_index, node_features, protein_names, perturbation_scores, pagerank_scores = load_real_ppi_data(
            ppi_path, drug_path, mutation_path
        )
        
        # Prepare edge data
        train_data, val_data, test_data = prepare_edge_data(
            edge_index, num_nodes=node_features.size(0), test_ratio=0.1, val_ratio=0.1
        )
        
        # Create PyG data object
        graph_data = Data(
            x=node_features,
            edge_index=edge_index
        )
        
        # Initialize model
        in_channels = node_features.size(1)
        hidden_channels = 64
        model = PPI_GNN_Optimized(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            heads=8,
            dropout=0.2,
            use_pagerank=True
        )
        print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Initialize optimizer and scheduler
        # In main():
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-4)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        epochs = 200
        patience = 20
        best_val_f1 = 0
        best_model_state = None
        counter = 0
        
        print("\nTraining started...")
        for epoch in range(epochs):
            # Train
            loss = train_epoch_optimized(
                model, optimizer, graph_data, train_data,
                perturbation_scores, pagerank_scores, scheduler
            )
            
            # Validate
            normal_val_metrics, perturbed_val_metrics, _, _ = evaluate_optimized(
                model, graph_data, val_data, perturbation_scores, pagerank_scores
            )
            
            # Track best model
            if perturbed_val_metrics['f1'] > best_val_f1:
                best_val_f1 = perturbed_val_metrics['f1']
                best_model_state = model.state_dict().copy()
                counter = 0
            else:
                counter += 1
            
            # Early stopping
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Print progress
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}: Loss: {loss:.4f}, " +
                      f"Val F1 (Normal/Perturbed): {normal_val_metrics['f1']:.4f}/{perturbed_val_metrics['f1']:.4f}, " +
                      f"Val AUC (Normal/Perturbed): {normal_val_metrics['auc']:.4f}/{perturbed_val_metrics['auc']:.4f}")
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        normal_metrics, perturbed_metrics, normal_probs, perturbed_probs = evaluate_optimized(
            model, graph_data, test_data, perturbation_scores, pagerank_scores
        )
        
        print("\nNormal prediction metrics:")
        for key, value in normal_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        print("\nPerturbed prediction metrics:")
        for key, value in perturbed_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Calculate edge-specific metrics
        edge_perturbations = []
        edge_pageranks = []
        for i in range(test_data['edges'].size(1)):
            src, dst = test_data['edges'][0, i], test_data['edges'][1, i]
            edge_perturbations.append(perturbation_scores[src].item() * perturbation_scores[dst].item())
            edge_pageranks.append((pagerank_scores[src].item() + pagerank_scores[dst].item()) / 2)
        
        edge_perturbations = torch.tensor(edge_perturbations)
        edge_pageranks = torch.tensor(edge_pageranks)
        
        # Compare predictions
        compare_predictions(
            test_data, normal_probs, perturbed_probs, 
            perturbation_scores, pagerank_scores, protein_names, top_n=20
        )
        
        # Plot comparison
        print("\nGenerating visualization...")
        plot_comparison(
            test_data, normal_probs, perturbed_probs,
            edge_perturbations, edge_pageranks
        )
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'normal_metrics': normal_metrics,
            'perturbed_metrics': perturbed_metrics
        }, 'ppi_gnn_model.pth')
        print("Model saved as 'ppi_gnn_model.pth'")
        
        # Find edges with largest impact from perturbation
        diff = perturbed_probs - normal_probs
        abs_diff = torch.abs(diff)
        
        # High impact edges
        top_diff_idx = torch.argsort(abs_diff, descending=True)[:100]  # Top 100 most impacted
        high_impact_edges = []
        
        for idx in top_diff_idx:
            src, dst = test_data['edges'][0, idx].item(), test_data['edges'][1, idx].item()
            src_name = protein_names[src]
            dst_name = protein_names[dst]
            high_impact_edges.append({
                'source': src_name,
                'target': dst_name,
                'normal_prob': normal_probs[idx].item(),
                'perturbed_prob': perturbed_probs[idx].item(),
                'difference': diff[idx].item(),
                'edge_perturbation': edge_perturbations[idx].item(),
                'edge_pagerank': edge_pageranks[idx].item(),
                'true_label': test_data['labels'][idx].item()
            })
        
        # Save high impact edges to CSV
        high_impact_df = pd.DataFrame(high_impact_edges)
        high_impact_df.to_csv('high_impact_edges.csv', index=False)
        print("High impact edges saved to 'high_impact_edges.csv'")
        
        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nExecution completed in {execution_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()