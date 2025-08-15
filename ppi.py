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
import matplotlib
matplotlib.use('Agg')
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
    
    # Create simple node features
    print("Generating node features...")
    node_features = torch.eye(num_nodes)  # One-hot features

    # Load drug-target interactions
    print("Loading drug targets...")
    drug_df = pd.read_csv(drug_path, sep='\t')
    drug_targets = set(drug_df.iloc[:, 1].astype(str).str.upper())

    # Load mutated proteins
    print("Loading mutations...")
    with open(mutation_path, 'r') as f:
        mutated_proteins = set(line.strip().upper() for line in f if line.strip())

    # Generate perturbation scores based on real data
    print("Generating perturbation scores...")
    perturbation_scores = np.zeros(num_nodes)

    for protein, idx in protein_to_idx.items():
        if protein.upper() in drug_targets:
            perturbation_scores[idx] += 0.7  # Higher score for drug targets
        if protein.upper() in mutated_proteins:
            perturbation_scores[idx] += 1.0  # Highest score for mutations

    # Normalize scores
    if perturbation_scores.max() > 0:
        perturbation_scores = perturbation_scores / perturbation_scores.max()
    perturbation_scores = torch.tensor(perturbation_scores, dtype=torch.float)

    print("✅ Real network loaded successfully.")
    return edge_index, node_features, list(protein_to_idx.keys()), perturbation_scores

def prepare_edge_data(edge_index, num_nodes, test_ratio=0.1, val_ratio=0.1):
    """Prepare edge data for training with real data only"""
    # Convert edge_index to set of tuples for faster lookup
    edge_set = set()
    for i in range(edge_index.size(1)):
        src, dst = int(edge_index[0, i].item()), int(edge_index[1, i].item())
        edge_set.add((min(src, dst), max(src, dst)))
    
    unique_edges = torch.tensor(list(edge_set), dtype=torch.long).t()
    
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
    num_neg_edges_needed = num_edges
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
    def __init__(self, in_channels, hidden_channels, heads=8, dropout=0.2):
        super(PPI_GNN_Optimized, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels * 2, heads=4, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2 * 4)
        self.conv3 = GATv2Conv(hidden_channels * 2 * 4, hidden_channels, heads=1, dropout=dropout)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 4, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, 2)
        )
        
        self.perturbed_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 1, hidden_channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 4, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, 2)
        )
        
    def encode(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)
        
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        
        return x3
    
    def decode(self, z, edge_index):
        src, dst = edge_index
        edge_features = torch.cat([z[src], z[dst]], dim=1)
        return self.edge_predictor(edge_features)
    
    def decode_perturbed(self, z, edge_index, perturbation_scores):
        src, dst = edge_index
        pert_effect = (perturbation_scores[src] * perturbation_scores[dst]).unsqueeze(1)
        edge_features = torch.cat([z[src], z[dst], pert_effect], dim=1)
        return self.perturbed_predictor(edge_features)
    
    def forward(self, x, edge_index, pred_edges, perturbation_scores=None):
        z = self.encode(x, edge_index)
        normal_pred = self.decode(z, pred_edges)
        perturbed_pred = self.decode_perturbed(z, pred_edges, perturbation_scores) if perturbation_scores is not None else normal_pred
        return normal_pred, perturbed_pred

def train_epoch_optimized(model, optimizer, graph_data, train_data, perturbation_scores=None, scheduler=None):
    """Train for one epoch with improved techniques"""
    model.train()
    optimizer.zero_grad()
    
    # Get normal and perturbed predictions
    normal_pred, perturbed_pred = model(
        graph_data.x, 
        graph_data.edge_index, 
        train_data['edges'],
        perturbation_scores
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
    loss = 0.4 * normal_loss + 0.6 * perturbed_loss
    
    # Backpropagation
    loss.backward()
    
    # Gradient clipping to prevent explosion
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Step learning rate scheduler if provided
    if scheduler is not None:
        scheduler.step()
    
    return loss.item()
def evaluate_optimized(model, graph_data, data, perturbation_scores=None):
    model.eval()
    with torch.no_grad():
        normal_pred, perturbed_pred = model(
            graph_data.x, 
            graph_data.edge_index, 
            data['edges'],
            perturbation_scores
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

def compare_predictions(test_data, normal_probs, perturbed_probs, perturbation_scores, node_names, top_n=20):
    prob_diffs = perturbed_probs - normal_probs
    abs_diffs = torch.abs(prob_diffs)
    sorted_indices = torch.argsort(abs_diffs, descending=True)
    
    edge_perturbations = []
    for i in range(test_data['edges'].size(1)):
        src, dst = test_data['edges'][0, i], test_data['edges'][1, i]
        edge_perturbations.append(perturbation_scores[src].item() * perturbation_scores[dst].item())
    
    edge_perturbations = torch.tensor(edge_perturbations)
    
    print(f"\nTop {top_n} edges with largest prediction differences:")
    print("----------------------------------------------------------------------")
    print("| {:^10} | {:^10} | {:^8} | {:^8} | {:^8} | {:^10} | {:^8} |".format(
        "Source", "Target", "Normal", "Perturbed", "Diff", "Edge Pert", "True"
    ))
    print("----------------------------------------------------------------------")
    
    for i in range(min(top_n, len(sorted_indices))):
        idx = sorted_indices[i].item()
        src_idx = test_data['edges'][0, idx].item()
        dst_idx = test_data['edges'][1, idx].item()
        
        src_name = node_names[src_idx][-10:] if len(node_names[src_idx]) > 10 else node_names[src_idx]
        dst_name = node_names[dst_idx][-10:] if len(node_names[dst_idx]) > 10 else node_names[dst_idx]
        
        print("| {:^10} | {:^10} | {:.6f} | {:.6f} | {:+.6f} | {:.6f} | {:^8} |".format(
            src_name, dst_name,
            normal_probs[idx].item(),
            perturbed_probs[idx].item(),
            prob_diffs[idx].item(),
            edge_perturbations[idx].item(),
            test_data['labels'][idx].item()
        ))
    
    print("----------------------------------------------------------------------")
    print(f"\nCorrelation between edge perturbation and prediction difference: {np.corrcoef(edge_perturbations.numpy(), abs_diffs.numpy())[0, 1]:.4f}")

def plot_comparison(test_data, normal_probs, perturbed_probs, edge_perturbations):
    """
    Plot comparison between normal and perturbed predictions
    
    Args:
        test_data: Test data containing edges and labels
        normal_probs: Probability predictions without perturbation
        perturbed_probs: Probability predictions with perturbation
        edge_perturbations: Perturbation values for each edge
    """
    # Convert to numpy for plotting
    normal_np = normal_probs.numpy()
    perturbed_np = perturbed_probs.numpy()
    true_labels = test_data['labels'].numpy()
    edge_pert_np = edge_perturbations.numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
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
    
    # 3. Edge perturbation vs prediction difference
    axes[1, 0].scatter(edge_pert_np, np.abs(diff), alpha=0.3)
    axes[1, 0].set_xlabel('Edge Perturbation Score')
    axes[1, 0].set_ylabel('Absolute Prediction Difference')
    axes[1, 0].set_title('Edge Perturbation vs Prediction Difference')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add trend line with error handling
    try:
        # Filter out any NaN or infinite values
        valid_mask = np.isfinite(edge_pert_np) & np.isfinite(np.abs(diff))
        if np.sum(valid_mask) > 1:  # Need at least 2 points to fit a line
            z = np.polyfit(edge_pert_np[valid_mask], np.abs(diff)[valid_mask], 1)
            p = np.poly1d(z)
            x_sorted = np.sort(edge_pert_np[valid_mask])
            axes[1, 0].plot(x_sorted, p(x_sorted), "r--", alpha=0.7)
    except Exception as e:
        print(f"Could not fit trend line: {e}")
    
    # 4. ROC curves
    from sklearn.metrics import roc_curve
    
    fpr_normal, tpr_normal, _ = roc_curve(true_labels, normal_np)
    fpr_perturbed, tpr_perturbed, _ = roc_curve(true_labels, perturbed_np)
    
    axes[1, 1].plot(fpr_normal, tpr_normal, label='Normal')
    axes[1, 1].plot(fpr_perturbed, tpr_perturbed, label='Perturbed')
    axes[1, 1].plot([0, 1], [0, 1], 'k--')
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Curve Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add overall title and adjust layout
    plt.tight_layout()
    plt.suptitle('Comparison of Normal vs Perturbed PPI Predictions', fontsize=16, y=1.02)
    
    # Save figure
    plt.savefig('ppi_perturbation_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved comparison plot to 'ppi_perturbation_comparison.png'")


def export_network_analysis(model, graph_data, node_names, perturbation_scores, output_file='network_analysis.csv'):
    model.eval()
    with torch.no_grad():
        node_embeddings = model.encode(graph_data.x, graph_data.edge_index)
    
    results = []
    edge_index = graph_data.edge_index.numpy()
    
    for node_idx in range(len(node_names)):
        outgoing_edges = edge_index[1, edge_index[0, :] == node_idx]
        node_pert = perturbation_scores[node_idx].item()
        
        if len(outgoing_edges) > 0:
            # Fixed the torch.full() syntax here
            edge_pairs = torch.stack([
                torch.full((len(outgoing_edges),), node_idx, dtype=torch.long),
                torch.tensor(outgoing_edges, dtype=torch.long)
            ])
            
            normal_preds, perturbed_preds = model(
                graph_data.x, graph_data.edge_index, edge_pairs, perturbation_scores
            )
            
            normal_probs = F.softmax(normal_preds, dim=1)[:, 1]
            perturbed_probs = F.softmax(perturbed_preds, dim=1)[:, 1]
            
            avg_diff = (perturbed_probs - normal_probs).mean().item()
            abs_avg_diff = torch.abs(perturbed_probs - normal_probs).mean().item()
        else:
            avg_diff = abs_avg_diff = 0.0
        
        results.append({
            'node_id': node_idx,
            'node_name': node_names[node_idx],
            'perturbation_score': node_pert,
            'degree': len(outgoing_edges),
            'avg_prediction_diff': avg_diff,
            'abs_avg_prediction_diff': abs_avg_diff
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Exported node analysis to {output_file}")
    
    print("\nSummary statistics:")
    print(f"Average node degree: {df['degree'].mean():.2f}")
    print(f"Average perturbation score: {df['perturbation_score'].mean():.4f}")
    print(f"Average absolute prediction difference: {df['abs_avg_prediction_diff'].mean():.4f}")
    
    return df.sort_values('perturbation_score', ascending=False).head(10)


def visualize_network(edge_index, node_names, perturbation_scores, top_nodes=None):
    """
    Visualize the protein interaction network with perturbation highlights
    
    Args:
        edge_index: Edge indices
        node_names: List of node names
        perturbation_scores: Perturbation scores
        top_nodes: List of top node indices to highlight
    """
    # Convert to NetworkX graph for visualization
    G = nx.Graph()
    
    # Add nodes with perturbation attributes
    for i, name in enumerate(node_names):
        G.add_node(i, name=name, perturbation=perturbation_scores[i].item())
    
    # Add edges (only one direction since NetworkX uses undirected graphs by default)
    edge_set = set()
    for i in range(edge_index.size(1)):
        src, dst = int(edge_index[0, i].item()), int(edge_index[1, i].item())
        if (src, dst) not in edge_set and (dst, src) not in edge_set:
            G.add_edge(src, dst)
            edge_set.add((src, dst))
    
    # If top_nodes is provided, create a subgraph of top nodes and their neighbors
    if top_nodes is not None:
        nodes_to_keep = set(top_nodes)
        for node in top_nodes:
            nodes_to_keep.update(G.neighbors(node))
        G = G.subgraph(nodes_to_keep)
    
    # Set up plot
    plt.figure(figsize=(12, 12))
    
    # Get positions using spring layout
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
    # Get node colors based on perturbation scores
    node_colors = [G.nodes[n].get('perturbation', 0.0) for n in G.nodes()]
    
    # Draw nodes with size based on degree
    node_sizes = [20 + 5 * G.degree(n) for n in G.nodes()]
    
    # Draw network - now we store the node collection for the colorbar
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_colors, 
        node_size=node_sizes,
        cmap=plt.cm.plasma, 
        alpha=0.9
    )
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    
    # If visualizing a subgraph, add labels for better interpretation
    if top_nodes is not None:
        labels = {n: node_names[n] for n in G.nodes() if n in top_nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # Add colorbar using the node collection we created
    plt.colorbar(nodes, label='Perturbation Score')
    
    plt.title('Protein-Protein Interaction Network with Perturbation Scores')
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure
    plt.savefig('ppi_network_visualization.png', dpi=300, bbox_inches='tight')
    print("Saved network visualization to 'ppi_network_visualization.png'")


def main():
    print("Starting PPI Network Analysis with Real Data Only")
    
    try:
        # Load real data
        edge_index, node_features, node_names, perturbation_scores = load_real_ppi_data(
            ppi_path, drug_path, mutation_path
        )
        num_nodes = len(node_names)
        
        # Prepare data splits
        train_data, val_data, test_data = prepare_edge_data(
            edge_index, num_nodes, test_ratio=0.15, val_ratio=0.15
        )
        
        # Create graph data object
        graph_data = Data(x=node_features, edge_index=edge_index)
        
        # Initialize model
        model = PPI_GNN_Optimized(
            in_channels=node_features.size(1),
            hidden_channels=128,
            heads=8,
            dropout=0.25
        )
        
        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5
        )
        
        # Training loop
        best_val_auc = 0.0
        patience = 10
        patience_counter = 0
        
        print("\nStarting training...")
        for epoch in range(1, 101):
            loss = train_epoch_optimized(model, optimizer, graph_data, train_data, perturbation_scores)
            
            normal_metrics, perturbed_metrics, _, _ = evaluate_optimized(
                model, graph_data, val_data, perturbation_scores
            )
            
            current_val_auc = 0.5 * (normal_metrics['auc'] + perturbed_metrics['auc'])
            scheduler.step(current_val_auc)
            
            print(f"Epoch {epoch:03d}: Loss: {loss:.4f}, Val AUC: {current_val_auc:.4f}")
            
            if current_val_auc > best_val_auc:
                best_val_auc = current_val_auc
                patience_counter = 0
                torch.save(model.state_dict(), 'best_ppi_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping")
                    break
        
        # Load best model
        try:
            model.load_state_dict(torch.load('best_ppi_model.pt'))
        except:
            print("Using current model state")
        
        # Final evaluation
        print("\nFinal Evaluation:")
        normal_test_metrics, perturbed_test_metrics, normal_probs, perturbed_probs = evaluate_optimized(
            model, graph_data, test_data, perturbation_scores
        )
        
        print("\nNormal Predictions:")
        for metric, value in normal_test_metrics.items():
            print(f"  {metric}: {value:.4f}")
            
        print("\nPerturbed Predictions:")
        for metric, value in perturbed_test_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Analysis
        edge_perturbations = []
        for i in range(test_data['edges'].size(1)):
            src, dst = test_data['edges'][0, i], test_data['edges'][1, i]
            edge_perturbations.append(perturbation_scores[src].item() * perturbation_scores[dst].item())
        edge_perturbations = torch.tensor(edge_perturbations)
        
        compare_predictions(test_data, normal_probs, perturbed_probs, perturbation_scores, node_names)
        plot_comparison(test_data, normal_probs, perturbed_probs, edge_perturbations)
        
        top_nodes_df = export_network_analysis(model, graph_data, node_names, perturbation_scores)
        visualize_network(edge_index, node_names, perturbation_scores, top_nodes_df['node_id'].tolist())
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()