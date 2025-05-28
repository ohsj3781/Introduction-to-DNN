# hw02_gcn_link_prediction.py
"""
Homework 2 – GCN Link Prediction (Cora)
===================================================
This script reproduces the full pipeline described in the
“Homework2_GCN_Link_Prediction.pdf”.  Each numbered *STEP* in the
comments lines up with the roadmap supplied earlier.

Usage (CPU example):
    python hw02_gcn_link_prediction.py --epochs 100

Options:
    --data-zip   Path to data.zip   (default: ./data.zip)
    --epochs     Number of epochs   (default: 30)
    --lr         Learning‑rate      (default: 1e-2)
    --hidden     Hidden dim         (default: 16)
    --dropout    Dropout prob       (default: 0.1)
    --patience   Early‑stop window  (default: 10)

Requires PyTorch ≥2.0, pandas, networkx, scikit‑learn, matplotlib and
(torch‑sparse or scipy as a fallback).
"""

# STEP 0 – Imports & seed ------------------------------------------------------
import argparse, copy, random, zipfile, math, time
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# -----------------------------------------------------------------------------
# STEP 1 – Dataset extraction & loading
# -----------------------------------------------------------------------------

def ensure_cora(data_zip: Path, out_dir: Path) -> Path:
    """Extracts *data.zip* (if necessary) so we get  data/cora/*."""
    cora_dir = out_dir / "cora"
    if cora_dir.exists():
        return cora_dir
    assert data_zip.exists(), f"{data_zip} not found"
    with zipfile.ZipFile(data_zip) as z:
        z.extractall(out_dir)
    return cora_dir


def load_cora(cora_dir):
    """Return (nx.Graph, torch.Tensor[node_features])."""
    content = pd.read_csv(f"{cora_dir}/cora.content", sep="\t", header=None)
    cites   = pd.read_csv(f"{cora_dir}/cora.cites",   sep="\t", header=None)

    paper_ids = content[0].values
    id_map = {pid: i for i, pid in enumerate(paper_ids)}

    # node features (1433 dims)
    features = torch.tensor(content.iloc[:, 1:-1].values, dtype=torch.float32)

    # undirected citation graph
    edges = cites.applymap(id_map.get).values
    G = nx.Graph()
    G.add_nodes_from(range(len(paper_ids)))
    G.add_edges_from(edges)
    return G, features


# -----------------------------------------------------------------------------
# STEP 2 – Adjacency (\tilde D^{-1/2}\tilde A\tilde D^{-1/2})
# -----------------------------------------------------------------------------
try:
    from torch_sparse import SparseTensor
except ImportError:
    SparseTensor = None  # fallback later


def build_adj(G):
    N = G.number_of_nodes()
    src, dst = zip(*G.edges())
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    if SparseTensor is not None:
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(N, N))
        adj = adj.set_diag()
        deg = adj.sum(dim=1).pow(-0.5)
        adj = adj.mul(deg.view(-1)).mul(deg.view(-1, 1))
        return adj
    else:
        # Scipy fallback (slower, but no extra deps)
        import scipy.sparse as sp
        m = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(N, N))
        m = m + m.T + sp.eye(N)
        deg = np.array(m.sum(1)).flatten()
        deg_inv_sqrt = np.power(deg, -0.5)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.
        D = sp.diags(deg_inv_sqrt)
        m_norm = D @ m @ D
        m_norm = m_norm.tocoo()
        indices = torch.tensor([m_norm.row, m_norm.col], dtype=torch.long)
        values  = torch.tensor(m_norm.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, (N, N))


# -----------------------------------------------------------------------------
# STEP 3 – Model definition
# -----------------------------------------------------------------------------
class GraphConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=False)

    def forward(self, x, adj):
        # adj may be torch_sparse.SparseTensor or torch.sparse tensor
        h = adj.matmul(x) if SparseTensor is not None else torch.sparse.mm(adj, x)
        return torch.relu(self.linear(h))


class GCNLink(nn.Module):
    def __init__(self, in_dim, hidden_dim=16, dropout=0.1):
        super().__init__()
        self.g1 = GraphConv(in_dim, hidden_dim)
        self.g2 = GraphConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(2 * hidden_dim, 1)
        self.drop = nn.Dropout(dropout)

    def node_encode(self, x, adj):
        h1 = self.drop(self.g1(x, adj))
        return self.g2(h1, adj)

    def forward(self, x, adj, edge_index):
        """edge_index – 2×E LongTensor."""
        h = self.node_encode(x, adj)
        src, dst = edge_index
        z = torch.cat([h[src], h[dst]], dim=1)
        return torch.sigmoid(self.lin(z)).squeeze()


# -----------------------------------------------------------------------------
# STEP 4 – Utility: negative edge sampler
# -----------------------------------------------------------------------------

def sample_negative_edges(G, num):
    """Return LongTensor[num,2] of non‑existing undirected edges."""
    N = G.number_of_nodes()
    existing = set(G.edges()) | { (v, u) for u, v in G.edges() }
    neg = []
    while len(neg) < num:
        u = random.randrange(N)
        v = random.randrange(N)
        if u == v or (u, v) in existing or (v, u) in existing:
            continue
        neg.append([u, v])
    return torch.tensor(neg, dtype=torch.long)


# -----------------------------------------------------------------------------
# STEP 5 – Train / evaluate helpers
# -----------------------------------------------------------------------------

def create_batches(edges,batch_size):
    """Create batches from edge list."""
    for i in range(0,len(edges),batch_size):
        yield edges[i:i+batch_size]

def shuffle_edges(edges):
    """Shuffle edges for each epoch."""
    indices=torch.randperm(len(edges))
    return edges[indices]


def link_loss(pos, neg, Q=2):
    eps = 1e-15
    return -(torch.log(pos + eps).mean() + Q * torch.log(1 - neg + eps).mean())


def auc_score(model, X, adj, edges_pos, G):
    with torch.no_grad():
        pos_logits = model(X, adj, edges_pos.t()).cpu().numpy()
        neg_edges = sample_negative_edges(G, len(edges_pos))
        neg_logits = model(X, adj, neg_edges.t()).cpu().numpy()
    y_true = np.concatenate([np.ones_like(pos_logits), np.zeros_like(neg_logits)])
    y_pred = np.concatenate([pos_logits,       neg_logits      ])
    return roc_auc_score(y_true, y_pred)


# -----------------------------------------------------------------------------
# STEP 6 – Main training loop with early stopping
# -----------------------------------------------------------------------------

from torch.utils.data import DataLoader, TensorDataset

def train(model, G, X, adj, train_e, val_e, epochs, lr, patience,batch_size=256):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=4e-5)
    best_val, wait, best_state = 0, 0, None

    train_edges=torch.tensor(train_e, dtype=torch.long)
    dataset=TensorDataset(train_edges)
    dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # for epoch in range(1, epochs + 1):
    #     model.train(); opt.zero_grad()
    #     edge_pos = torch.tensor(train_e, dtype=torch.long)
    #     pos_logits = model(X, adj, edge_pos.t())
    #     edge_neg = sample_negative_edges(G, len(edge_pos))
    #     neg_logits = model(X, adj, edge_neg.t())
    #     loss = link_loss(pos_logits, neg_logits)
    #     loss.backward(); opt.step()

    #     val_auc = auc_score(model, X, adj, torch.tensor(val_e, dtype=torch.long), G)
    #     print(f"Epoch {epoch:3d} | loss {loss.item():.4f} | val AUC {val_auc:.4f}")

    #     if val_auc > best_val:
    #         best_val, wait = val_auc, 0
    #         best_state = copy.deepcopy(model.state_dict())
    #     else:
    #         wait += 1
    #         if wait == patience:
    #             print("Early stopping…")
    #             break
    # model.load_state_dict(best_state)
    # return best_val
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        
        for batch_edges, in dataloader:
            opt.zero_grad()
            
            pos_logits = model(X, adj, batch_edges.t())
            neg_edges = sample_negative_edges(G, len(batch_edges))
            neg_logits = model(X, adj, neg_edges.t())
            
            loss = link_loss(pos_logits, neg_logits)
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        val_auc = auc_score(model, X, adj, torch.tensor(val_e, dtype=torch.long), G)
        print(f"Epoch {epoch:3d} | loss {avg_loss:.4f} | val AUC {val_auc:.4f}")

        if val_auc > best_val:
            best_val, wait = val_auc, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait == patience:
                print("Early stopping…")
                break
    
    model.load_state_dict(best_state)
    return best_val


# -----------------------------------------------------------------------------
# STEP 7 – Entry‑point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-zip", type=Path, default=Path("data.zip"))
    parser.add_argument("--out-dir",  type=Path, default=Path("data"))
    parser.add_argument("--epochs",   type=int,   default=30)
    parser.add_argument("--lr",       type=float, default=1e-2)
    parser.add_argument("--hidden",   type=int,   default=16)
    parser.add_argument("--dropout",  type=float, default=0.1)
    parser.add_argument("--patience", type=int,   default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    set_seed(SEED)

    # Check device availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1 – data
    # cora_dir = ensure_cora(args.data_zip, args.out_dir)
    cora_dir="data/cora"
    G, X = load_cora(cora_dir)
    adj = build_adj(G)

    # Move data to device
    X = X.to(device)
    adj = adj.to(device) if hasattr(adj, 'to') else adj

    # 2 – edge split
    edges = np.array(G.edges())
    train_e, test_e = train_test_split(edges, test_size=0.2, random_state=SEED)
    train_e, val_e  = train_test_split(train_e,  test_size=0.125, random_state=SEED)

    # 3 – model
    model = GCNLink(in_dim=X.size(1), hidden_dim=args.hidden, dropout=args.dropout)
    model = model.to(device)
    print(model)

    # 4 – train
    best_val = train(model, G, X, adj, train_e, val_e,
                     epochs=args.epochs, lr=args.lr, patience=args.patience)
    print(f"Best validation AUC = {best_val:.4f}")

    # 5 – test
    test_auc = auc_score(model, X, adj, torch.tensor(test_e, dtype=torch.long), G)
    print("Test AUC =", test_auc)


if __name__ == "__main__":
    main()
