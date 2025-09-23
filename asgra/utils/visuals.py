
# utils/visuals.py
# ──────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix
import networkx as nx

def plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        outdir: Path,
        normalize: bool = True,
        fname: str = "confusion_matrix.png",
):
    """
    Save a confusion-matrix figure to <outdir>/reports/fname.

    Parameters
    ----------
    y_true, y_pred : 1-D arrays / lists of int label-ids.
    class_names    : list of str, indexed by label-id (len = #classes).
    outdir         : experiment directory (the one you stamp).
    normalize      : if True, rows are shown as percentages.
    fname          : file name for the PNG.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest")     # default colormap
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # cell labels
    thresh = cm.max() * 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]}"
            ax.text(j, i, value,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    rep_dir = Path(outdir) / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(rep_dir / fname, dpi=300)
    plt.close(fig)


def plot_scene_graph(data, attention_scores, dataset, figsize=(6, 6), edge_cmap='autumn', width_scale=5):
    """
    Plot a scene graph with attention-weighted edges.
    
    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph data object with attributes:
        - x: [N, 1+4] tensor, first col token_id, next 4 are bbox [x1, y1, x2, y2]
        - edge_index: [2, E] tensor of source->target node indices
        - edge_attr: relation ids (not used here)
    attention_scores : array-like of shape [E]
        Attention score per edge (float).
    dataset : Places8SceneGraphDataset
        Dataset object providing word2idx and rel2idx. Used to get labels.
    figsize : tuple
        Figure size.
    edge_cmap : str
        Matplotlib colormap for edges.
    width_scale : float
        Scaling factor for edge widths based on attention.
    """
    # Reverse mappings
    idx2tok = {v: k for k, v in dataset.word2idx.items()}
    idx2rel = {v: k for k, v in dataset.rel2idx.items()}

    # Node positions = bbox centers
    boxes = data.x[:, 1:5].cpu().numpy()
    centers = {i: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for i, b in enumerate(boxes)}

    # Build graph
    G = nx.DiGraph()
    N = data.x.shape[0]
    for i in range(N):
        G.add_node(i, label=idx2tok[int(data.x[i, 0])])

    edge_index = data.edge_index.cpu().numpy()
    E = edge_index.shape[1]
    for ei in range(E):
        s, t = edge_index[:, ei]
        weight = attention_scores[ei]
        G.add_edge(int(s), int(t), weight=weight, rel=idx2rel[int(data.edge_attr[ei])])

    # Draw
    plt.figure(figsize=figsize)
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    nx.draw_networkx_nodes(G, centers, node_size=300, node_color='lightblue')
    nx.draw_networkx_labels(G, centers, labels=nx.get_node_attributes(G, 'label'))
    # Edge widths and colors
    widths = [w * width_scale for w in weights]
    nx.draw_networkx_edges(
        G, centers,
        edgelist=edges,
        width=widths,
        edge_color=weights,
        edge_cmap=plt.get_cmap(edge_cmap)
    )
    # Edge labels = relations
    edge_labels = {(u, v): d['rel'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, centers, edge_labels=edge_labels)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

