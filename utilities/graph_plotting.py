import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def adjacency_to_edge_list(adjacency):
    """Convert a symmetric adjacency matrix to one row per undirected edge."""
    values = adjacency.to_numpy(dtype=float)
    rows, columns = np.where(np.triu(values, k=1) > 0.0)
    return pd.DataFrame({
        "Source": adjacency.index.to_numpy()[rows],
        "Target": adjacency.columns.to_numpy()[columns],
        "Weight": values[rows, columns],
    }).sort_values("Weight", ascending=False, ignore_index=True)


def _spectral_positions(nodes, edges):
    node_count = len(nodes)
    if node_count < 3 or edges.empty:
        angles = np.linspace(0.0, 2.0 * np.pi, node_count, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])

    locations = {node: position for position, node in enumerate(nodes)}
    adjacency = np.zeros((node_count, node_count), dtype=float)
    for row in edges.itertuples(index=False):
        left = locations[row.Source]
        right = locations[row.Target]
        adjacency[left, right] = row.Weight
        adjacency[right, left] = row.Weight
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.divide(
        1.0,
        np.sqrt(degree),
        out=np.zeros_like(degree),
        where=degree > 0.0,
    )
    laplacian = np.eye(node_count) - inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    _, eigenvectors = np.linalg.eigh(laplacian)
    if eigenvectors.shape[1] < 3:
        return np.column_stack([eigenvectors[:, -1], np.zeros(node_count)])
    return eigenvectors[:, 1:3]


def plot_weighted_graph(edges, node_values=None, max_nodes=40, max_edges=250,
                        minimum_weight=0.0, title=None, ax=None):
    """Plot the strongest portion of an undirected weighted graph."""
    required = {"Source", "Target", "Weight"}
    if not isinstance(edges, pd.DataFrame) or not required.issubset(edges.columns):
        raise ValueError("edges must contain Source, Target, and Weight columns.")
    max_nodes = int(max_nodes)
    max_edges = int(max_edges)
    if max_nodes < 2 or max_edges < 1:
        raise ValueError("max_nodes must be at least 2 and max_edges must be positive.")

    filtered = edges.loc[edges["Weight"].ge(float(minimum_weight))].copy()
    if filtered.empty:
        raise ValueError("No graph edges satisfy the plotting threshold.")
    degree = pd.concat([
        filtered.groupby("Source")["Weight"].sum(),
        filtered.groupby("Target")["Weight"].sum(),
    ], axis=1).fillna(0.0).sum(axis=1)
    nodes = degree.nlargest(max_nodes).index.tolist()
    filtered = filtered[
        filtered["Source"].isin(nodes) & filtered["Target"].isin(nodes)
    ].nlargest(max_edges, "Weight")
    connected = pd.Index(filtered["Source"]).union(pd.Index(filtered["Target"]))
    nodes = [node for node in nodes if node in connected]
    if len(nodes) < 2:
        raise ValueError("Fewer than two connected nodes remain after filtering.")

    positions = _spectral_positions(nodes, filtered)
    position_map = dict(zip(nodes, positions))
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 9))
    else:
        fig = ax.figure

    maximum_weight = filtered["Weight"].max()
    for row in filtered.itertuples(index=False):
        start = position_map[row.Source]
        end = position_map[row.Target]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="#59788e",
            alpha=0.12 + 0.55 * row.Weight / maximum_weight,
            linewidth=0.4 + 2.2 * row.Weight / maximum_weight,
            zorder=1,
        )

    node_degree = degree.reindex(nodes).fillna(0.0)
    sizes = 90.0 + 510.0 * node_degree.div(node_degree.max()).fillna(0.0)
    if node_values is None:
        colors = node_degree.to_numpy(dtype=float)
        color_label = "Weighted degree"
        cmap = "viridis"
    else:
        values = pd.Series(node_values, dtype=float).reindex(nodes)
        colors = values.fillna(0.0).to_numpy(dtype=float)
        color_label = values.name or "Node value"
        cmap = "coolwarm"
    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        s=sizes.to_numpy(),
        c=colors,
        cmap=cmap,
        edgecolor="#15242d",
        linewidth=0.7,
        zorder=2,
    )
    for node, coordinates in position_map.items():
        ax.text(
            coordinates[0],
            coordinates[1],
            str(node),
            fontsize=8,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            zorder=3,
        )
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.72)
    colorbar.set_label(color_label)
    ax.set_title(title or "Weighted Equity Graph")
    ax.set_axis_off()
    fig.tight_layout()
    return fig, ax
