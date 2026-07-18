# akgvp_prior_visualizer.py
# Standalone visualizer and sanity-test runner for AKGVP-style prior_graph.pt.
# It does not import Isaac Sim / Isaac Lab.

from __future__ import annotations

import argparse
import json
import math
import webbrowser
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


# =============================================================================
# EDIT THIS CONFIG FIRST
# =============================================================================

CONFIG: dict[str, Any] = {
    "prior_path": "source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/off_train_modules/prior_graph.pt",
    "output_dir": "source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/off_train_modules/prior_debug",

    # Figures / reports.
    "fig_dpi": 180,
    "top_n_objects": 5,
    "edge_threshold": 0.02,
    "node_size_base": 900,
    "node_size_scale": 2500,

    # AKGVP_VIS_CENTER_FIX:
    # "zone_centers_xy" in prior_graph.pt is an averaged/aligned prior center.
    # "grid_xy" + "grid_to_zone" is a reference grid map used for visualization.
    # If these are mixed, zone circles can appear shifted relative to the colored cells.
    # For visual maps, draw node circles at centroids recomputed from grid labels.
    # Supported: "grid_labels" | "stored_prior".
    "zone_center_source": "grid_labels",

    # Tests.
    # strict=True means schema/shape/numeric failures raise RuntimeError.
    # Semantic-quality warnings are still written to tests_report.json but do not stop the script.
    "run_tests": True,
    "strict": True,

    # Interactive graph.
    "make_interactive_html": True,
    "open_html": False,
}


# =============================================================================
# Generic helpers
# =============================================================================

def load_torch_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"prior_graph.pt not found: {path}")
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected torch file to contain dict, got {type(obj).__name__}")
    return obj


def as_tensor(x: Any, key: str, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
    else:
        t = torch.as_tensor(x)
    if dtype is not None:
        t = t.to(dtype=dtype)
    if not torch.isfinite(t.float()).all():
        raise RuntimeError(f"Tensor {key!r} contains NaN or Inf")
    return t


def get_object_vocab(prior: dict[str, Any]) -> list[str]:
    vocab = prior.get("prior_object_vocab", prior.get("object_vocab", None))
    if vocab is None:
        raise RuntimeError("Missing object vocabulary: expected 'prior_object_vocab' or 'object_vocab'")
    if not isinstance(vocab, list) or not all(isinstance(x, str) for x in vocab):
        raise RuntimeError("Object vocabulary must be list[str]")
    if len(vocab) == 0:
        raise RuntimeError("Object vocabulary is empty")
    return vocab


def get_object_probs(prior: dict[str, Any]) -> torch.Tensor:
    if "node_object_probs" in prior:
        return as_tensor(prior["node_object_probs"], "node_object_probs", dtype=torch.float32)
    if "zone_object_probs" in prior:
        return as_tensor(prior["zone_object_probs"], "zone_object_probs", dtype=torch.float32)
    raise RuntimeError("Missing object probabilities: expected 'node_object_probs' or 'zone_object_probs'")


def get_builder_config(prior: dict[str, Any]) -> dict[str, Any]:
    cfg = prior.get("builder_config", {})
    return cfg if isinstance(cfg, dict) else {}


def get_room_bounds(prior: dict[str, Any], cfg: dict[str, Any]) -> dict[str, float]:
    b = cfg.get("room_bounds", None)
    if isinstance(b, dict):
        return {"x_min": float(b["x_min"]), "x_max": float(b["x_max"]),
                "y_min": float(b["y_min"]), "y_max": float(b["y_max"])}
    grid_xy = as_tensor(prior["grid_xy"], "grid_xy", dtype=torch.float32)
    pad = 0.5
    return {
        "x_min": float(grid_xy[:, 0].min().item() - pad),
        "x_max": float(grid_xy[:, 0].max().item() + pad),
        "y_min": float(grid_xy[:, 1].min().item() - pad),
        "y_max": float(grid_xy[:, 1].max().item() + pad),
    }


def compute_label_centers_from_reference_grid(prior: dict[str, Any]) -> torch.Tensor:
    """Return visualization centers [K, 2] recomputed from grid labels.

    AKGVP_VIS_CENTER_FIX:
    The saved ``zone_centers_xy`` is an averaged prior quantity after aligning
    many sampled scene graphs. The saved ``grid_xy`` / ``grid_to_zone`` pair is
    only a readable reference map from one aligned graph. If we draw reference
    grid cells but use averaged prior centers, node circles can look misplaced.
    For visualization, centroids should therefore be recomputed from exactly the
    grid labels that are being shown.

    Empty zones should not happen after validation. If they do, fall back to the
    stored prior center for that zone instead of crashing inside plotting code.
    """
    k = int(prior["K"])
    grid_xy = as_tensor(prior["grid_xy"], "grid_xy", dtype=torch.float32)
    labels = as_tensor(prior["grid_to_zone"], "grid_to_zone", dtype=torch.long)
    stored_centers = as_tensor(prior["zone_centers_xy"], "zone_centers_xy", dtype=torch.float32)

    if stored_centers.shape != (k, 2):
        raise RuntimeError(f"zone_centers_xy must have shape {(k, 2)}, got {tuple(stored_centers.shape)}")

    centers = stored_centers.clone()
    for z in range(k):
        mask = labels == z
        if bool(mask.any()):
            centers[z] = grid_xy[mask].mean(dim=0)
    return centers


def get_plot_zone_centers(prior: dict[str, Any], cfg: dict[str, Any]) -> torch.Tensor:
    """Return zone centers used for plotting node circles/edges."""
    source = str(cfg.get("zone_center_source", "grid_labels"))
    if source == "grid_labels":
        return compute_label_centers_from_reference_grid(prior)
    if source == "stored_prior":
        return as_tensor(prior["zone_centers_xy"], "zone_centers_xy", dtype=torch.float32)
    raise RuntimeError(
        f"Unsupported zone_center_source={source!r}. "
        "Expected 'grid_labels' or 'stored_prior'."
    )


def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def top_object_strings(object_probs: torch.Tensor, vocab: list[str], top_n: int) -> list[str]:
    lines = []
    k = int(object_probs.shape[0])
    n = min(top_n, len(vocab))
    for zone in range(k):
        vals, idx = torch.topk(object_probs[zone], k=n)
        parts = [f"{vocab[int(i)]}:{float(v):.3f}" for v, i in zip(vals, idx)]
        lines.append(f"zone_{zone}: " + ", ".join(parts))
    return lines


def edge_list_from_adjacency(adjacency: torch.Tensor, threshold: float) -> list[tuple[int, int, float]]:
    k = int(adjacency.shape[0])
    edges: list[tuple[int, int, float]] = []
    for i in range(k):
        for j in range(i + 1, k):
            w = max(float(adjacency[i, j].item()), float(adjacency[j, i].item()))
            if w > threshold:
                edges.append((i, j, w))
    return edges


# =============================================================================
# Tests
# =============================================================================

def add_test(report: list[dict[str, Any]], name: str, status: str, details: str = "", fatal: bool = False) -> None:
    report.append({"name": name, "status": status, "fatal": bool(fatal), "details": details})


def validate_prior(prior: dict[str, Any], *, strict: bool) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    fatal_errors: list[str] = []

    def fail(name: str, details: str) -> None:
        add_test(report, name, "FAIL", details, fatal=True)
        fatal_errors.append(f"{name}: {details}")

    def ok(name: str, details: str = "") -> None:
        add_test(report, name, "PASS", details, fatal=False)

    def warn(name: str, details: str) -> None:
        add_test(report, name, "WARN", details, fatal=False)

    required = ["K", "node_features", "adjacency", "zone_centers_xy", "grid_xy", "grid_to_zone", "builder_config"]
    for key in required:
        if key not in prior:
            fail(f"schema_has_{key}", f"Missing key {key!r}")
        else:
            ok(f"schema_has_{key}")

    try:
        vocab = get_object_vocab(prior)
        ok("schema_has_object_vocab", f"V={len(vocab)}")
    except Exception as e:
        fail("schema_has_object_vocab", str(e))
        vocab = []

    try:
        object_probs = get_object_probs(prior)
        ok("schema_has_object_probs", f"shape={tuple(object_probs.shape)}")
    except Exception as e:
        fail("schema_has_object_probs", str(e))
        object_probs = torch.empty(0)

    if fatal_errors:
        if strict:
            raise RuntimeError("Invalid prior schema:\n" + "\n".join(fatal_errors))
        return report

    try:
        k = int(prior["K"])
        node_features = as_tensor(prior["node_features"], "node_features", dtype=torch.float32)
        adjacency = as_tensor(prior["adjacency"], "adjacency", dtype=torch.float32)
        centers = as_tensor(prior["zone_centers_xy"], "zone_centers_xy", dtype=torch.float32)
        grid_xy = as_tensor(prior["grid_xy"], "grid_xy", dtype=torch.float32)
        grid_to_zone = as_tensor(prior["grid_to_zone"], "grid_to_zone", dtype=torch.long)
    except Exception as e:
        fail("tensor_loading", str(e))
        if strict:
            raise RuntimeError(str(e)) from e
        return report

    if k <= 0:
        fail("K_positive", f"K={k}")
    else:
        ok("K_positive", f"K={k}")

    if node_features.ndim != 2 or node_features.shape[0] != k:
        fail("node_features_shape", f"expected [K,D], got {tuple(node_features.shape)}")
    else:
        ok("node_features_shape", f"shape={tuple(node_features.shape)}")

    if adjacency.shape != (k, k):
        fail("adjacency_shape", f"expected {(k, k)}, got {tuple(adjacency.shape)}")
    else:
        ok("adjacency_shape", f"shape={tuple(adjacency.shape)}")

    if centers.shape != (k, 2):
        fail("zone_centers_shape", f"expected {(k, 2)}, got {tuple(centers.shape)}")
    else:
        ok("zone_centers_shape", f"shape={tuple(centers.shape)}")

    if grid_xy.ndim != 2 or grid_xy.shape[1] != 2:
        fail("grid_xy_shape", f"expected [G,2], got {tuple(grid_xy.shape)}")
    else:
        ok("grid_xy_shape", f"shape={tuple(grid_xy.shape)}")

    if grid_to_zone.ndim != 1 or grid_to_zone.shape[0] != grid_xy.shape[0]:
        fail("grid_to_zone_shape", f"grid_to_zone={tuple(grid_to_zone.shape)}, grid_xy={tuple(grid_xy.shape)}")
    else:
        ok("grid_to_zone_shape", f"shape={tuple(grid_to_zone.shape)}")

    if object_probs.ndim != 2 or object_probs.shape != (k, len(vocab)):
        fail("object_probs_shape", f"expected {(k, len(vocab))}, got {tuple(object_probs.shape)}")
    else:
        ok("object_probs_shape", f"shape={tuple(object_probs.shape)}")

    if fatal_errors:
        if strict:
            raise RuntimeError("Invalid prior shapes:\n" + "\n".join(fatal_errors))
        return report

    if not ((adjacency >= -1e-6) & (adjacency <= 1.0 + 1e-6)).all():
        fail("adjacency_range", "adjacency must be in [0, 1]")
    else:
        ok("adjacency_range", f"min={float(adjacency.min()):.4f}, max={float(adjacency.max()):.4f}")

    diag_abs = float(torch.diag(adjacency).abs().max().item())
    if diag_abs > 1e-5:
        fail("adjacency_diagonal_zero", f"max_abs_diag={diag_abs:.6f}")
    else:
        ok("adjacency_diagonal_zero")

    asym = float((adjacency - adjacency.T).abs().max().item())
    if asym > 1e-4:
        warn("adjacency_symmetry", f"max_asym={asym:.6f}; acceptable for row-normalized transition probabilities")
    else:
        ok("adjacency_symmetry", f"max_asym={asym:.6f}")

    if not ((grid_to_zone >= 0) & (grid_to_zone < k)).all():
        fail("grid_to_zone_range", f"labels must be in [0,{k - 1}]")
    else:
        ok("grid_to_zone_range")

    counts = torch.bincount(grid_to_zone, minlength=k)
    empty = [int(i) for i in torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()]
    if empty:
        fail("each_zone_has_grid_cells", f"empty zones={empty}")
    else:
        ok("each_zone_has_grid_cells", f"counts={counts.tolist()}")

    offdiag = adjacency.clone()
    offdiag.fill_diagonal_(0.0)
    if float(offdiag.sum().item()) <= 0.0:
        warn("graph_not_empty", "No non-diagonal adjacency edges")
    else:
        ok("graph_not_empty", f"edge_mass={float(offdiag.sum().item()):.4f}")

    complete_possible = k * (k - 1)
    positive_edges = int((offdiag > 1e-6).sum().item())
    if positive_edges == complete_possible:
        warn("graph_not_complete", "Every directed off-diagonal edge is positive; adjacency may be too dense")
    else:
        ok("graph_not_complete", f"positive_directed_edges={positive_edges}/{complete_possible}")

    if object_probs.numel() > 0:
        if not ((object_probs >= -1e-6) & (object_probs <= 1.0 + 1e-6)).all():
            fail("object_probs_range", "object probabilities must be in [0, 1]")
        else:
            ok("object_probs_range", f"min={float(object_probs.min()):.4f}, max={float(object_probs.max()):.4f}")

        max_per_zone = object_probs.max(dim=1).values
        silent = [int(i) for i in torch.nonzero(max_per_zone <= 1e-6, as_tuple=False).flatten().tolist()]
        if silent:
            warn("semantic_signal_per_zone", f"zones with no visible object signal={silent}")
        else:
            ok("semantic_signal_per_zone", f"max_per_zone={[round(float(x), 4) for x in max_per_zone]}")

    optional_std = ["node_features_std", "zone_object_probs_std", "node_object_probs_std", "zone_centers_xy_std", "adjacency_std"]
    for key in optional_std:
        if key in prior:
            try:
                t = as_tensor(prior[key], key, dtype=torch.float32)
                ok(f"optional_{key}", f"shape={tuple(t.shape)}")
            except Exception as e:
                fail(f"optional_{key}", str(e))
        else:
            warn(f"optional_{key}", "missing; visualization will skip this stability artifact")

    if fatal_errors and strict:
        raise RuntimeError("Invalid prior numeric content:\n" + "\n".join(fatal_errors))
    return report


# =============================================================================
# Static visualizations
# =============================================================================

def plot_zone_assignment_map(prior: dict[str, Any], out_path: Path, cfg: dict[str, Any], dpi: int) -> None:
    import matplotlib.pyplot as plt

    grid_xy = as_tensor(prior["grid_xy"], "grid_xy", dtype=torch.float32)
    labels = as_tensor(prior["grid_to_zone"], "grid_to_zone", dtype=torch.long)
    centers = get_plot_zone_centers(prior, cfg)

    plt.figure(figsize=(8, 8), dpi=dpi)
    plt.scatter(grid_xy[:, 0], grid_xy[:, 1], c=labels, s=18, alpha=0.85)
    plt.scatter(centers[:, 0], centers[:, 1], marker="x", s=160, linewidths=3)
    for i, xy in enumerate(centers):
        plt.text(float(xy[0]), float(xy[1]), f"  z{i}", fontsize=10, weight="bold")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Zone assignment map: grid cell -> prior zone")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_topdown_graph(prior: dict[str, Any], out_path: Path, cfg: dict[str, Any], dpi: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    grid_xy = as_tensor(prior["grid_xy"], "grid_xy", dtype=torch.float32)
    labels = as_tensor(prior["grid_to_zone"], "grid_to_zone", dtype=torch.long)
    centers = get_plot_zone_centers(prior, cfg)
    adjacency = as_tensor(prior["adjacency"], "adjacency", dtype=torch.float32)
    bounds = get_room_bounds(prior, get_builder_config(prior))
    edges = edge_list_from_adjacency(adjacency, float(cfg["edge_threshold"]))

    plt.figure(figsize=(8, 8), dpi=dpi)
    ax = plt.gca()
    ax.add_patch(Rectangle(
        (bounds["x_min"], bounds["y_min"]),
        bounds["x_max"] - bounds["x_min"],
        bounds["y_max"] - bounds["y_min"],
        fill=False,
        linewidth=2,
    ))
    plt.scatter(grid_xy[:, 0], grid_xy[:, 1], c=labels, s=14, alpha=0.35)

    for i, j, w in edges:
        x = [float(centers[i, 0]), float(centers[j, 0])]
        y = [float(centers[i, 1]), float(centers[j, 1])]
        plt.plot(x, y, linewidth=1.0 + 6.0 * w, alpha=0.75)
        mx = 0.5 * (x[0] + x[1])
        my = 0.5 * (y[0] + y[1])
        plt.text(mx, my, f"{w:.2f}", fontsize=8)

    plt.scatter(centers[:, 0], centers[:, 1], s=260, edgecolors="black", linewidths=1.2, zorder=5)
    for i, xy in enumerate(centers):
        plt.text(float(xy[0]), float(xy[1]), f"z{i}", fontsize=11, weight="bold", ha="center", va="center", zorder=6)

    ax.set_xlim(bounds["x_min"] - 0.25, bounds["x_max"] + 0.25)
    ax.set_ylim(bounds["y_min"] - 0.25, bounds["y_max"] + 0.25)
    ax.set_aspect("equal", adjustable="box")
    plt.title("Prior graph top-down: zones and adjacency")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_matrix(matrix: torch.Tensor, out_path: Path, title: str, xlabels: list[str] | None, ylabels: list[str] | None, dpi: int) -> None:
    import matplotlib.pyplot as plt

    matrix = matrix.detach().cpu().float()
    h = max(5.0, min(12.0, 0.35 * matrix.shape[0] + 4.0))
    w = max(6.0, min(18.0, 0.45 * matrix.shape[1] + 4.0))
    plt.figure(figsize=(w, h), dpi=dpi)
    plt.imshow(matrix.numpy(), aspect="auto")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)

    if xlabels is not None:
        plt.xticks(range(len(xlabels)), xlabels, rotation=70, ha="right", fontsize=8)
    if ylabels is not None:
        plt.yticks(range(len(ylabels)), ylabels, fontsize=9)

    if matrix.numel() <= 100:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(j, i, f"{float(matrix[i, j]):.2f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_node_similarity(prior: dict[str, Any], out_path: Path, dpi: int) -> None:
    node_features = as_tensor(prior["node_features"], "node_features", dtype=torch.float32)
    sim = F.normalize(node_features, dim=-1) @ F.normalize(node_features, dim=-1).T
    labels = [f"z{i}" for i in range(node_features.shape[0])]
    plot_matrix(sim, out_path, "Node feature cosine similarity", labels, labels, dpi)


def plot_static_artifacts(prior: dict[str, Any], out_dir: Path, cfg: dict[str, Any]) -> list[Path]:
    dpi = int(cfg["fig_dpi"])
    vocab = get_object_vocab(prior)
    object_probs = get_object_probs(prior)
    k = int(prior["K"])
    zone_labels = [f"z{i}" for i in range(k)]

    paths: list[Path] = []

    p = out_dir / "zone_assignment_map.png"
    plot_zone_assignment_map(prior, p, cfg, dpi)
    paths.append(p)

    p = out_dir / "prior_graph_topdown.png"
    plot_topdown_graph(prior, p, cfg, dpi)
    paths.append(p)

    p = out_dir / "adjacency_matrix.png"
    plot_matrix(as_tensor(prior["adjacency"], "adjacency", dtype=torch.float32), p, "Adjacency / transition probability", zone_labels, zone_labels, dpi)
    paths.append(p)

    p = out_dir / "node_object_distribution.png"
    plot_matrix(object_probs, p, "Mean object visibility probability per zone", vocab, zone_labels, dpi)
    paths.append(p)

    p = out_dir / "node_similarity_matrix.png"
    plot_node_similarity(prior, p, dpi)
    paths.append(p)

    if "adjacency_std" in prior:
        p = out_dir / "adjacency_std_matrix.png"
        plot_matrix(as_tensor(prior["adjacency_std"], "adjacency_std", dtype=torch.float32), p, "Adjacency std across aligned scenes", zone_labels, zone_labels, dpi)
        paths.append(p)

    std_key = "node_object_probs_std" if "node_object_probs_std" in prior else "zone_object_probs_std"
    if std_key in prior:
        p = out_dir / "node_object_distribution_std.png"
        plot_matrix(as_tensor(prior[std_key], std_key, dtype=torch.float32), p, "Object probability std across aligned scenes", vocab, zone_labels, dpi)
        paths.append(p)

    return paths


# =============================================================================
# Interactive HTML graph
# =============================================================================

def make_interactive_html(prior: dict[str, Any], out_path: Path, cfg: dict[str, Any]) -> Path:
    import plotly.graph_objects as go

    vocab = get_object_vocab(prior)
    object_probs = get_object_probs(prior)
    centers = get_plot_zone_centers(prior, cfg)
    adjacency = as_tensor(prior["adjacency"], "adjacency", dtype=torch.float32)
    grid_xy = as_tensor(prior["grid_xy"], "grid_xy", dtype=torch.float32)
    grid_to_zone = as_tensor(prior["grid_to_zone"], "grid_to_zone", dtype=torch.long)
    bounds = get_room_bounds(prior, get_builder_config(prior))
    edges = edge_list_from_adjacency(adjacency, float(cfg["edge_threshold"]))
    top_lines = top_object_strings(object_probs, vocab, int(cfg["top_n_objects"]))

    fig = go.Figure()

    # Grid background, one trace per zone for readable legend.
    for z in range(int(prior["K"])):
        mask = grid_to_zone == z
        if bool(mask.any()):
            pts = grid_xy[mask]
            fig.add_trace(go.Scatter(
                x=pts[:, 0].numpy(),
                y=pts[:, 1].numpy(),
                mode="markers",
                marker=dict(size=5, opacity=0.25),
                name=f"grid z{z}",
                hovertemplate="zone=%{customdata}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
                customdata=[z] * int(pts.shape[0]),
            ))

    # Edges, one trace per edge because Plotly line width is per trace.
    for i, j, w in edges:
        fig.add_trace(go.Scatter(
            x=[float(centers[i, 0]), float(centers[j, 0])],
            y=[float(centers[i, 1]), float(centers[j, 1])],
            mode="lines",
            line=dict(width=max(1.0, 8.0 * w)),
            opacity=0.65,
            name=f"edge z{i}-z{j}: {w:.3f}",
            hovertemplate=f"z{i} ↔ z{j}<br>weight={w:.4f}<extra></extra>",
        ))

    node_strength = adjacency.sum(dim=1) + adjacency.sum(dim=0)
    node_sizes = float(cfg["node_size_base"]) ** 0.5 + float(cfg["node_size_scale"]) ** 0.5 * node_strength / node_strength.clamp_min(1e-6).max()
    hover = []
    for z in range(int(prior["K"])):
        hover.append(
            f"<b>zone {z}</b>"
            f"<br>x={float(centers[z, 0]):.3f}, y={float(centers[z, 1]):.3f}"
            f"<br>strength={float(node_strength[z]):.3f}"
            f"<br>{top_lines[z]}"
        )

    fig.add_trace(go.Scatter(
        x=centers[:, 0].numpy(),
        y=centers[:, 1].numpy(),
        mode="markers+text",
        text=[f"z{i}" for i in range(int(prior["K"]))],
        textposition="middle center",
        marker=dict(size=node_sizes.numpy(), line=dict(width=2, color="black")),
        name="zone nodes",
        hovertext=hover,
        hoverinfo="text",
    ))

    # Room rectangle.
    fig.add_shape(
        type="rect",
        x0=bounds["x_min"], y0=bounds["y_min"],
        x1=bounds["x_max"], y1=bounds["y_max"],
        line=dict(width=2),
        fillcolor="rgba(0,0,0,0)",
    )

    fig.update_layout(
        title="Interactive AKGVP-style prior graph",
        xaxis_title="x",
        yaxis_title="y",
        yaxis_scaleanchor="x",
        template="plotly_white",
        width=1000,
        height=850,
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    return out_path


# =============================================================================
# Text / JSON reports
# =============================================================================

def write_summary(prior: dict[str, Any], tests: list[dict[str, Any]], out_path: Path, cfg: dict[str, Any]) -> None:
    vocab = get_object_vocab(prior)
    object_probs = get_object_probs(prior)
    adjacency = as_tensor(prior["adjacency"], "adjacency", dtype=torch.float32)
    centers = get_plot_zone_centers(prior, cfg)
    stored_centers = as_tensor(prior["zone_centers_xy"], "zone_centers_xy", dtype=torch.float32)
    grid_xy = as_tensor(prior["grid_xy"], "grid_xy", dtype=torch.float32)
    grid_to_zone = as_tensor(prior["grid_to_zone"], "grid_to_zone", dtype=torch.long)
    builder_config = get_builder_config(prior)

    pass_count = sum(t["status"] == "PASS" for t in tests)
    fail_count = sum(t["status"] == "FAIL" for t in tests)
    warn_count = sum(t["status"] == "WARN" for t in tests)

    lines = []
    lines.append("AKGVP prior graph summary")
    lines.append("==========================")
    lines.append(f"scene_family: {prior.get('scene_family', 'unknown')}")
    lines.append(f"K: {int(prior['K'])}")
    lines.append(f"num_scenes: {prior.get('num_scenes', 'unknown')}")
    lines.append(f"object_vocab_size: {len(vocab)}")
    lines.append(f"node_features_shape: {tuple(as_tensor(prior['node_features'], 'node_features').shape)}")
    lines.append(f"object_probs_shape: {tuple(object_probs.shape)}")
    lines.append(f"adjacency_shape: {tuple(adjacency.shape)}")
    lines.append(f"grid_cells_reference: {int(grid_xy.shape[0])}")
    lines.append(f"builder_grid_step: {builder_config.get('grid_step', 'unknown')}")
    lines.append(f"builder_num_yaws: {builder_config.get('num_yaws', 'unknown')}")
    lines.append(f"builder_fov_deg: {builder_config.get('fov_deg', 'unknown')}")
    lines.append(f"builder_max_visible_distance: {builder_config.get('max_visible_distance', 'unknown')}")
    lines.append("")

    lines.append("Zone centers used for visualization")
    lines.append("-----------------------------------")
    lines.append(f"center_source: {cfg.get('zone_center_source', 'grid_labels')}")
    for z, xy in enumerate(centers):
        count = int((grid_to_zone == z).sum().item())
        lines.append(f"zone_{z}: x={float(xy[0]): .3f}, y={float(xy[1]): .3f}, reference_cells={count}, stored_prior=({float(stored_centers[z, 0]): .3f}, {float(stored_centers[z, 1]): .3f})")
    lines.append("")

    lines.append("Top objects per zone")
    lines.append("--------------------")
    lines.extend(top_object_strings(object_probs, vocab, int(cfg["top_n_objects"])))
    lines.append("")

    lines.append("Adjacency")
    lines.append("---------")
    lines.append(f"min={float(adjacency.min()):.4f}, max={float(adjacency.max()):.4f}, sum={float(adjacency.sum()):.4f}")
    lines.append("")

    lines.append("Tests")
    lines.append("-----")
    lines.append(f"PASS={pass_count}, WARN={warn_count}, FAIL={fail_count}")
    for t in tests:
        lines.append(f"[{t['status']}] {t['name']}: {t.get('details', '')}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_tests_report(tests: list[dict[str, Any]], out_path: Path) -> None:
    payload = {
        "num_pass": sum(t["status"] == "PASS" for t in tests),
        "num_warn": sum(t["status"] == "WARN" for t in tests),
        "num_fail": sum(t["status"] == "FAIL" for t in tests),
        "tests": tests,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# =============================================================================
# Pipeline
# =============================================================================

def visualize_prior(cfg: dict[str, Any]) -> dict[str, Any]:
    prior_path = Path(cfg["prior_path"])
    out_dir = ensure_output_dir(cfg["output_dir"])
    prior = load_torch_file(prior_path)

    tests: list[dict[str, Any]] = []
    if bool(cfg["run_tests"]):
        tests = validate_prior(prior, strict=bool(cfg["strict"]))
        write_tests_report(tests, out_dir / "tests_report.json")

    static_paths = plot_static_artifacts(prior, out_dir, cfg)

    html_path = None
    if bool(cfg["make_interactive_html"]):
        html_path = make_interactive_html(prior, out_dir / "interactive_prior_graph.html", cfg)
        if bool(cfg["open_html"]):
            webbrowser.open(html_path.resolve().as_uri())

    write_summary(prior, tests, out_dir / "summary.txt", cfg)

    return {
        "prior_path": str(prior_path),
        "output_dir": str(out_dir),
        "static_paths": [str(p) for p in static_paths],
        "html_path": str(html_path) if html_path is not None else None,
        "tests_report": str(out_dir / "tests_report.json") if tests else None,
        "summary": str(out_dir / "summary.txt"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize and test AKGVP-style prior_graph.pt")
    parser.add_argument("--prior", type=str, default=CONFIG["prior_path"], help="Path to prior_graph.pt")
    parser.add_argument("--out", type=str, default=CONFIG["output_dir"], help="Output directory for plots/reports")
    parser.add_argument("--top-n", type=int, default=CONFIG["top_n_objects"], help="Top objects per zone")
    parser.add_argument("--edge-threshold", type=float, default=CONFIG["edge_threshold"], help="Min adjacency weight shown as graph edge")
    parser.add_argument("--no-tests", action="store_true", help="Skip validation tests")
    parser.add_argument("--non-strict", action="store_true", help="Do not raise on schema/shape/numeric test failures")
    parser.add_argument("--no-html", action="store_true", help="Do not save interactive HTML graph")
    parser.add_argument("--open-html", action="store_true", help="Open interactive HTML graph in the default browser")
    parser.add_argument(
        "--center-source",
        type=str,
        choices=["grid_labels", "stored_prior"],
        default=CONFIG["zone_center_source"],
        help="Where to draw zone circles: grid-label centroids or stored averaged prior centers",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = dict(CONFIG)
    cfg.update({
        "prior_path": args.prior,
        "output_dir": args.out,
        "top_n_objects": args.top_n,
        "edge_threshold": args.edge_threshold,
        "run_tests": not args.no_tests,
        "strict": not args.non_strict,
        "make_interactive_html": not args.no_html,
        "open_html": args.open_html,
        "zone_center_source": args.center_source,
    })
    result = visualize_prior(cfg)
    print("Saved AKGVP prior debug artifacts:")
    print(f"  output_dir: {result['output_dir']}")
    print(f"  summary: {result['summary']}")
    print(f"  tests_report: {result['tests_report']}")
    print(f"  interactive_html: {result['html_path']}")
    print("  static plots:")
    for p in result["static_paths"]:
        print(f"    - {p}")


if __name__ == "__main__":
    main()
