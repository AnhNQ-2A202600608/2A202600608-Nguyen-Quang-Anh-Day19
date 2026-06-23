"""
graph_builder.py
================
Xây dựng Knowledge Graph từ tập triples đã được trích xuất.
Sử dụng NetworkX DiGraph + Matplotlib để visualize.

Node attributes: {type, mentions, source_docs, description}
Edge attributes: {relation, weight, source_docs}
"""

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Any, Tuple

import networkx as nx
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from rapidfuzz import fuzz

# ─────────────────────────────────────────────────────────────
# Cấu hình
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR   = Path(__file__).parent.parent / "output"
GRAPH_FILE   = OUTPUT_DIR / "knowledge_graph.gpickle"
VIZ_FILE     = OUTPUT_DIR / "knowledge_graph.png"
VIZ_MINI_FILE = OUTPUT_DIR / "knowledge_graph_mini.png"

# Ngưỡng fuzzy matching để gộp entity tương tự
FUZZY_THRESHOLD = 88

# Màu theo loại entity (phân loại heuristic)
ENTITY_COLORS = {
    "COMPANY":   "#4FC3F7",   # xanh nhạt
    "PERSON":    "#FF8A65",   # cam
    "PRODUCT":   "#A5D6A7",   # xanh lá
    "LOCATION":  "#CE93D8",   # tím nhạt
    "METRIC":    "#FFF176",   # vàng
    "CONCEPT":   "#80DEEA",   # cyan
    "DATE":      "#BCAAA4",   # nâu nhạt
    "OTHER":     "#B0BEC5",   # xám
}

# Các từ khóa để phân loại entity heuristic
_COMPANY_KEYWORDS   = ["tesla", "byd", "ford", "gm", "rivian", "volkswagen", "bmw",
                       "mercedes", "hyundai", "kia", "toyota", "audi", "cadillac",
                       "chevrolet", "lucid", "nio", "xpeng", "catl", "lg energy",
                       "panasonic", "openai", "google", "apple", "microsoft", "amazon",
                       "cox automotive", "citi", "volvo", "polestar", "smart", "mg",
                       "great wall", "saic", "baic", "li auto", "berkshire", "vinfast",
                       "lexus", "geely", "gotion", "stanford", "mit", "berkeley"]
_PERSON_KEYWORDS    = ["elon musk", "musk", "sam altman", "altman", "mary barra",
                       "jim farley", "jensen huang", "lisa su", "satya nadella",
                       "sundar pichai", "tim cook", "stephanie valdez", "itay michaeli",
                       "ilaria mazzocco", "anh bui", "peter slowik", "nic lutsey"]
_LOCATION_KEYWORDS  = ["united states", "us", "china", "europe", "california",
                       "michigan", "germany", "thailand", "brazil", "india",
                       "southeast asia", "global south", "hungary", "washington",
                       "new york", "silicon valley", "shanghai"]
_DATE_KEYWORDS      = ["2020", "2021", "2022", "2023", "2024", "2025", "2030",
                       "q1", "q2", "q3", "q4", "january", "march", "april"]


def _classify_entity(name: str) -> str:
    """Phân loại entity dựa trên heuristic keywords."""
    nl = name.lower()
    for kw in _PERSON_KEYWORDS:
        if kw in nl:
            return "PERSON"
    for kw in _COMPANY_KEYWORDS:
        if kw in nl:
            return "COMPANY"
    for kw in _LOCATION_KEYWORDS:
        if kw in nl:
            return "LOCATION"
    for kw in _DATE_KEYWORDS:
        if kw in nl:
            return "DATE"
    if any(x in nl for x in ["%", "$", "billion", "million", "share", "growth", "rate", "price"]):
        return "METRIC"
    if any(x in nl for x in ["vehicle", "battery", "ev", "bev", "phev", "regulation",
                               "policy", "subsidy", "charging", "infrastructure"]):
        return "CONCEPT"
    return "OTHER"


# ─────────────────────────────────────────────────────────────
# Entity normalization & deduplication
# ─────────────────────────────────────────────────────────────
def _normalize_entity(name: str) -> str:
    """Chuẩn hóa tên entity: title-case, bỏ hậu tố thừa."""
    name = name.strip()
    # Bỏ hậu tố công ty
    for suffix in [" Inc.", " Inc", " Corp.", " Corp", " LLC", " Ltd.", " Ltd",
                   " Co.", " Co", " Group", " Holdings", " plc"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    # Title-case cho consistency
    if name.isupper():
        name = name.title()
    return name.strip()


def _build_entity_mapping(all_entities: List[str]) -> Dict[str, str]:
    """
    Fuzzy deduplication: gộp các entity tương tự nhau vào một canonical name.
    Canonical = entity xuất hiện nhiều nhất trong cluster.
    """
    canonical: Dict[str, str] = {}  # original → canonical
    clusters: List[List[str]] = []

    normalized = [_normalize_entity(e) for e in all_entities]
    unique_norm = list(set(normalized))

    # Greedy clustering bằng fuzzy ratio
    assigned = set()
    for i, e1 in enumerate(unique_norm):
        if e1 in assigned:
            continue
        cluster = [e1]
        assigned.add(e1)
        for e2 in unique_norm[i + 1:]:
            if e2 in assigned:
                continue
            if fuzz.ratio(e1.lower(), e2.lower()) >= FUZZY_THRESHOLD:
                cluster.append(e2)
                assigned.add(e2)
        clusters.append(cluster)

    # Canonical = tên ngắn nhất, title-cased
    for cluster in clusters:
        canon = min(cluster, key=len)
        for e in cluster:
            canonical[e] = canon

    return canonical


# ─────────────────────────────────────────────────────────────
# Build graph
# ─────────────────────────────────────────────────────────────
def build_graph(triples: List[Dict]) -> nx.DiGraph:
    """
    Xây dựng NetworkX DiGraph từ tập triples.

    Parameters
    ----------
    triples : List[{subject, relation, object, source_doc}]

    Returns
    -------
    nx.DiGraph với node/edge attributes đầy đủ
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Thu thập tất cả entities ──
    all_entities_raw = [t["subject"] for t in triples] + [t["object"] for t in triples]
    entity_map = _build_entity_mapping(all_entities_raw)

    # ── Normalize triples ──
    norm_triples = []
    for t in triples:
        subj = entity_map.get(_normalize_entity(t["subject"]), _normalize_entity(t["subject"]))
        obj  = entity_map.get(_normalize_entity(t["object"]),  _normalize_entity(t["object"]))
        rel  = t["relation"].upper().replace(" ", "_")
        src  = t.get("source_doc", "unknown")
        norm_triples.append((subj, rel, obj, src))

    # ── Đếm mentions ──
    entity_mentions = Counter()
    for subj, _, obj, _ in norm_triples:
        entity_mentions[subj] += 1
        entity_mentions[obj]  += 1

    entity_sources: Dict[str, set] = defaultdict(set)
    for subj, _, obj, src in norm_triples:
        entity_sources[subj].add(src)
        entity_sources[obj].add(src)

    edge_data: Dict[Tuple, Dict] = defaultdict(lambda: {"weight": 0, "source_docs": set()})
    for subj, rel, obj, src in norm_triples:
        key = (subj, obj, rel)
        edge_data[key]["weight"] += 1
        edge_data[key]["source_docs"].add(src)

    # ── Tạo đồ thị ──
    G = nx.DiGraph()

    # Thêm nodes
    all_nodes = set(entity_mentions.keys())
    for node in all_nodes:
        G.add_node(
            node,
            type=_classify_entity(node),
            mentions=entity_mentions[node],
            source_docs=list(entity_sources[node]),
        )

    # Thêm edges
    for (subj, obj, rel), data in edge_data.items():
        if subj in G and obj in G:
            G.add_edge(
                subj, obj,
                relation=rel,
                weight=data["weight"],
                source_docs=list(data["source_docs"]),
            )

    print(f"\n[GraphBuilder] Knowledge Graph built:")
    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")
    print(f"  Weakly connected: {nx.is_weakly_connected(G)}")

    # Lưu graph
    with open(GRAPH_FILE, "wb") as f:
        pickle.dump(G, f)
    print(f"  Saved to: {GRAPH_FILE}")

    return G


def load_graph() -> nx.DiGraph:
    """Load graph from pickle file."""
    if not GRAPH_FILE.exists():
        raise FileNotFoundError(f"Graph file not found: {GRAPH_FILE}. Run build_graph() first.")
    with open(GRAPH_FILE, "rb") as f:
        G = pickle.load(f)
    print(f"[GraphBuilder] Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────
def visualize_graph(
    G: nx.DiGraph,
    title: str = "EV Industry Knowledge Graph",
    top_n: int = 80,
    save_path: str | Path = VIZ_FILE,
    figsize: tuple = (22, 16),
) -> None:
    """
    Visualize top-N nodes by mentions.
    Tô màu theo entity type, kích thước node theo số mentions.
    """
    # Lấy top-N nodes
    sorted_nodes = sorted(G.nodes(data=True), key=lambda x: x[1].get("mentions", 0), reverse=True)
    top_nodes = [n for n, _ in sorted_nodes[:top_n]]
    subgraph = G.subgraph(top_nodes).copy()

    fig, ax = plt.subplots(figsize=figsize, facecolor="#0D1117")
    ax.set_facecolor("#0D1117")

    # Layout
    pos = nx.spring_layout(subgraph, k=2.5 / np.sqrt(len(subgraph)), seed=42, iterations=80)

    # Node sizes & colors
    node_sizes  = [max(300, subgraph.nodes[n].get("mentions", 1) * 120) for n in subgraph.nodes()]
    node_colors = [ENTITY_COLORS.get(subgraph.nodes[n].get("type", "OTHER"), "#B0BEC5") for n in subgraph.nodes()]

    # Draw edges
    edge_weights = [subgraph[u][v].get("weight", 1) for u, v in subgraph.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_alphas = [0.2 + 0.5 * (w / max_w) for w in edge_weights]

    nx.draw_networkx_edges(
        subgraph, pos, ax=ax,
        edge_color="#58A6FF",
        alpha=0.35,
        width=0.8,
        arrows=True,
        arrowsize=10,
        connectionstyle="arc3,rad=0.1",
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        subgraph, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.92,
        linewidths=0.8,
        edgecolors="#FFFFFF",
    )

    # Labels (chỉ top 40 để dễ đọc)
    top40 = [n for n, _ in sorted_nodes[:40]]
    labels = {n: n for n in subgraph.nodes() if n in top40}
    nx.draw_networkx_labels(
        subgraph, pos, labels, ax=ax,
        font_size=7,
        font_color="white",
        font_weight="bold",
    )

    # Legend
    patches = [mpatches.Patch(color=color, label=etype) for etype, color in ENTITY_COLORS.items()]
    ax.legend(handles=patches, loc="upper left", facecolor="#161B22", edgecolor="#30363D",
              labelcolor="white", fontsize=9, framealpha=0.9)

    ax.set_title(title, color="white", fontsize=16, fontweight="bold", pad=20)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"[GraphBuilder] Graph saved to: {save_path}")


def visualize_subgraph(
    G: nx.DiGraph,
    center_node: str,
    hops: int = 2,
    title: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple = (14, 10),
) -> None:
    """Visualize subgraph xung quanh một node trung tâm trong phạm vi `hops` hop."""
    # Lấy nodes trong phạm vi hops
    ego_nodes = {center_node}
    current = {center_node}
    for _ in range(hops):
        neighbors = set()
        for n in current:
            neighbors.update(G.successors(n))
            neighbors.update(G.predecessors(n))
        ego_nodes.update(neighbors)
        current = neighbors

    subgraph = G.subgraph(ego_nodes).copy()
    if len(subgraph.nodes()) == 0:
        print(f"[GraphBuilder] Subgraph not found for node: {center_node}")
        return

    fig, ax = plt.subplots(figsize=figsize, facecolor="#0D1117")
    ax.set_facecolor("#0D1117")

    pos = nx.spring_layout(subgraph, k=3.0 / np.sqrt(len(subgraph) + 1), seed=42)

    node_sizes  = [800 if n == center_node else 400 for n in subgraph.nodes()]
    node_colors = ["#F97316" if n == center_node
                   else ENTITY_COLORS.get(subgraph.nodes[n].get("type", "OTHER"), "#B0BEC5")
                   for n in subgraph.nodes()]

    nx.draw_networkx_edges(subgraph, pos, ax=ax, edge_color="#58A6FF", alpha=0.5,
                           width=1.0, arrows=True, arrowsize=12,
                           connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, alpha=0.95,
                           linewidths=1.0, edgecolors="#FFFFFF")
    nx.draw_networkx_labels(subgraph, pos, ax=ax, font_size=8,
                            font_color="white", font_weight="bold")

    # Edge labels (relation)
    edge_labels = {(u, v): d.get("relation", "")[:20] for u, v, d in subgraph.edges(data=True)}
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, ax=ax,
                                 font_size=5.5, font_color="#A0C4FF",
                                 bbox=dict(boxstyle="round,pad=0.1", fc="#0D1117", alpha=0.5))

    title = title or f"Subgraph: {center_node} ({hops}-hop)"
    ax.set_title(title, color="white", fontsize=14, fontweight="bold")
    ax.axis("off")

    if save_path is None:
        save_path = OUTPUT_DIR / f"subgraph_{center_node.replace(' ', '_')}.png"

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"[GraphBuilder] Subgraph saved to: {save_path}")


# ─────────────────────────────────────────────────────────────
# Graph statistics
# ─────────────────────────────────────────────────────────────
def get_graph_stats(G: nx.DiGraph) -> Dict[str, Any]:
    """Thống kê đồ thị chi tiết."""
    type_counts = Counter(nx.get_node_attributes(G, "type").values())
    relation_counts = Counter(d["relation"] for _, _, d in G.edges(data=True))
    top_nodes_by_degree = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_components": nx.number_weakly_connected_components(G),
        "entity_types": dict(type_counts),
        "top_relations": relation_counts.most_common(10),
        "top_nodes_by_degree": top_nodes_by_degree,
        "avg_degree": sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1),
    }
