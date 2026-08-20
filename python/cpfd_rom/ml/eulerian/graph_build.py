# cpfd_rom/ml/eulerian/graph_build.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from cpfd_rom.util.file_parsing import (
    get_columns_from_json_cached,
    get_simulation_time_from_json_fast,
)
from cpfd_rom.util.logging_config import detail, progress_enabled


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Source discovery + indexing (RAW.CELL only)
# --------------------------------------------------------------------------------------


def _source_has_raw_cell(src_dir: Path) -> bool:
    return any(src_dir.glob("Raw.cell.*.npy"))


def _pair_npy_with_json(npy_path: Union[str, Path]) -> Path:
    npy_path = Path(npy_path)
    json_path = npy_path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(
            f"Missing JSON header for {npy_path}: expected {json_path}"
        )
    return json_path


def _list_raw_cell_npy(src_dir: Path) -> list[Path]:
    return sorted(src_dir.glob("Raw.cell.*.npy"))


def _build_cell_time_index(src_dir: Path) -> list[tuple[float, Path]]:
    npy_files = _list_raw_cell_npy(src_dir)
    if not npy_files:
        raise FileNotFoundError(
            f"No Raw.cell.*.npy files found in {src_dir}"
        )

    entries: list[tuple[float, Path]] = []
    for npy_path in npy_files:
        json_path = _pair_npy_with_json(npy_path)
        simulation_time = get_simulation_time_from_json_fast(str(json_path))
        entries.append((simulation_time, npy_path))

    entries.sort(key=lambda item: item[0])
    return entries


# --------------------------------------------------------------------------------------
# Snapshot loading
# --------------------------------------------------------------------------------------


def _resolve_colnames_from_json(json_path: Path) -> list[str]:
    return list(get_columns_from_json_cached(str(json_path)))


def _load_raw_cell_df(npy_path: Path, json_path: Path) -> pd.DataFrame:
    array = np.load(str(npy_path), allow_pickle=False)

    if (
        getattr(array, "dtype", None) is not None
        and array.dtype.names is not None
    ):
        return pd.DataFrame(
            {name: array[name] for name in array.dtype.names}
        )

    colnames = _resolve_colnames_from_json(json_path)
    if array.ndim != 2:
        raise ValueError(
            f"Expected 2D array in {npy_path}, got shape {array.shape}"
        )
    if array.shape[1] != len(colnames):
        raise ValueError(
            f"Column count mismatch in {npy_path}: expected "
            f"{len(colnames)} (from JSON), got {array.shape[1]}"
        )
    return pd.DataFrame(array, columns=colnames)


def _require_cols(
    df: pd.DataFrame,
    cols: list[str],
    context: str = "",
) -> None:
    missing = [column for column in cols if column not in df.columns]
    if missing:
        message = f"Missing required columns {missing}"
        if context:
            message += f" ({context})"
        message += (
            f". Available columns include: {list(df.columns)[:40]} ..."
        )
        raise ValueError(message)


# --------------------------------------------------------------------------------------
# Graph builder (stencil on provided ijk)
# --------------------------------------------------------------------------------------


def _df_to_nodes_df(df_ref: pd.DataFrame) -> pd.DataFrame:
    """
    NODE order = sorted by Cell ID (stable across snapshots).

    Must contain: Cell ID, i, j, k, Cell center x/y/z.
    Produces nodes.parquet with: node_id, Cell ID, i, j, k, x, y, z.
    """
    _require_cols(
        df_ref,
        [
            "Cell ID",
            "i",
            "j",
            "k",
            "Cell center x",
            "Cell center y",
            "Cell center z",
        ],
        context="build nodes",
    )

    # Force independent frame + deterministic order.
    nodes = (
        df_ref.sort_values("Cell ID")
        .reset_index(drop=True)
        .copy(deep=True)
    )

    nodes.insert(0, "node_id", np.arange(len(nodes), dtype=np.int64))

    # CoW-safe dtype coercions.
    nodes.loc[:, "Cell ID"] = nodes["Cell ID"].to_numpy(dtype=np.int64)
    nodes.loc[:, "i"] = nodes["i"].to_numpy(dtype=np.int64)
    nodes.loc[:, "j"] = nodes["j"].to_numpy(dtype=np.int64)
    nodes.loc[:, "k"] = nodes["k"].to_numpy(dtype=np.int64)

    # x/y/z aliases.
    nodes.loc[:, "x"] = nodes["Cell center x"].to_numpy(dtype=float)
    nodes.loc[:, "y"] = nodes["Cell center y"].to_numpy(dtype=float)
    nodes.loc[:, "z"] = nodes["Cell center z"].to_numpy(dtype=float)

    return nodes.loc[
        :, ["node_id", "Cell ID", "i", "j", "k", "x", "y", "z"]
    ].copy(deep=True)


def _nodes_to_edge_index(
    nodes: pd.DataFrame,
    *,
    neighbor_set: str = "n6",
    bidirectional: bool = True,
) -> np.ndarray:
    required = {"node_id", "i", "j", "k"}
    if not required.issubset(nodes.columns):
        raise ValueError(f"nodes must have columns {required}")

    triplets = list(
        zip(
            nodes["i"].to_numpy(),
            nodes["j"].to_numpy(),
            nodes["k"].to_numpy(),
        )
    )
    node_ids = nodes["node_id"].to_numpy()
    lookup = {
        ijk: node_id for ijk, node_id in zip(triplets, node_ids)
    }

    offsets = []
    for delta_i in (-1, 0, 1):
        for delta_j in (-1, 0, 1):
            for delta_k in (-1, 0, 1):
                if delta_i == 0 and delta_j == 0 and delta_k == 0:
                    continue
                manhattan = abs(delta_i) + abs(delta_j) + abs(delta_k)
                if neighbor_set == "n6":
                    if manhattan == 1:
                        offsets.append((delta_i, delta_j, delta_k))
                elif neighbor_set == "n18":
                    if 1 <= manhattan <= 2:
                        offsets.append((delta_i, delta_j, delta_k))
                elif neighbor_set == "n26":
                    if 1 <= manhattan <= 3:
                        offsets.append((delta_i, delta_j, delta_k))
                else:
                    raise ValueError(
                        f"Unknown neighbor_set={neighbor_set}"
                    )

    edges = set()
    for (i, j, k), source in zip(triplets, node_ids):
        for delta_i, delta_j, delta_k in offsets:
            neighbor = (i + delta_i, j + delta_j, k + delta_k)
            destination = lookup.get(neighbor)
            if destination is None:
                continue
            if bidirectional:
                edge = (
                    (source, destination)
                    if source <= destination
                    else (destination, source)
                )
                edges.add(edge)
            else:
                edges.add((source, destination))

    if bidirectional:
        directed_edges = []
        for source, destination in edges:
            if source == destination:
                continue
            directed_edges.append((source, destination))
            directed_edges.append((destination, source))
        edge_index = np.asarray(directed_edges, dtype=np.int64).T
    else:
        edge_index = (
            np.asarray(list(edges), dtype=np.int64).T
            if edges
            else np.empty((2, 0), dtype=np.int64)
        )

    if edge_index.size:
        node_count = len(nodes)
        edge_count = edge_index.shape[1]
        detail(
            logger,
            (
                "Graph edge statistics: nodes=%d, directed_edges=%d, "
                "average_out_degree=%.2f, bidirectional=%s, "
                "neighbor_set=%s"
            ),
            node_count,
            edge_count,
            edge_count / node_count,
            bidirectional,
            neighbor_set,
        )
    else:
        logger.warning("No graph edges were generated")

    return edge_index


def _select_reference_frames(
    src_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load earliest Raw.cell snapshot and return two data frames.

    df_ref is the compact frame used for nodes and edges. df_file preserves
    FILE/FLATTEN order. Both include Cell ID, Cell center x/y/z, and i/j/k
    from the solver output.
    """
    entries = _build_cell_time_index(src_dir)
    reference_time, npy_path = entries[0]
    json_path = _pair_npy_with_json(npy_path)

    detail(
        logger,
        "Using reference snapshot %s at simulation time %.6g",
        npy_path.name,
        reference_time,
    )

    reference_data = _load_raw_cell_df(npy_path, json_path).copy(deep=True)

    columns = [
        "Cell ID",
        "Cell center x",
        "Cell center y",
        "Cell center z",
        "i",
        "j",
        "k",
    ]
    _require_cols(reference_data, columns, context="reference snapshot")

    reference_data.loc[:, "Cell ID"] = reference_data[
        "Cell ID"
    ].to_numpy(dtype=np.int64)

    for column in ("Cell center x", "Cell center y", "Cell center z"):
        reference_data.loc[:, column] = reference_data[column].to_numpy(
            dtype=float
        )

    for column in ("i", "j", "k"):
        reference_data.loc[:, column] = reference_data[column].to_numpy(
            dtype=np.int64
        )

    df_file = reference_data.loc[:, columns].copy()
    df_ref = reference_data.loc[:, columns].copy()
    return df_ref, df_file


def _build_nodes_and_edges(
    df_ref: pd.DataFrame,
    df_file: pd.DataFrame,
    out_dir: Path,
    *,
    edge_bidir: bool = True,
    neighbor_set: str = "n6",
) -> None:
    logger.info("Building Eulerian graph artifacts")
    detail(logger, "Graph artifact directory: %s", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = _df_to_nodes_df(df_ref)

    nodes_path = out_dir / "nodes.parquet"
    nodes_path.unlink(missing_ok=True)
    nodes.to_parquet(nodes_path, index=False)
    if not nodes_path.exists():
        raise FileNotFoundError(f"Failed to create {nodes_path}")

    coords_node_path = out_dir / "coords_node_order.parquet"
    coords_node_path.unlink(missing_ok=True)
    nodes[["Cell ID", "i", "j", "k", "x", "y", "z"]].to_parquet(
        coords_node_path,
        index=False,
    )

    # Coordinates in file order and FILE index -> NODE id alignment map.
    df_file_order = df_file.copy(deep=True)
    df_file_order.loc[:, "Cell ID"] = df_file_order[
        "Cell ID"
    ].to_numpy(dtype=np.int64)
    df_file_order.loc[:, "x"] = df_file_order[
        "Cell center x"
    ].to_numpy(dtype=float)
    df_file_order.loc[:, "y"] = df_file_order[
        "Cell center y"
    ].to_numpy(dtype=float)
    df_file_order.loc[:, "z"] = df_file_order[
        "Cell center z"
    ].to_numpy(dtype=float)

    coords_file_path = out_dir / "coords_file_order.parquet"
    coords_file_path.unlink(missing_ok=True)
    df_file_order[["i", "j", "k", "x", "y", "z"]].to_parquet(
        coords_file_path,
        index=False,
    )
    detail(
        logger,
        "Wrote %s with %d rows",
        coords_file_path.name,
        len(df_file_order),
    )

    cellid_to_nodeid = dict(
        zip(nodes["Cell ID"].to_numpy(), nodes["node_id"].to_numpy())
    )
    file_cell_ids = df_file_order["Cell ID"].to_numpy(dtype=np.int64)

    column_map = np.empty_like(file_cell_ids, dtype=np.int64)
    for file_index, cell_id in enumerate(file_cell_ids):
        try:
            column_map[file_index] = cellid_to_nodeid[int(cell_id)]
        except KeyError as error:
            raise ValueError(
                f"Cell ID {cell_id} in FILE order was not found in the "
                "NODE map; this is unexpected for a fixed mesh"
            ) from error

    column_map_path = out_dir / "colmap_file_to_nodes.npy"
    np.save(column_map_path, column_map)
    detail(
        logger,
        "Wrote %s with %d entries",
        column_map_path.name,
        len(column_map),
    )

    cellmap_path = out_dir / "cellid_to_nodeid.parquet"
    cellmap_path.unlink(missing_ok=True)
    nodes[["Cell ID", "node_id"]].to_parquet(cellmap_path, index=False)

    edge_index = _nodes_to_edge_index(
        nodes,
        neighbor_set=neighbor_set,
        bidirectional=edge_bidir,
    )
    edges_df = pd.DataFrame(
        {"src": edge_index[0], "dst": edge_index[1]},
        dtype=np.int64,
    )

    edges_path = out_dir / f"edges_{neighbor_set}.csv"
    edges_path.unlink(missing_ok=True)
    edges_df.to_csv(edges_path, index=False)

    detail(
        logger,
        "Graph artifacts written: nodes=%d, directed_edges=%d",
        len(nodes),
        len(edges_df),
    )
    summarize_connectivity(nodes, edge_index, neighbor_set=neighbor_set)


def ensure_graph_artifacts(
    cfg,
    field_var: str,
    rebuild: bool = False,
    *,
    neighbor_set: str = "n6",
) -> Path:
    base_dir = Path(cfg["base_data_dir"])
    test_dir = cfg["test_dir"]
    source_dir = base_dir / test_dir

    graph_dir = Path(cfg["output_dir"]) / "graph" / test_dir
    snapshot_dir = graph_dir / "snapshots"
    graph_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    nodes_path = graph_dir / "nodes.parquet"
    edges_path = graph_dir / f"edges_{neighbor_set}.csv"
    cellmap_path = graph_dir / "cellid_to_nodeid.parquet"
    coords_file_path = graph_dir / "coords_file_order.parquet"
    column_map_path = graph_dir / "colmap_file_to_nodes.npy"

    need_nodes_edges = rebuild or any(
        not path.exists()
        for path in (
            nodes_path,
            edges_path,
            cellmap_path,
            coords_file_path,
            column_map_path,
        )
    )
    need_targets = rebuild or not any(snapshot_dir.glob("target_*.*"))

    if not _source_has_raw_cell(source_dir):
        raise FileNotFoundError(
            f"No Raw.cell.*.npy files found under {source_dir}"
        )

    if need_nodes_edges:
        reference_df, file_order_df = _select_reference_frames(source_dir)
        _build_nodes_and_edges(
            reference_df,
            file_order_df,
            graph_dir,
            edge_bidir=True,
            neighbor_set=neighbor_set,
        )
    else:
        logger.info("Using existing Eulerian graph artifacts")

    if need_targets:
        logger.info("Generating Eulerian snapshot targets")
        nodes = (
            pd.read_parquet(nodes_path)
            .sort_values("node_id")
            .reset_index(drop=True)
        )
        cellmap = pd.read_parquet(cellmap_path)
        node_id_order = nodes[["node_id", "Cell ID"]].copy()

        entries = _build_cell_time_index(source_dir)
        detail(
            logger,
            "Generating %d targets for field '%s'",
            len(entries),
            field_var,
        )

        first_json = _pair_npy_with_json(entries[0][1])
        colnames = _resolve_colnames_from_json(first_json)
        if field_var not in colnames:
            raise ValueError(
                f"field_var '{field_var}' not found in Raw.cell JSON "
                f"columns. Available: {colnames}"
            )

        for simulation_time, npy_path in tqdm(
            entries,
            total=len(entries),
            desc="Targets: from Raw.cell",
            leave=False,
            disable=not progress_enabled(),
        ):
            json_path = _pair_npy_with_json(npy_path)
            snapshot_df = _load_raw_cell_df(npy_path, json_path)

            _require_cols(
                snapshot_df,
                ["Cell ID", field_var],
                context=f"targets @ t={simulation_time}",
            )

            snapshot_df = snapshot_df.copy(deep=True)
            snapshot_df.loc[:, "Cell ID"] = snapshot_df[
                "Cell ID"
            ].to_numpy(dtype=np.int64)

            snapshot = snapshot_df.loc[:, ["Cell ID", field_var]].copy()
            snapshot = snapshot.merge(cellmap, on="Cell ID", how="left")

            if snapshot["node_id"].isna().any():
                missing_count = int(snapshot["node_id"].isna().sum())
                raise ValueError(
                    f"{missing_count} rows in snapshot have Cell ID not "
                    "found in reference mapping"
                )

            snapshot.loc[:, "node_id"] = snapshot["node_id"].to_numpy(
                dtype=np.int64
            )
            snapshot = (
                node_id_order.merge(
                    snapshot[["node_id", field_var]],
                    on="node_id",
                    how="left",
                )
                .sort_values("node_id")
            )

            if snapshot[field_var].isna().any():
                missing_count = int(snapshot[field_var].isna().sum())
                raise ValueError(
                    f"{missing_count} nodes missing '{field_var}' after "
                    "alignment; this is unexpected for a fixed mesh"
                )

            output_path = (
                snapshot_dir / f"target_{simulation_time:09.3f}s.parquet"
            )
            output_path.parent.mkdir(exist_ok=True, parents=True)
            snapshot[[field_var]].to_parquet(output_path, index=False)

        detail(
            logger,
            "Eulerian snapshot targets written to %s",
            snapshot_dir,
        )
    else:
        logger.info("Using existing Eulerian snapshot targets")

    return graph_dir


# --------------------------------------------------------------------------------------
# Public API for in-memory edge construction (integer ijk only)
# --------------------------------------------------------------------------------------


def build_edge_index(
    coords: Union[
        pd.DataFrame,
        pd.Index,
        Iterable[Tuple[int, int, int]],
    ],
    *,
    neighbor_set: str = "n6",
    bidirectional: bool = True,
) -> np.ndarray:
    if isinstance(coords, pd.DataFrame):
        if not {"i", "j", "k"}.issubset(coords.columns):
            raise TypeError("DataFrame must include columns ['i','j','k']")
        nodes = coords[["i", "j", "k"]].copy().reset_index(drop=True)
    elif isinstance(coords, (pd.MultiIndex, pd.Index)):
        ijk = np.asarray(list(coords))
        if ijk.ndim != 2 or ijk.shape[1] != 3:
            raise TypeError("Index must contain (i,j,k) triplets")
        if not np.issubdtype(ijk.dtype, np.integer):
            raise TypeError("Index levels must be integer (i,j,k)")
        nodes = pd.DataFrame(ijk, columns=["i", "j", "k"])
    else:
        ijk = np.asarray(list(coords))
        if ijk.ndim != 2 or ijk.shape[1] != 3:
            raise TypeError("Iterable must yield (i,j,k) triplets")
        if not np.issubdtype(ijk.dtype, np.integer):
            raise TypeError(
                "Iterable must yield integer (i,j,k) values"
            )
        nodes = pd.DataFrame(ijk, columns=["i", "j", "k"])

    nodes.insert(0, "node_id", np.arange(len(nodes), dtype=np.int64))
    return _nodes_to_edge_index(
        nodes,
        neighbor_set=neighbor_set,
        bidirectional=bidirectional,
    )


# --------------------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------------------


def summarize_connectivity(
    nodes: Union[pd.DataFrame, int],
    edge_index: np.ndarray,
    *,
    neighbor_set: str = "n6",
) -> dict:
    node_count = len(nodes) if not isinstance(nodes, int) else nodes
    edge_count = int(edge_index.shape[1]) if edge_index.size else 0
    degree = np.bincount(
        edge_index[0], minlength=node_count
    ) + np.bincount(edge_index[1], minlength=node_count)

    stats = {
        "N": node_count,
        "E_directed": edge_count,
        "deg_min": int(degree.min()) if node_count else 0,
        "deg_median": float(np.median(degree)) if node_count else 0.0,
        "deg_p75": (
            float(np.percentile(degree, 75)) if node_count else 0.0
        ),
        "deg_max": int(degree.max()) if node_count else 0,
        "deg_mean": float(degree.mean()) if node_count else 0.0,
        "expected_E_approx": (
            6 * node_count
            if neighbor_set == "n6"
            else 18 * node_count
            if neighbor_set == "n18"
            else 26 * node_count
        ),
        "zero_deg": int((degree == 0).sum()) if node_count else 0,
    }

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            (
                "Graph connectivity: nodes=%d, directed_edges=%d, "
                "degree[min/median/p75/max/mean]=%d/%.1f/%.1f/%d/%.2f, "
                "zero_degree_nodes=%d"
            ),
            stats["N"],
            stats["E_directed"],
            stats["deg_min"],
            stats["deg_median"],
            stats["deg_p75"],
            stats["deg_max"],
            stats["deg_mean"],
            stats["zero_deg"],
        )

    return stats


def assert_feature_alignment(
    n_features: int,
    edge_index: np.ndarray,
) -> None:
    graph_node_count = int(edge_index.max()) + 1 if edge_index.size else 0
    assert n_features == graph_node_count, (
        f"Feature nodes {n_features} != graph nodes {graph_node_count}. "
        "Ensure consistent node order for features, graph, writer."
    )
