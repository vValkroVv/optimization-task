# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Apportion shards/samples to nodes/ranks/workers for elastically deterministic sample order."""

import logging
import math
import warnings
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1 << 18


def _node_geometry(num_samples: int, num_canonical_nodes: int, num_physical_nodes: int,
                   drop_first: int) -> tuple[int, int, int, int]:
    """Check the node counts and size the (canonical nodes, samples) matrix of ``orig``.

    Args:
        num_samples (int): Dataset size.
        num_canonical_nodes (int): Number of canonical nodes.
        num_physical_nodes (int): Number of physical nodes.
        drop_first (int): Number of samples seen already, which are dropped.

    Returns:
        Tuple[int, int, int, int]: ``drop_first`` rounded down to a multiple of the physical
            nodes, samples per canonical node, physical nodes per canonical node (``0`` unless
            there are more physical nodes) and the row padding that makes them divide evenly.
    """
    if num_canonical_nodes % num_physical_nodes and num_physical_nodes % num_canonical_nodes:
        raise ValueError('Either canonical or physical nodes must be evenly divisible by ' +
                         'the other, otherwise striping slices of shards over nodes may ' +
                         'lead to each node downloading all shards')

    # If drop_first isn't a multiple of num_physical_nodes, round down to make it divisible.
    if drop_first % num_physical_nodes:
        logger.warning(
            '`drop_first` was not divisible by `num_physical_nodes`. Rounding it down ' +
            'to make it divisible.')
        drop_first -= drop_first % num_physical_nodes

    # Divide the full dataset sample range into a sample range per canonical node.
    samples_per_canonical_node = (num_samples + num_canonical_nodes - 1) // num_canonical_nodes
    node_ratio = 0
    padding = 0
    if num_canonical_nodes < num_physical_nodes:
        node_ratio = num_physical_nodes // num_canonical_nodes
        padding = -samples_per_canonical_node % node_ratio
    return drop_first, samples_per_canonical_node, node_ratio, padding


def _rank_geometry(samples_per_node: int, ranks_per_node: int, workers_per_rank: int,
                   batch_size: int) -> tuple[int, int, int]:
    """Size a physical node's samples over its ranks and workers the way ``orig`` pads them.

    Args:
        samples_per_node (int): Samples of the physical node.
        ranks_per_node (int): Number of ranks per node.
        workers_per_rank (int): Number of worker partitions per rank.
        batch_size (int): Batch size of DataLoader and dataset.

    Returns:
        Tuple[int, int, int]: Samples repeated to split the node evenly over its ranks, samples
            per rank, and batches per worker once the rank is padded with ``-1``.
    """
    rank_underflow = -samples_per_node % ranks_per_node
    samples_per_rank = (samples_per_node + rank_underflow) // ranks_per_node
    num_batches = math.ceil(samples_per_rank / (workers_per_rank * batch_size))
    return rank_underflow, samples_per_rank, num_batches


def get_partitions_orig(num_samples: int,
                        num_canonical_nodes: int,
                        num_physical_nodes: int,
                        ranks_per_node: int,
                        workers_per_rank: int,
                        batch_size: int,
                        drop_first: int = 0,
                        initial_physical_nodes: Optional[int] = None) -> NDArray[np.int64]:
    """Partition the given number of samples to nodes, ranks, and workers.

    Either canonical or physical nodes must be evenly divisible by the other.

    It is suggested to set num_canonical_nodes higher than your expected number of physical nodes,
    because scaling your number of nodes below that level may result in more shards being used
    across node boundaries due to preserving the same global sample order.

    Args:
        num_samples (int): Dataset size.
        num_canonical_nodes (int): Number of canonical nodes.
        num_physical_nodes (int): Number of physical nodes.
        ranks_per_node (int): Number of ranks per node.
        workers_per_rank (int): Number of worker partitions per rank.
        batch_size (int): Batch size of DataLoader and dataset, which affects how the dataset is
            partitioned over the workers.
        drop_first (int): Number of samples seen already, which are dropped. Defaults to ``0``.
        initial_physical_nodes (int, optional): Number of physical nodes at the start of training.
            Defaults to ``None``.

    Returns:
        NDArray[np.int64]: Partitions of shape (physical nodes, ranks per node, workers per rank,
            batches per worker, batch size).
    """
    drop_first, samples_per_canonical_node, node_ratio, padding = _node_geometry(
        num_samples, num_canonical_nodes, num_physical_nodes, drop_first)
    padded_samples_per_canonical_node = samples_per_canonical_node + padding

    # For samples to be properly split across canonical nodes, there must be more samples than nodes.
    # The edge case is when the number of samples is equal to the number of canonical nodes, but this only works when
    # there is an equal or greater number of canonical nodes than physical nodes.
    # If these conditions are not met, an alternative sampling approach is used that leads to many repeats.
    if num_samples > num_canonical_nodes or (num_samples == num_canonical_nodes and
                                             num_canonical_nodes >= num_physical_nodes):
        # Create the initial sample ID matrix.
        #
        # ids: (canonical nodes, padded samples per canonical node).
        ids = np.arange(num_canonical_nodes * padded_samples_per_canonical_node, dtype=np.int64)
        ids = ids.reshape(num_canonical_nodes, padded_samples_per_canonical_node)

        # Adjust row offsets to ignore the padding.
        #
        # row_offsets: (canonical nodes, 1).
        row_offsets = np.arange(num_canonical_nodes) * padding
        row_offsets = np.expand_dims(row_offsets, 1)
        ids -= row_offsets

        # Reconfigure where each row starts iterating for irregular-sized rows.
        #
        # row_starts: (canonical nodes, 1).
        row_starts = np.arange(num_canonical_nodes) * num_samples // num_canonical_nodes
        row_starts = np.expand_dims(row_starts, 1)
        ids += row_starts - ids[:, :1]

        # For short rows (length not evenly divisible), repeat the last ID to get even length.
        #
        # row_stops: (canonical nodes, 1).
        row_stops = np.arange(1, 1 + num_canonical_nodes) * num_samples // num_canonical_nodes
        row_stops = np.expand_dims(row_stops, 1)
        are_rows_short = row_stops - row_starts < samples_per_canonical_node
        ids[:, samples_per_canonical_node - 1:samples_per_canonical_node] -= are_rows_short

        # If padding we needed, repeat samples to populate it.
        if padding:
            ids[:, -padding:] = ids[:,
                                    -padding - node_ratio + 1 - padding:-padding - node_ratio + 1]
    else:
        warnings.warn(f'Trying to partition {num_samples} samples over {num_canonical_nodes}' +
                      f' canonical nodes. This will result in many samples being repeated, and ' +
                      f'depending on your batching method, batches being completely dropped. ' +
                      f'Check if your dataset has the expected number of samples, and consider ' +
                      f'decreasing the number of canonical nodes.')
        shape_needed = (num_canonical_nodes, padded_samples_per_canonical_node)
        total_samples_needed = num_canonical_nodes * padded_samples_per_canonical_node
        current_samples = np.arange(num_samples, dtype=np.int64)
        full_repeats = total_samples_needed // num_samples
        leftover_samples = total_samples_needed % num_samples
        ids = np.concatenate(
            [np.tile(current_samples, full_repeats), current_samples[:leftover_samples]])
        ids = ids.reshape(shape_needed)

    # Flatten, drop samples that have already been seen, reshape back.
    #
    # ids: (physical nodes, samples per node).
    ids = ids.transpose()
    ids = ids.flatten()
    ids = ids[drop_first:]
    ids = ids.reshape(-1, num_physical_nodes)
    ids = ids.transpose()

    # Interleave the node sample ranges over each node's ranks, padding with -1 for reshaping.
    #
    # ids: (physical nodes, samples per rank, ranks per node).
    samples_per_node = ids.shape[1]
    rank_underflow, samples_per_rank, num_batches = _rank_geometry(samples_per_node,
                                                                   ranks_per_node,
                                                                   workers_per_rank, batch_size)
    if rank_underflow:
        enough_padding_samples = ranks_per_node + rank_underflow - 1 <= samples_per_node
        if enough_padding_samples:
            last = ids[:, -ranks_per_node - rank_underflow + 1:-ranks_per_node + 1]
        else:
            # There are less samples than ranks. Usually, we pad by trying to ensure that the same
            # samples don't get repeated over and over, but with in this case, we are forced to.
            warnings.warn(
                f'Attempting to partition {samples_per_node} samples per physical node ' +
                f'over {ranks_per_node} gpus. This will result in many samples being ' +
                f'repeated, and depending on your batching method, batches being ' +
                f'completely dropped. Check if your dataset has the expected number ' +
                f'of samples.')
            full_repeats = rank_underflow // samples_per_node
            leftover_samples = rank_underflow % samples_per_node
            last = np.concatenate([np.tile(ids, full_repeats), ids[:, :leftover_samples]], 1)
        ids = np.concatenate([ids, last], 1)

    ids = ids.reshape(num_physical_nodes, -1, ranks_per_node)

    # Pad with -1 adequately for reshaping across workers.
    #
    # ids: (physical nodes, samples per rank, ranks per node).
    overflow = num_batches * workers_per_rank * batch_size - samples_per_rank
    if overflow:
        last = np.full((num_physical_nodes, overflow, ranks_per_node), -1, np.int64)
        ids = np.concatenate([ids, last], 1)

    # Interleave each rank's padded samples across its workers.
    #
    # ids: (physical nodes, ranks per node, workers per rank, batches per worker, batch size).
    ids = ids.transpose(0, 2, 1)
    ids = ids.reshape(num_physical_nodes, ranks_per_node, -1, workers_per_rank, batch_size)
    return ids.transpose(0, 1, 3, 2, 4)


def get_partitions_for_worker_orig(
        num_samples: int,
        num_canonical_nodes: int,
        num_physical_nodes: int,
        ranks_per_node: int,
        workers_per_rank: int,
        batch_size: int,
        drop_first: int = 0,
        initial_physical_nodes: Optional[int] = None,
        target_node: int = 0,
        target_rank: int = 0,
        target_worker: int = 0,
        row_bases: Optional[NDArray[np.int64]] = None) -> NDArray[np.int64]:
    """Compute partition indices for a single worker without building the full array.

    Produces the same result as
    ``get_partitions_orig(...)[target_node, target_rank, target_worker].flatten()`` but without
    allocating the full 5D array (which can be tens of GB). Every reshape/transpose/padding step
    of ``get_partitions_orig`` is mirrored analytically on this worker's flat indices only, in
    chunks of ``CHUNK_SIZE`` positions to bound temporary memory.

    Args:
        num_samples (int): Dataset size (epoch_size after resampling).
        num_canonical_nodes (int): Number of canonical nodes.
        num_physical_nodes (int): Number of physical nodes.
        ranks_per_node (int): Number of ranks per node.
        workers_per_rank (int): Number of worker partitions per rank.
        batch_size (int): Batch size of DataLoader and dataset.
        drop_first (int): Number of samples seen already, which are dropped. Defaults to ``0``.
        initial_physical_nodes (int, optional): Unused by ``orig``, kept for signature parity.
            Defaults to ``None``.
        target_node (int): Physical node index for this worker. Defaults to ``0``.
        target_rank (int): Rank-within-node index for this worker. Defaults to ``0``.
        target_worker (int): Worker-within-rank index for this worker. Defaults to ``0``.
        row_bases (NDArray[np.int64], optional): Where each canonical node's samples start in
            the caller's ID space, for callers that hold only the rows this worker reads (see
            ``get_node_rows``). Defaults to the rows' sample ID starts.

    Returns:
        NDArray[np.int64]: 1D array of "big" sample IDs for this worker, identical to
            ``get_partitions_orig(...)[target_node, target_rank, target_worker].flatten()``.
    """
    drop_first, samples_per_canonical_node, node_ratio, padding = _node_geometry(
        num_samples, num_canonical_nodes, num_physical_nodes, drop_first)
    padded_samples_per_canonical_node = samples_per_canonical_node + padding

    total_flat = num_canonical_nodes * padded_samples_per_canonical_node
    samples_per_node = max(total_flat - drop_first, 0) // num_physical_nodes
    rank_underflow, samples_per_rank, num_batches = _rank_geometry(samples_per_node,
                                                                   ranks_per_node,
                                                                   workers_per_rank, batch_size)
    worker_total = num_batches * batch_size

    regular = num_samples > num_canonical_nodes or (num_samples == num_canonical_nodes and
                                                    num_canonical_nodes >= num_physical_nodes)
    cn_indices = np.arange(num_canonical_nodes, dtype=np.int64)
    row_starts = cn_indices * num_samples // num_canonical_nodes
    row_stops = (cn_indices + 1) * num_samples // num_canonical_nodes
    are_rows_short = (row_stops - row_starts) < samples_per_canonical_node
    if row_bases is None:
        row_bases = row_starts

    # Node padding: ids[:, -padding:] = ids[:, src], with numpy slice clipping and broadcasting
    src_start = src_len = 0
    if regular and padding:
        src = slice(-padding - node_ratio + 1 - padding, -padding - node_ratio + 1)
        src_start, src_stop, _ = src.indices(padded_samples_per_canonical_node)
        src_len = max(src_stop - src_start, 0)
        if src_len != padding and src_len != 1:
            raise ValueError(f'Cannot pad {padding} columns from a source slice of length ' +
                             f'{src_len} (num_samples={num_samples}, ' +
                             f'num_canonical_nodes={num_canonical_nodes}, ' +
                             f'num_physical_nodes={num_physical_nodes})')

    result = np.empty(worker_total, np.int64)
    for start in range(0, worker_total, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, worker_total)
        i = np.arange(start, stop, dtype=np.int64)

        # j: position within the rank in (batch, sample-in-batch) order
        j = (i // batch_size) * (workers_per_rank * batch_size) + target_worker * batch_size
        j += i % batch_size
        k = j * ranks_per_node + target_rank

        # Rank padding: columns beyond samples_per_node repeat existing ones
        if rank_underflow:
            rank_pad = k >= samples_per_node
            if ranks_per_node + rank_underflow - 1 <= samples_per_node:
                # Mirrors ids[:, -ranks_per_node - underflow + 1:-ranks_per_node + 1]
                k[rank_pad] -= ranks_per_node + rank_underflow - 1
            else:
                # Mirrors concatenate([tile(ids, full_repeats), ids[:, :leftover]], 1)
                k[rank_pad] = (k[rank_pad] - samples_per_node) % samples_per_node

        # Column-major index into the (canonical nodes, padded samples) matrix
        flat = k * num_physical_nodes + target_node + drop_first
        cn = flat % num_canonical_nodes
        pos = flat // num_canonical_nodes

        if regular:
            if padding:
                node_pad = pos >= samples_per_canonical_node
                if src_len == padding:
                    pos[node_pad] += src_start - samples_per_canonical_node
                else:
                    pos[node_pad] = src_start
            # ids[cn, pos] = row_starts[cn] + pos, last ID repeated for short rows
            chunk = row_bases[cn] + pos
            chunk[(pos == samples_per_canonical_node - 1) & are_rows_short[cn]] -= 1
        else:
            # Tiled-arange fallback for too few samples per canonical node: every row holds at
            # most one sample, so an ID is the base of the last row starting at or before it
            chunk = (cn * padded_samples_per_canonical_node + pos) % num_samples
            chunk = row_bases[np.searchsorted(row_starts, chunk, side='right') - 1]

        # Worker padding: positions beyond the real samples per rank become -1
        chunk[j >= samples_per_rank] = -1
        result[start:stop] = chunk
    return result


def get_node_rows(
    num_samples: int, num_canonical_nodes: int, num_physical_nodes: int,
    target_nodes: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Canonical nodes whose samples reach the given physical nodes, with their sample ID ranges.

    ``get_partitions_orig`` walks the (canonical nodes, samples per canonical node) matrix
    column-major and hands physical node ``n`` the flat indices ``k * num_physical_nodes + n +
    drop_first``. As the node counts divide one another and ``drop_first`` is a multiple of
    ``num_physical_nodes``, those indices only fall into rows ``c == n`` modulo the smaller node
    count, so the node never needs the other rows.

    Args:
        num_samples (int): Dataset size.
        num_canonical_nodes (int): Number of canonical nodes.
        num_physical_nodes (int): Number of physical nodes.
        target_nodes (NDArray[np.int64]): Physical node indices.

    Returns:
        Tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]: ``(rows, starts, stops)``:
            the canonical node indices in increasing order and their sample ID ranges.
    """
    if num_samples <= num_canonical_nodes:
        # At most one sample per row, so every row is published (covers the tiled fallback too)
        rows = np.arange(num_canonical_nodes)
    else:
        step = min(num_physical_nodes, num_canonical_nodes)
        residues = np.unique(target_nodes % step)
        rows = (np.arange(0, num_canonical_nodes, step)[:, None] + residues).ravel()
    starts = rows * num_samples // num_canonical_nodes
    stops = (rows + 1) * num_samples // num_canonical_nodes
    return rows, starts, stops
