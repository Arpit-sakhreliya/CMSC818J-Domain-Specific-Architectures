#!/usr/bin/env python3
"""
it converts sparse matrices into a tile-based compressed format
and compares storage sizes with COO and CSR formats.


Scans MAT_data/*.mtx and for each matrix:
 - prints its COO data + metadata
 - converts to CSR and prints CSR metadata
 - creates a tile-based compression (64x64 tiles, incomplete tiles kept),
   groups identical tiles and stores unique tiles as COO; repeated tiles
   are stored as pointers to the first occurrence.
 - prints approximate storage (bytes) for COO, CSR, and tile-based method

Read the matrix and convert it into a dense format. Then tile it into 64×64 blocks. Any tile that does not fit perfectly should be left as it is. At this point, you will have a matrix of tiles, containing both complete and incomplete tiles.

Now, we need to compare each tile with every other tile to check whether they are exactly equal in both values and dimensions. First compare the dimensions, and then compare all the values. If two tiles are equal, we will create a compressed format such as:

[row, column of the tile, [tile data]],

[row, column of the tile, and instead of tile data, the row and column of the tile it is equal to]]

The tile data will be stored in COO format. and the tile index and pointer will be in int16 format.

####
tile update is easy due to search will be easy to serach firts tile and then update the tile elements
####

"""

import os
import glob
import hashlib
import numpy as np
from scipy import io as sio
from scipy.sparse import coo_matrix, csr_matrix
import matplotlib.pyplot as plt

# Parameters
DATA_DIR = "CU_sparse"   # Mat_data CU_sparse
TILE = 64
INT_BYTE = 8  # approximate bytes per stored integer (use 8 -> int64)
FLOAT_BYTE = 8  # float64
INT_16 = 2 # for tile index and pointers means 65535 values max


def print_coo_info(mat_coo, label="COO"):
    """Print basic metadata and a small sample of arrays from a coo_matrix."""
    print(f"--- {label} ---")
    print(f"shape: {mat_coo.shape}, dtype: {mat_coo.dtype}, nnz: {mat_coo.nnz}")
    # array sizes
    print(f"row array length: {len(mat_coo.row)}, col array length: {len(mat_coo.col)}, data length: {len(mat_coo.data)}")
    # # show small sample
    # sample_n = min(10, mat_coo.nnz)
    # if sample_n > 0:
    #     print("sample entries (row, col, value):")
    #     for i in range(sample_n):
    #         print(f"  ({mat_coo.row[i]}, {mat_coo.col[i]}, {mat_coo.data[i]})")
    # else:
    #     print("matrix has no nonzeros")
    # print()

def print_csr_info(mat_csr, label="CSR"):
    """Print basic metadata about CSR matrix."""
    print(f"--- {label} ---")
    print(f"shape: {mat_csr.shape}, dtype: {mat_csr.dtype}, nnz: {mat_csr.nnz}")
    print(f"data length: {len(mat_csr.data)}, indices length: {len(mat_csr.indices)}, indptr length: {len(mat_csr.indptr)}")
    # # sample few nonzeros by converting small portion to coo for readability
    # sample_coo = mat_csr.tocoo(copy=False)
    # sample_n = min(10, sample_coo.nnz)
    # if sample_n > 0:
    #     print("sample entries (row, col, value):")
    #     for i in range(sample_n):
    #         print(f"  ({sample_coo.row[i]}, {sample_coo.col[i]}, {sample_coo.data[i]})")
    # print()

def approx_sparse_bytes_coo(mat_coo):
    """Approximate bytes used by COO arrays (row, col, data). Use actual nbytes when possible."""
    # If underlying arrays are numpy arrays we can get nbytes directly
    try:
        bytes_total = int(mat_coo.row.nbytes + mat_coo.col.nbytes + mat_coo.data.nbytes)
    except Exception:
        # fallback: estimate
        bytes_total = mat_coo.nnz * (INT_BYTE + INT_BYTE + FLOAT_BYTE)
    return bytes_total

def approx_sparse_bytes_csr(mat_csr):
    try:
        bytes_total = int(mat_csr.data.nbytes + mat_csr.indices.nbytes + mat_csr.indptr.nbytes)
    except Exception:
        # fallback: data length * float, indices length * int, indptr length * int
        bytes_total = mat_csr.nnz * FLOAT_BYTE + len(mat_csr.indices) * INT_BYTE + len(mat_csr.indptr) * INT_BYTE
    return bytes_total

def tile_hash_and_shape(tile_arr):
    """Return a digest that depends on tile shape and contents + dtype."""
    h = hashlib.sha256()
    # include shape and dtype as bytes
    h.update(np.array(tile_arr.shape, dtype=np.int64).tobytes())
    h.update(str(tile_arr.dtype).encode('utf-8'))
    # for stable behavior, convert to contiguous array and include its bytes
    h.update(np.ascontiguousarray(tile_arr).tobytes())
    return h.hexdigest()

def tile_to_coo(tile_arr):
    """Convert numpy 2D tile array to a local COO (rows and cols relative to tile)."""
    if tile_arr.size == 0:
        return coo_matrix((0, 0))
    # Find nonzeros (we treat exact zeros as zeros)
    nz_mask = tile_arr != 0
    if not np.any(nz_mask):
        # empty COO of same shape
        return coo_matrix((tile_arr.shape))
    rows, cols = np.nonzero(nz_mask)
    data = tile_arr[rows, cols]
    return coo_matrix((data, (rows, cols)), shape=tile_arr.shape)

def tile_compress_dense(dense, tile_size=TILE):
    nrows, ncols = dense.shape
    tile_rows = (nrows + tile_size - 1) // tile_size
    tile_cols = (ncols + tile_size - 1) // tile_size

    tiles_info = []        # only for NON-ZERO tiles
    unique_tiles = {}
    hash_to_reps = {}

    total_tiles = tile_rows * tile_cols
    zero_tiles = 0

    for tr in range(tile_rows):
        r0 = tr * tile_size
        r1 = min(r0 + tile_size, nrows)
        for tc in range(tile_cols):
            c0 = tc * tile_size
            c1 = min(c0 + tile_size, ncols)
            tile = dense[r0:r1, c0:c1]

            # ✅ SKIP ZERO TILES COMPLETELY
            if not np.any(tile):
                zero_tiles += 1
                continue

            key = tile_hash_and_shape(tile)
            found = False

            if key in hash_to_reps:
                for rep in hash_to_reps[key]:
                    if rep['tile_arr'].shape == tile.shape and np.array_equal(rep['tile_arr'], tile):
                        tiles_info.append({
                            'tile_pos': (tr, tc),
                            'stored_as': 'ptr',
                            'ptr_to': rep['pos'],
                            'data_key': key
                        })
                        unique_tiles[key]['count'] += 1
                        found = True
                        break

            if not found:
                coo = tile_to_coo(tile)
                unique_tiles[key] = {
                    'coo': coo,
                    'first_pos': (tr, tc),
                    'count': 1
                }
                hash_to_reps.setdefault(key, []).append({
                    'tile_arr': tile.copy(),
                    'pos': (tr, tc),
                    'key': key
                })
                tiles_info.append({
                    'tile_pos': (tr, tc),
                    'stored_as': 'data',
                    'data_key': key
                })

    stats = {
        'num_tiles_total': total_tiles,
        'num_zero_tiles': zero_tiles,
        'num_nonzero_tiles': total_tiles - zero_tiles,
        'num_unique_tiles': len(unique_tiles),
        'tile_rows': tile_rows,
        'tile_cols': tile_cols,
        'tile_size': tile_size
    }

    return tiles_info, unique_tiles, stats

def approx_tile_storage_bytes(tiles_info, unique_tiles):
    total_nonzero_tiles = len(tiles_info)

    # Only NON-ZERO tiles store positions
    bytes_positions = total_nonzero_tiles * 2 * INT_16

    # Only pointer tiles store pointer coords
    bytes_ptrs = 0
    for t in tiles_info:
        if t['stored_as'] == 'ptr':
            bytes_ptrs += 2 * INT_16

    # Unique tile COO storage
    unique_bytes = 0
    for v in unique_tiles.values():
        coo = v['coo']
        try:
            unique_bytes += coo.row.nbytes + coo.col.nbytes + coo.data.nbytes
        except:
            unique_bytes += coo.nnz * (2 * INT_BYTE + FLOAT_BYTE)

    total_bytes = bytes_positions + bytes_ptrs + unique_bytes

    return total_bytes, {
        'positions_nbytes': bytes_positions,
        'ptrs_nbytes': bytes_ptrs,
        'unique_tiles_nbytes': unique_bytes
    }

def plot_compression_ratios(results, output_name="tile_compression_ratios.png"):
    """
    Plots bar charts of Tile/COO and Tile/CSR ratios per file
    with slanted, readable filenames on x-axis.
    """

    filenames = []
    tile_to_coo = []
    tile_to_csr = []

    for r in results:
        if r['tile_bytes'] is None:
            continue

        fname = os.path.basename(r['path'])
        coo_b = r['coo_bytes']
        csr_b = r['csr_bytes']
        tile_b = r['tile_bytes']

        if coo_b > 0:
            tile_to_coo.append(tile_b / coo_b)
        else:
            tile_to_coo.append(0)

        if csr_b > 0:
            tile_to_csr.append(tile_b / csr_b)
        else:
            tile_to_csr.append(0)

        filenames.append(fname)

    x = np.arange(len(filenames))
    width = 0.35

    plt.figure(figsize=(14, 6))
    plt.bar(x - width/2, tile_to_coo, width, label="CTL / COO") #compressed tile list
    plt.bar(x + width/2, tile_to_csr, width, label="CTL / CSR") #compressed tile list

    plt.xticks(x, filenames, rotation=30, ha="right")
    plt.ylabel("Compression Ratio")
    plt.xlabel("Matrix File")
    plt.title("Tile-based Compression Ratios per File")
    plt.legend()
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(output_name, dpi=200)
    plt.close()

    print(f"\n✅ Compression ratio plot saved as: {output_name}")

def analyze_tile_duplicates(matrix, tile_size=64):
    """
    Converts matrix to dense if needed, splits into n×n tiles,
    detects zero tiles, finds duplicate tiles, and categorizes
    duplicates into:
      - same tile row
      - same tile column
      - elsewhere
    """

    # ✅ Convert to dense if sparse
    if not isinstance(matrix, np.ndarray):
        dense = matrix.toarray()
    else:
        dense = matrix.copy()

    nrows, ncols = dense.shape
    tile_rows = (nrows + tile_size - 1) // tile_size
    tile_cols = (ncols + tile_size - 1) // tile_size

    tiles = {}
    zero_tiles = 0

    # ✅ STEP 1: Extract all nonzero tiles
    for tr in range(tile_rows):
        r0 = tr * tile_size
        r1 = min(r0 + tile_size, nrows)
        for tc in range(tile_cols):
            c0 = tc * tile_size
            c1 = min(c0 + tile_size, ncols)

            tile = dense[r0:r1, c0:c1]

            if not np.any(tile):
                zero_tiles += 1
                continue

            key = hashlib.sha256(
                np.ascontiguousarray(tile).tobytes()
            ).hexdigest()

            tiles.setdefault(key, {
                "tile": tile,
                "positions": []
            })["positions"].append((tr, tc))

    # ✅ STEP 2: Categorize duplicates
    same_row = 0
    same_col = 0
    elsewhere = 0
    total_duplicates = 0

    for entry in tiles.values():
        pos = entry["positions"]
        if len(pos) > 1:
            total_duplicates += len(pos) - 1

            base_r, base_c = pos[0]
            for (r, c) in pos[1:]:
                if r == base_r:
                    same_row += 1
                elif c == base_c:
                    same_col += 1
                else:
                    elsewhere += 1

    stats = {
        "matrix_shape": dense.shape,
        "tile_size": tile_size,
        "tile_grid": (tile_rows, tile_cols),
        "total_tiles": tile_rows * tile_cols,
        "zero_tiles": zero_tiles,
        "nonzero_tiles": tile_rows * tile_cols - zero_tiles,
        "unique_tiles": len(tiles),
        "duplicate_tiles": total_duplicates,
        "duplicates_same_row": same_row,
        "duplicates_same_col": same_col,
        "duplicates_elsewhere": elsewhere
    }

    return stats

def theoretical_flop_memory_model(tile_stats, dup_stats, matrix_shape, tile_size, dtype_bytes=8):
    """
    Theoretical FLOP + memory model for:
      - Baseline tiled SpMM
      - Your TC-SpMM method (row/col/elsewhere reuse)

    Assumptions:
      - B is dense and same size as A
      - Tile multiply cost = 2 * tr * tc * p FLOPs
      - Tile A read = tr * tc elements
      - Tile B read = tc * p elements
      - Output tile C write = tr * p elements
    """

    m, n = matrix_shape
    tr = tc = tile_size
    p = n  # since B is same size as A

    # --- Tile counts ---
    T_nz = tile_stats["num_nonzero_tiles"]
    unique = tile_stats["num_unique_tiles"]

    row_dup = dup_stats["duplicates_same_row"]
    col_dup = dup_stats["duplicates_same_col"]
    elsewhere_dup = dup_stats["duplicates_elsewhere"]

    # --- FLOP MODEL ---

    tile_flops = 2 * tr * tc * p

    # Baseline FLOPs (each nonzero tile does one multiply)
    baseline_flops = T_nz * tile_flops

    # TC-SpMM FLOPs
    tc_flops = baseline_flops

    # Row reuse saves:
    # Each row-duplicate removes one tile multiply, but adds a B-sum cost
    # Saving per row duplicate:
    #   saved = tile_flops - (tc * p)
    row_saving = row_dup * (tile_flops - (tc * p))
    tc_flops -= row_saving

    # Column reuse saves:
    # Each column duplicate removes one full tile multiply
    col_saving = col_dup * tile_flops
    tc_flops -= col_saving

    # Elsewhere: no FLOP savings

    flop_reduction_pct = 100.0 * (baseline_flops - tc_flops) / max(baseline_flops, 1)

    # --- MEMORY MODEL (GLOBAL MEMORY TRAFFIC, ELEMENTS) ---

    tile_A_elems = tr * tc
    tile_B_elems = tc * p
    tile_C_elems = tr * p

    # Baseline memory (per nonzero tile)
    baseline_mem_elems = T_nz * (tile_A_elems + tile_B_elems + tile_C_elems)

    # TC-SpMM memory:
    # A tiles only unique loaded
    A_mem = unique * tile_A_elems

    # B blocks:
    # Row duplicates require extra B reads for summation,
    # so we approximate: B reads = (T_nz - col_dup)
    B_mem = (T_nz - col_dup) * tile_B_elems

    # C writes unchanged
    C_mem = T_nz * tile_C_elems

    tc_mem_elems = A_mem + B_mem + C_mem

    mem_reduction_pct = 100.0 * (baseline_mem_elems - tc_mem_elems) / max(baseline_mem_elems, 1)

    # Convert to bytes
    baseline_mem_bytes = baseline_mem_elems * dtype_bytes
    tc_mem_bytes = tc_mem_elems * dtype_bytes

    return {
        "baseline_flops": int(baseline_flops),
        "tc_flops": int(tc_flops),
        "flop_reduction_pct": flop_reduction_pct,
        "baseline_mem_bytes": int(baseline_mem_bytes),
        "tc_mem_bytes": int(tc_mem_bytes),
        "mem_reduction_pct": mem_reduction_pct
    }

def plot_flops_gflops(results, output_name="flops_gflops_comparison.png"):
    """
    Bar plot:
      X-axis: Matrix file names
      Y-axis: GFLOPs
      Bars  : Baseline FLOPs vs TC-SpMM FLOPs
    """

    filenames = []
    baseline_gflops = []
    tc_gflops = []

    for r in results:
        if "perf_model" not in r:
            continue

        fname = os.path.basename(r["path"])
        perf = r["perf_model"]

        filenames.append(fname)
        baseline_gflops.append(perf["baseline_flops"] / 1e9)
        tc_gflops.append(perf["tc_flops"] / 1e9)

    x = np.arange(len(filenames))
    width = 0.35

    plt.figure(figsize=(14, 6))
    plt.bar(x - width/2, baseline_gflops, width, label="Baseline FLOPs (GFLOPs)")
    plt.bar(x + width/2, tc_gflops, width, label="TC-SpMM FLOPs (GFLOPs)")

    plt.xticks(x, filenames, rotation=30, ha="right")
    plt.ylabel("GFLOPs")
    plt.xlabel("Matrix File")
    plt.title("Baseline vs TC-SpMM FLOPs (GigaFLOPs)")
    plt.legend()
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(output_name, dpi=200)
    plt.close()

    print(f"✅ FLOPs plot saved as: {output_name}")
def plot_memory_gb(results, output_name="memory_gb_comparison.png"):
    """
    Bar plot:
      X-axis: Matrix file names
      Y-axis: Memory Access in GB
      Bars  : Baseline Memory vs TC-SpMM Memory
    """

    filenames = []
    baseline_gb = []
    tc_gb = []

    for r in results:
        if "perf_model" not in r:
            continue

        fname = os.path.basename(r["path"])
        perf = r["perf_model"]

        filenames.append(fname)
        baseline_gb.append(perf["baseline_mem_bytes"] / 1e9)
        tc_gb.append(perf["tc_mem_bytes"] / 1e9)

    x = np.arange(len(filenames))
    width = 0.35

    plt.figure(figsize=(14, 6))
    plt.bar(x - width/2, baseline_gb, width, label="Baseline Memory (GB)")
    plt.bar(x + width/2, tc_gb, width, label="TC-SpMM Memory (GB)")

    plt.xticks(x, filenames, rotation=30, ha="right")
    plt.ylabel("Memory Access (GB)")
    plt.xlabel("Matrix File")
    plt.title("Baseline vs TC-SpMM Memory Access (GB)")
    plt.legend()
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(output_name, dpi=200)
    plt.close()

    print(f"✅ Memory plot saved as: {output_name}")
def plot_flop_reduction_pct(results, output_name="flop_reduction_percentage.png"):
    """
    Bar plot:
      X-axis: Matrix file names
      Y-axis: FLOP Reduction Percentage (%)
      Bars  : TC-SpMM FLOP reduction vs baseline
      Text  : Percentage value on each bar
    """

    filenames = []
    flop_reduction = []

    for r in results:
        if "perf_model" not in r:
            continue

        fname = os.path.basename(r["path"])
        perf = r["perf_model"]

        filenames.append(fname)
        flop_reduction.append(perf["flop_reduction_pct"])

    x = np.arange(len(filenames))

    plt.figure(figsize=(14, 6))
    bars = plt.bar(x, flop_reduction)

    plt.xticks(x, filenames, rotation=30, ha="right")
    plt.ylabel("FLOP Reduction (%)")
    plt.xlabel("Matrix File")
    plt.title("TC-SpMM FLOP Reduction vs Baseline (%)")
    plt.grid(True, axis="y")

    # ✅ Write percentage on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(output_name, dpi=200)
    plt.close()

    print(f"✅ FLOP reduction plot saved as: {output_name}")
def plot_memory_reduction_pct(results, output_name="memory_reduction_percentage.png"):
    """
    Bar plot:
      X-axis: Matrix file names
      Y-axis: Memory Reduction Percentage (%)
      Bars  : TC-SpMM memory reduction vs baseline
      Text  : Percentage value on each bar
    """

    filenames = []
    mem_reduction = []

    for r in results:
        if "perf_model" not in r:
            continue

        fname = os.path.basename(r["path"])
        perf = r["perf_model"]

        filenames.append(fname)
        mem_reduction.append(perf["mem_reduction_pct"])

    x = np.arange(len(filenames))

    plt.figure(figsize=(14, 6))
    bars = plt.bar(x, mem_reduction)

    plt.xticks(x, filenames, rotation=30, ha="right")
    plt.ylabel("Memory Reduction (%)")
    plt.xlabel("Matrix File")
    plt.title("TC-SpMM Memory Reduction vs Baseline (%)")
    plt.grid(True, axis="y")

    # ✅ Write percentage on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(output_name, dpi=200)
    plt.close()

    print(f"✅ Memory reduction plot saved as: {output_name}")


def process_matrix_file(path, tile_size=TILE):
    print(f"\nProcessing file: {path}")
    mat = sio.mmread(path)  # may return sparse or dense
    # Ensure we have a sparse COO representation as read
    if isinstance(mat, np.ndarray):
        # mmread returned dense; convert to sparse COO
        mat_coo = coo_matrix(mat)
    else:
        # convert to COO explicitly
        try:
            mat_coo = mat.tocoo(copy=False)
        except Exception:
            mat_coo = coo_matrix(mat)

    # Print COO metadata & sample
    print_coo_info(mat_coo, label="ORIGINAL (COO)")

    # Convert to CSR and print metadata
    mat_csr = mat_coo.tocsr()
    print_csr_info(mat_csr, label="CONVERTED (CSR)")

    # Compute approximate storage sizes (bytes)
    coo_bytes = approx_sparse_bytes_coo(mat_coo)
    csr_bytes = approx_sparse_bytes_csr(mat_csr)
    print(f"Approx storage (bytes): COO: {coo_bytes}, CSR: {csr_bytes}\n")

    # Tile-based method: convert to dense first (careful about memory)
    try:
        dense = mat_coo.toarray()
    except MemoryError:
        print("Skipping tile-based compression: matrix too large to convert to dense (MemoryError).")
        return {
            'path': path,
            'coo_bytes': coo_bytes,
            'csr_bytes': csr_bytes,
            'tile_bytes': None,
            'tile_stats': None
        }

    # Tile and compress
    tiles_info, unique_tiles, tile_stats = tile_compress_dense(dense, tile_size=tile_size)
    tile_bytes, tile_breakdown = approx_tile_storage_bytes(tiles_info, unique_tiles)
    # ✅ Analyze tile duplicates & spatial patterns
    dup_stats = analyze_tile_duplicates(dense, tile_size=tile_size)

    # ✅ Theoretical FLOP & Memory Model
    perf_model = theoretical_flop_memory_model(
        tile_stats=tile_stats,
        dup_stats=dup_stats,
        matrix_shape=dense.shape,
        tile_size=tile_size,
        dtype_bytes=8  # float64
    )
    # Summaries
    print("Tile-based compression summary:")
    print(f"  Matrix tile grid        : {tile_stats['tile_rows']} x {tile_stats['tile_cols']}")
    print(f"  ✅ TOTAL tiles           : {tile_stats['num_tiles_total']}")
    print(f"  ✅ Zero tiles (skipped)  : {tile_stats['num_zero_tiles']}")
    print(f"  ✅ Nonzero tiles used   : {tile_stats['num_nonzero_tiles']}")
    print(f"  ✅ Unique nonzero tiles : {tile_stats['num_unique_tiles']}")
    print(f"  Tile size parameter     : {tile_stats['tile_size']}")
    print(f"  Approx storage (bytes)  : {tile_bytes}")
    print(f"    Breakdown → positions {tile_breakdown['positions_nbytes']} | pointers {tile_breakdown['ptrs_nbytes']} | unique tiles data {tile_breakdown['unique_tiles_nbytes']}")
    print()
    print("Tile duplicate spatial analysis:")
    print(f"  Total tiles         : {dup_stats['total_tiles']}")
    print(f"  Zero tiles          : {dup_stats['zero_tiles']}")
    print(f"  Unique nonzero tiles: {dup_stats['unique_tiles']}")
    print(f"  Duplicate tiles     : {dup_stats['duplicate_tiles']}")
    print(f"    Same tile row     : {dup_stats['duplicates_same_row']}")
    print(f"    Same tile column  : {dup_stats['duplicates_same_col']}")
    print(f"    Elsewhere         : {dup_stats['duplicates_elsewhere']}")
    print()
    print("=== Theoretical Performance Model ===")
    print(f"Baseline FLOPs        : {perf_model['baseline_flops']:,}")
    print(f"TC-SpMM FLOPs         : {perf_model['tc_flops']:,}")
    print(f"✅ FLOP Reduction     : {perf_model['flop_reduction_pct']:.2f} %")
    print()
    print(f"Baseline Memory (B)   : {perf_model['baseline_mem_bytes']:,}")
    print(f"TC-SpMM Memory (B)    : {perf_model['tc_mem_bytes']:,}")
    print(f"✅ Memory Reduction   : {perf_model['mem_reduction_pct']:.2f} %")
    print()

    # # Optionally: print a small example of tile entries and unique tiles
    # print("Example tile entries (first 12):")
    # for i, t in enumerate(tiles_info[:12]):
    #     pos = t['tile_pos']
    #     if t['stored_as'] == 'ptr':
    #         print(f"  tile {pos} -> PTR to {t['ptr_to']}")
    #     else:
    #         key = t['data_key']
    #         coo = unique_tiles[key]['coo']
    #         print(f"  tile {pos} -> STORED DATA, uniq tile nnz={coo.nnz}, shape={coo.shape}")
    # print()

    return {
        'path': path,
        'coo_bytes': coo_bytes,
        'csr_bytes': csr_bytes,
        'tile_bytes': tile_bytes,
        'tile_stats': tile_stats,
        'num_unique_tiles': tile_stats['num_unique_tiles'],
        'num_tiles': tile_stats['num_tiles_total'],
        'perf_model': perf_model
    }



def main():
    mt_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.mtx")))
    if not mt_files:
        print(f"No .mtx files found in folder '{DATA_DIR}'. Place them there and retry.")
        return

    results = []
    for f in mt_files:
        try:
            res = process_matrix_file(f, tile_size=TILE)
            results.append(res)
        except Exception as e:
            print(f"ERROR processing '{f}': {e}")

    print("\n=== SUMMARY ===")
    for r in results:
        path = r['path']
        coo_b = r['coo_bytes']
        csr_b = r['csr_bytes']
        tb = r['tile_bytes']
        print(f"File: {os.path.basename(path)}")
        print(f"  COO bytes: {coo_b}")
        print(f"  CSR bytes: {csr_b}")
        if tb is not None:
            print(f"  Tile-based bytes: {tb} (unique tiles: {r['num_unique_tiles']}, total tiles: {r['num_tiles']})")
            # Compression ratios
            try:
                ratio_to_coo = tb / coo_b if coo_b > 0 else float('inf')
                ratio_to_csr = tb / csr_b if csr_b > 0 else float('inf')
                print(f"  Tile/COO ratio: {ratio_to_coo:.4f}, Tile/CSR ratio: {ratio_to_csr:.4f}")
            except Exception:
                pass
        else:
            print("  Tile-based: skipped (too large to densify)")
        print()
    # plot_compression_ratios(results)
    # plot_flops_gflops(results)
    # plot_memory_gb(results)
    plot_flop_reduction_pct(results)
    plot_memory_reduction_pct(results)

if __name__ == "__main__":
    main()
