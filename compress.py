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
DATA_DIR = "Mat_data"   # Mat_data CU_sparse
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
        'num_tiles': tile_stats['num_tiles_total']
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
    plot_compression_ratios(results)

if __name__ == "__main__":
    main()
