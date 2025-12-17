// spmm_bcsr_baseline_fixed.cu
// Compile: nvcc -std=c++14 -O3 spmm_bcsr_baseline_fixed.cu -lcudart -o spmm_bcsr_baseline_fixed
// has bug doiesnt not work anyother value but only fdor BS=4
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <stdint.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                            \
    cudaError_t e = (call);                                              \
    if (e != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                              \
    } } while(0)

static inline void cudaCheckKernel(const char* where) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "Kernel launch error at %s: %s\n", where, cudaGetErrorString(e));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

// ---------- Robust MTX reader ----------
void read_mtx_coo_robust(const char* filename, int *outM, int *outN, int *outNnz,
                         int **row, int **col, float **val)
{
    FILE *f = fopen(filename, "r");
    if (!f) { fprintf(stderr, "Error: cannot open %s\n", filename); exit(EXIT_FAILURE); }

    char line[1024];
    // skip comments and blank lines until header
    do {
        if (!fgets(line, sizeof(line), f)) { fprintf(stderr, "Error: empty or invalid file %s\n", filename); exit(EXIT_FAILURE); }
        char *p = line;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '%' || *p == '\0') continue;
        break;
    } while (1);

    int M=0, N=0, header_nnz=0;
    if (sscanf(line, "%d %d %d", &M, &N, &header_nnz) < 2) {
        fprintf(stderr, "Error: malformed header line: %s\n", line); exit(EXIT_FAILURE);
    }
    if (header_nnz < 0) header_nnz = 0;

    int cap = (header_nnz > 0) ? header_nnz : 1024;
    int count = 0;
    int *r = (int*) malloc(cap * sizeof(int));
    int *c = (int*) malloc(cap * sizeof(int));
    float *v = (float*) malloc(cap * sizeof(float));
    if (!r || !c || !v) { fprintf(stderr, "malloc failed\n"); exit(EXIT_FAILURE); }

    int line_no = 1;
    while (fgets(line, sizeof(line), f)) {
        line_no++;
        char *p = line;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '\0' || *p == '%') continue;
        int ri=-1, ci=-1; double vd=0.0;
        int nitems = sscanf(p, "%d %d %lf", &ri, &ci, &vd);
        if (nitems < 2) {
            fprintf(stderr, "Warning: unparsable line %d: '%s'\n", line_no, p);
            continue;
        }
        ri -= 1; ci -= 1; // to 0-based
        float vf = (nitems >= 3) ? (float)vd : 1.0f; // pattern -> 1.0
        if (count >= cap) {
            int newcap = cap * 2;
            r = (int*) realloc(r, newcap * sizeof(int));
            c = (int*) realloc(c, newcap * sizeof(int));
            v = (float*) realloc(v, newcap * sizeof(float));
            if (!r || !c || !v) { fprintf(stderr, "realloc failed\n"); exit(EXIT_FAILURE); }
            cap = newcap;
        }
        r[count] = ri; c[count] = ci; v[count] = vf; count++;
    }
    fclose(f);

    // shrink to fit
    r = (int*) realloc(r, count * sizeof(int));
    c = (int*) realloc(c, count * sizeof(int));
    v = (float*) realloc(v, count * sizeof(float));

    *outM = M; *outN = N; *outNnz = count;
    *row = r; *col = c; *val = v;
    if (header_nnz > 0 && header_nnz != count) {
        fprintf(stderr, "Note: header nnz=%d but read %d entries. Using %d.\n", header_nnz, count, count);
    }
}

// ---------- BCSR builder ----------
static inline uint64_t key_of(int br, int bc) {
    return ( (uint64_t)(uint32_t)br << 32 ) | (uint32_t)bc;
}

struct BCSR {
    int M, N;
    int BS;
    int Mb, Nb;
    int nnzb;
    std::vector<int> browptr;
    std::vector<int> bcolidx;
    std::vector<float> bval;
};

BCSR build_bcsr_from_coo(int M, int N, int nnz, const int* cooRow, const int* cooCol, const float* cooVal, int BS)
{
    BCSR out;
    out.M = M; out.N = N; out.BS = BS;
    out.Mb = (M + BS - 1) / BS;
    out.Nb = (N + BS - 1) / BS;

    std::unordered_map<uint64_t,int> blkMap;
    blkMap.reserve(nnz * 2 + 16);

    std::vector<int> blk_br; blk_br.reserve(nnz);
    std::vector<int> blk_bc; blk_bc.reserve(nnz);
    std::vector<std::vector<float>> blocks; blocks.reserve(nnz);

    for (int idx = 0; idx < nnz; ++idx) {
        int r = cooRow[idx];
        int c = cooCol[idx];
        float v = cooVal[idx];
        if (r < 0 || r >= M || c < 0 || c >= N) continue;
        int br = r / BS;
        int bc = c / BS;
        int intra_r = r % BS;
        int intra_c = c % BS;
        uint64_t k = key_of(br, bc);
        auto it = blkMap.find(k);
        if (it == blkMap.end()) {
            int id = (int)blocks.size();
            blkMap[k] = id;
            blk_br.push_back(br);
            blk_bc.push_back(bc);
            blocks.emplace_back(std::vector<float>((size_t)BS * BS, 0.0f));
            blocks[id][intra_r * BS + intra_c] = v;
        } else {
            int id = it->second;
            blocks[id][intra_r * BS + intra_c] += v;
        }
    }

    int nnzb = (int)blocks.size();
    out.nnzb = nnzb;

    std::vector<std::vector<int>> blocks_in_row(out.Mb);
    for (int i = 0; i < nnzb; ++i) {
        int br = blk_br[i];
        blocks_in_row[br].push_back(i);
    }

    out.browptr.assign(out.Mb + 1, 0);
    for (int br = 0; br < out.Mb; ++br) out.browptr[br+1] = out.browptr[br] + (int)blocks_in_row[br].size();

    out.bcolidx.resize(nnzb);
    out.bval.resize((size_t)nnzb * BS * BS);

    int pos = 0;
    for (int br = 0; br < out.Mb; ++br) {
        auto &vec = blocks_in_row[br];
        std::sort(vec.begin(), vec.end(), [&](int a, int b){ return blk_bc[a] < blk_bc[b]; });
        for (int id : vec) {
            out.bcolidx[pos] = blk_bc[id];
            memcpy(&out.bval[(size_t)pos * BS * BS], blocks[id].data(), (size_t)BS * BS * sizeof(float));
            pos++;
        }
    }
    if (pos != nnzb) { fprintf(stderr, "BCSR build inconsistent: pos=%d nnzb=%d\n", pos, nnzb); exit(1); }
    return out;
}

// ---------- CPU reference ----------
void cpu_bcsr_spmm(int M, int N, const BCSR &A, const float *B, int K, float *Y)
{
    int BS = A.BS;
    for (long long i = 0; i < (long long)M * K; ++i) Y[i] = 0.0f;

    for (int br = 0; br < A.Mb; ++br) {
        int row_base = br * BS;
        int brow_start = A.browptr[br];
        int brow_end   = A.browptr[br+1];
        for (int bi = brow_start; bi < brow_end; ++bi) {
            int bc = A.bcolidx[bi];
            int col_base = bc * BS;
            const float *blk = &A.bval[(size_t)bi * BS * BS];
            for (int i = 0; i < BS; ++i) {
                int global_r = row_base + i;
                if (global_r >= M) continue;
                float *yrow = &Y[(size_t)global_r * K];
                for (int kk = 0; kk < K; ++kk) {
                    float s = 0.0f;
                    for (int j = 0; j < BS; ++j) {
                        int global_c = col_base + j;
                        if (global_c >= N) continue;
                        float a = blk[i * BS + j];
                        float b = B[(size_t)global_c * K + kk];
                        s += a * b;
                    }
                    yrow[kk] += s;
                }
            }
        }
    }
}

// ---------- GPU kernel: simple BCSR SpMM ----------
__global__
void bcsr_spmm_kernel(int M, int N, int K, int BS,
                      const int *d_browptr, const int *d_bcolidx, const float *d_bval,
                      const float *d_B, float *d_Y)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int k   = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || k >= K) return;

    int br = row / BS;
    int intra_r = row % BS;

    float acc = 0.0f;
    int brow_start = d_browptr[br];
    int brow_end = d_browptr[br+1];
    for (int bi = brow_start; bi < brow_end; ++bi) {
        int bc = d_bcolidx[bi];
        int col_base = bc * BS;
        const float *blk = &d_bval[(size_t)bi * BS * BS];
        for (int j = 0; j < BS; ++j) {
            int global_c = col_base + j;
            if (global_c >= N) continue;
            float a = blk[intra_r * BS + j];
            float b = d_B[(size_t)global_c * K + k];
            acc += a * b;
        }
    }
    d_Y[(size_t)row * K + k] = acc;
}

// ---------- main ----------
int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("Usage: %s matrix.mtx [iters=10] [K=N] [BS=4]\n", argv[0]);
        return 0;
    }

    const char* matrix_file = argv[1];
    int ITERS = (argc >= 3) ? atoi(argv[2]) : 10;
    int K = (argc >= 4) ? atoi(argv[3]) : 0;
    int BS = (argc >= 5) ? atoi(argv[4]) : 4;
    if (BS <= 0) BS = 4;

    int M, N, nnz;
    int *cooRow = nullptr, *cooCol = nullptr;
    float *cooVal = nullptr;

    read_mtx_coo_robust(matrix_file, &M, &N, &nnz, &cooRow, &cooCol, &cooVal);
    printf("Loaded matrix %s : %d x %d with %d entries\n", matrix_file, M, N, nnz);

    if (K == 0) K = N;
    printf("Running BCSR SpMM : BS=%d, K=%d, iters=%d\n", BS, K, ITERS);

    // sanity: ensure BS reasonable
    if (BS > M && BS > N) {
        fprintf(stderr, "Warning: BS (%d) larger than both M (%d) and N (%d). Falling back to min(M,N)\n", BS, M, N);
        BS = std::min(M, N);
    }

    // Build BCSR
    BCSR A = build_bcsr_from_coo(M, N, nnz, cooRow, cooCol, cooVal, BS);
    printf("BCSR built: Mb=%d Nb=%d nnzb=%d (blocks)\n", A.Mb, A.Nb, A.nnzb);

    // Host dense B and outputs
    std::vector<float> h_B((size_t)N * K);
    for (size_t i = 0; i < h_B.size(); ++i) h_B[i] = 1.0f;

    std::vector<float> h_Y((size_t)M * K);
    std::vector<float> h_Y_ref((size_t)M * K);

    // CPU reference
    cpu_bcsr_spmm(M, N, A, h_B.data(), K, h_Y_ref.data());

    // Device allocations
    int *d_browptr = nullptr, *d_bcolidx = nullptr;
    float *d_bval = nullptr, *d_B = nullptr, *d_Y = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_browptr, (A.Mb + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_bcolidx, (size_t)A.nnzb * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_bval, (size_t)A.nnzb * BS * BS * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, (size_t)N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, (size_t)M * K * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_browptr, A.browptr.data(), (A.Mb + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bcolidx, A.bcolidx.data(), (size_t)A.nnzb * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bval, A.bval.data(), (size_t)A.nnzb * BS * BS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), (size_t)N * K * sizeof(float), cudaMemcpyHostToDevice));

    // launch config
    dim3 block(16, 16); // x->k, y->row
    int gridx = (K + block.x - 1) / block.x;
    int gridy = (M + block.y - 1) / block.y;
    dim3 grid(gridx, gridy);

    // warm-up (we'll zero d_Y before warmup and before each timed iteration)
    CUDA_CHECK(cudaMemset(d_Y, 0, (size_t)M * K * sizeof(float)));
    for (int w = 0; w < 3; ++w) {
        bcsr_spmm_kernel<<<grid, block>>>(M, N, K, BS, d_browptr, d_bcolidx, d_bval, d_B, d_Y);
        cudaCheckKernel("warmup kernel");
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < ITERS; ++it) {
        // crucial: reset output buffer for each iteration so results don't accumulate
        CUDA_CHECK(cudaMemset(d_Y, 0, (size_t)M * K * sizeof(float)));

        bcsr_spmm_kernel<<<grid, block>>>(M, N, K, BS, d_browptr, d_bcolidx, d_bval, d_B, d_Y);
        cudaCheckKernel("timed kernel");
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float msTotal = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&msTotal, start, stop));
    float avg_ms = msTotal / ITERS;

    // copy back result
    CUDA_CHECK(cudaMemcpy(h_Y.data(), d_Y, (size_t)M * K * sizeof(float), cudaMemcpyDeviceToHost));

    // verify
    int mismatches = 0;
    double max_err = 0.0;
    for (size_t i = 0; i < (size_t)M * K; ++i) {
        float a = h_Y_ref[i];
        float b = h_Y[i];
        float err = fabsf(a - b);
        if (err > 1e-3f) {
            if (mismatches < 10) printf("Mismatch idx %zu: cpu=%f gpu=%f err=%f\n", i, a, b, err);
            mismatches++;
        }
        if (err > max_err) max_err = err;
    }
    printf("Verification: %s (mismatches=%d max_err=%g)\n", (mismatches==0)?"PASSED":"FAILED", mismatches, max_err);

    // metrics
    double flops = 2.0 * (double)nnz * (double)K;
    double gflops = flops / (avg_ms / 1000.0) / 1e9;
    double bytes = (double)A.nnzb * (BS * BS * sizeof(float) + sizeof(int)) + (double)(A.Mb+1)*sizeof(int) + (double)N * K * sizeof(float) + (double)M * K * sizeof(float);
    double gbs = bytes / (avg_ms / 1000.0) / 1e9;
    double mem_bcsr_mb = ((double)A.nnzb * (BS * BS * sizeof(float) + sizeof(int)) + (A.Mb+1)*sizeof(int)) / 1e6;
    double mem_dense_mb = ((double)(N*K + M*K) * sizeof(float)) / 1e6;

    printf("\n===== BCSR Baseline SpMM Metrics =====\n");
    printf("Matrix: %s\n", matrix_file);
    printf("Dimensions: %d x %d\n", M, N);
    printf("Original NNZ: %d\n", nnz);
    printf("BS=%d Mb=%d Nb=%d nnzb=%d\n", BS, A.Mb, A.Nb, A.nnzb);
    printf("K: %d\n", K);
    printf("Avg Time: %.6f ms\n", avg_ms);
    printf("GFLOP/s (approx): %.3f\n", gflops);
    printf("Effective Bandwidth (approx): %.3f GB/s\n", gbs);
    printf("Memory Footprint: BCSR = %.2f MB, Dense matrices = %.2f MB\n", mem_bcsr_mb, mem_dense_mb);
    printf("====================================\n");

    // cleanup
    cudaFree(d_browptr); cudaFree(d_bcolidx); cudaFree(d_bval); cudaFree(d_B); cudaFree(d_Y);
    free(cooRow); free(cooCol); free(cooVal);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
