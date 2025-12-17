// spmm_ell_baseline.cu
// Compile: nvcc -std=c++14 -O3 spmm_ell_baseline.cu -lcudart -o spmm_ell_baseline
// Usage: ./spmm_ell_baseline matrix.mtx [iters=10] [K=N]
// Example: ./spmm_ell_baseline Mat_data/barth.mtx 10 128

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                            \
    cudaError_t e = (call);                                              \
    if (e != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                               \
    } } while(0)

// Robust MatrixMarket COO reader (handles leading whitespace, comments, pattern format)
void read_mtx_coo_robust(const char* filename, int *outM, int *outN, int *outNnz,
                         int **row, int **col, float **val)
{
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error: cannot open %s\n", filename);
        exit(EXIT_FAILURE);
    }

    char line[1024];

    // skip comments and blank lines until header
    do {
        if (!fgets(line, sizeof(line), f)) {
            fprintf(stderr, "Error: empty or invalid file %s\n", filename);
            exit(EXIT_FAILURE);
        }
        char *p = line;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '%' || *p == '\0') continue;
        break;
    } while (1);

    int M=0, N=0, header_nnz=0;
    if (sscanf(line, "%d %d %d", &M, &N, &header_nnz) < 2) {
        fprintf(stderr, "Error: malformed header line: %s\n", line);
        exit(EXIT_FAILURE);
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
        int ri=-1, ci=-1;
        double vd=0.0;
        int nitems = sscanf(p, "%d %d %lf", &ri, &ci, &vd);
        if (nitems < 2) {
            fprintf(stderr, "Warning: unparsable line %d: '%s'\n", line_no, p);
            continue;
        }
        // convert 1-based to 0-based
        ri -= 1; ci -= 1;
        float vf = (nitems >= 3) ? (float)vd : 1.0f;
        if (count >= cap) {
            int newcap = cap * 2;
            r = (int*) realloc(r, newcap * sizeof(int));
            c = (int*) realloc(c, newcap * sizeof(int));
            v = (float*) realloc(v, newcap * sizeof(float));
            if (!r || !c || !v) { fprintf(stderr, "realloc failed\n"); exit(EXIT_FAILURE); }
            cap = newcap;
        }
        r[count] = ri;
        c[count] = ci;
        v[count] = vf;
        count++;
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

// Convert COO -> ELLPACK (row-major storage)
// Outputs:
//  - max_nnz_per_row
//  - ell_col: int[M * max_nnz] (col indices, -1 for padding)
//  - ell_val: float[M * max_nnz] (values, 0.0 for padding)
// returns max_nnz_per_row
int coo_to_ell(int M, int N, int nnz, const int *cooRow, const int *cooCol, const float *cooVal,
               int **ell_col_out, float **ell_val_out)
{
    // compute counts per row
    int *row_counts = (int*) calloc(M, sizeof(int));
    for (int i = 0; i < nnz; ++i) {
        int r = cooRow[i];
        if (r < 0 || r >= M) { fprintf(stderr, "Row out of bounds in COO: %d\n", r); exit(1); }
        row_counts[r]++;
    }
    int max_nnz = 0;
    for (int i = 0; i < M; ++i) if (row_counts[i] > max_nnz) max_nnz = row_counts[i];
    if (max_nnz == 0) max_nnz = 1; // avoid zero

    // allocate ell arrays, initialize with padding values
    int *ell_col = (int*) malloc((size_t)M * max_nnz * sizeof(int));
    float *ell_val = (float*) malloc((size_t)M * max_nnz * sizeof(float));
    if (!ell_col || !ell_val) { fprintf(stderr, "malloc failed for ELL arrays\n"); exit(1); }

    for (int i = 0; i < M * max_nnz; ++i) {
        ell_col[i] = -1; // sentinel for empty
        ell_val[i] = 0.0f;
    }

    // temporary offsets to place items in each row
    int *offset = (int*) calloc(M, sizeof(int));
    for (int idx = 0; idx < nnz; ++idx) {
        int r = cooRow[idx];
        int pos = offset[r]++;
        if (pos >= max_nnz) {
            // should not happen because max_nnz computed from row_counts, but check
            fprintf(stderr, "INTERNAL ERROR: pos >= max_nnz for row %d\n", r);
            exit(1);
        }
        ell_col[r * max_nnz + pos] = cooCol[idx];
        ell_val[r * max_nnz + pos] = cooVal[idx];
    }

    free(offset);
    free(row_counts);

    *ell_col_out = ell_col;
    *ell_val_out = ell_val;
    return max_nnz;
}

// CPU reference (dense B row-major with ld=K)
void cpu_ell_spmm(int M, int N, int max_nnz, const int *ell_col, const float *ell_val,
                  const float *B, int K, float *Y)
{
    for (int i = 0; i < M; ++i) {
        float *yrow = &Y[(size_t)i * K];
        // zero
        for (int kk = 0; kk < K; ++kk) yrow[kk] = 0.0f;
        const int *crow = &ell_col[(size_t)i * max_nnz];
        const float *vrow = &ell_val[(size_t)i * max_nnz];
        for (int j = 0; j < max_nnz; ++j) {
            int c = crow[j];
            if (c < 0) continue;
            float a = vrow[j];
            const float *brow = &B[(size_t)c * K];
            for (int kk = 0; kk < K; ++kk) yrow[kk] += a * brow[kk];
        }
    }
}

// GPU kernel: each thread computes one (row, k) element of Y
__global__
void ell_spmm_kernel(int M, int K, int max_nnz, const int *ell_col, const float *ell_val, const float *B, float *Y, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int k   = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || k >= K) return;

    const int *crow = &ell_col[(size_t)row * max_nnz];
    const float *vrow = &ell_val[(size_t)row * max_nnz];

    float sum = 0.0f;
    // iterate nonzeros in this ELL row
    for (int j = 0; j < max_nnz; ++j) {
        int col = crow[j];
        if (col < 0) continue;
        float aval = vrow[j];
        // B is stored row-major: B[col * K + k]
        float bval = B[(size_t)col * K + k];
        sum += aval * bval;
    }
    Y[(size_t)row * K + k] = sum;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("Usage: %s matrix.mtx [iters=10] [K=N]\n", argv[0]);
        return 0;
    }

    const char* matrix_file = argv[1];
    int ITERS = 10;
    if (argc >= 3) ITERS = atoi(argv[2]);

    int M, N, nnz;
    int *cooRow=NULL, *cooCol=NULL;
    float *cooVal=NULL;
    read_mtx_coo_robust(matrix_file, &M, &N, &nnz, &cooRow, &cooCol, &cooVal);
    printf("Loaded matrix %s : %d x %d with %d entries\n", matrix_file, M, N, nnz);

    int K = N;
    if (argc >= 4) K = atoi(argv[3]);
    printf("Running ELL SpMM with K=%d, iterations=%d\n", K, ITERS);

    // Convert COO -> ELL
    int *ell_col = NULL;
    float *ell_val = NULL;
    int max_nnz = coo_to_ell(M, N, nnz, cooRow, cooCol, cooVal, &ell_col, &ell_val);
    printf("ELL format: max_nnz_per_row = %d, ELL arrays size = %lld entries\n", max_nnz, (long long)M * max_nnz);

    // host dense B and outputs
    float *h_B = (float*) malloc((size_t)N * K * sizeof(float));
    for (long long i = 0; i < (long long)N * K; ++i) h_B[i] = 1.0f; // simple test pattern

    float *h_Y = (float*) malloc((size_t)M * K * sizeof(float));
    float *h_Y_ref = (float*) malloc((size_t)M * K * sizeof(float));

    // CPU reference
    cpu_ell_spmm(M, N, max_nnz, ell_col, ell_val, h_B, K, h_Y_ref);

    // device allocations
    int *d_ell_col = NULL;
    float *d_ell_val = NULL, *d_B = NULL, *d_Y = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_ell_col, (size_t)M * max_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_ell_val, (size_t)M * max_nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, (size_t)N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, (size_t)M * K * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_ell_col, ell_col, (size_t)M * max_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ell_val, ell_val, (size_t)M * max_nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, (size_t)N * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_Y, 0, (size_t)M * K * sizeof(float)));

    // kernel launch configuration
    dim3 block(16, 16); // x -> k, y -> row
    int gridx = (K + block.x - 1) / block.x;
    int gridy = (M + block.y - 1) / block.y;
    dim3 grid(gridx, gridy);

    // warm-up
    for (int i = 0; i < 3; ++i) {
        ell_spmm_kernel<<<grid, block>>>(M, K, max_nnz, d_ell_col, d_ell_val, d_B, d_Y, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < ITERS; ++it) {
        ell_spmm_kernel<<<grid, block>>>(M, K, max_nnz, d_ell_col, d_ell_val, d_B, d_Y, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float msTotal = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msTotal, start, stop));
    float avg_ms = msTotal / ITERS;

    // copy back
    CUDA_CHECK(cudaMemcpy(h_Y, d_Y, (size_t)M * K * sizeof(float), cudaMemcpyDeviceToHost));

    // verify
    int mismatches = 0;
    double max_abs_err = 0.0;
    for (long long i = 0; i < (long long)M * K; ++i) {
        float a = h_Y_ref[i];
        float b = h_Y[i];
        float err = fabsf(a - b);
        if (err > 1e-3f) {
            if (mismatches < 10) printf("Mismatch idx %lld: cpu=%f gpu=%f err=%f\n", i, a, b, err);
            mismatches++;
        }
        if (err > max_abs_err) max_abs_err = err;
    }
    if (mismatches == 0) printf("Verification: PASSED (max err=%.6f)\n", max_abs_err);
    else printf("Verification: FAILED (%d mismatches, max err=%.6f)\n", mismatches, max_abs_err);

    // metrics (FLOPs = 2 * nnz * K)
    double flops = 2.0 * (double)nnz * (double)K;
    double gflops = flops / (avg_ms / 1000.0) / 1e9;
    double bytes = (double)M * max_nnz * (sizeof(int) + sizeof(float)) + (double)N * K * sizeof(float) + (double)M * K * sizeof(float);
    double gbs = bytes / (avg_ms / 1000.0) / 1e9;
    double mem_ell_mb = ((double)M * max_nnz * (sizeof(int) + sizeof(float))) / 1e6;
    double mem_dense_mb = ((double)(N*K + M*K) * sizeof(float)) / 1e6;

    printf("\n===== ELL Baseline SpMM Metrics =====\n");
    printf("Matrix: %s\n", matrix_file);
    printf("Dimensions: %d x %d\n", M, N);
    printf("NNZ: %d\n", nnz);
    printf("max_nnz_per_row: %d\n", max_nnz);
    printf("K: %d\n", K);
    printf("Avg Time: %.6f ms\n", avg_ms);
    printf("Performance: %.3f GFLOP/s\n", gflops);
    printf("Effective Bandwidth: %.3f GB/s\n", gbs);
    printf("Memory Footprint: ELL = %.2f MB, Dense matrices = %.2f MB\n", mem_ell_mb, mem_dense_mb);
    printf("====================================\n");

    // cleanup
    cudaFree(d_ell_col);
    cudaFree(d_ell_val);
    cudaFree(d_B);
    cudaFree(d_Y);

    free(cooRow);
    free(cooCol);
    free(cooVal);
    free(ell_col);
    free(ell_val);
    free(h_B);
    free(h_Y);
    free(h_Y_ref);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
