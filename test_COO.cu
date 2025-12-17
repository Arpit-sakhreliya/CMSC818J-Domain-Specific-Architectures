// spmm_coo_baseline.cu
// Compile: nvcc -std=c++14 -O3 spmm_coo_baseline.cu -lcudart -o spmm_coo_baseline
// Usage: ./spmm_coo_baseline matrix.mtx [iters=10] [K=N]
// Example: ./spmm_coo_baseline Mat_data/barth.mtx 20 128

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
        exit(EXIT_FAILURE);                                              \
    } } while(0)

// Robust MatrixMarket COO reader (handles comments, pattern, mismatched header nnz)
void read_mtx_coo_robust(const char* filename, int *outM, int *outN, int *outNnz,
                         int **row, int **col, float **val)
{
    FILE *f = fopen(filename, "r");
    if (!f) { fprintf(stderr, "Error: cannot open %s\n", filename); exit(EXIT_FAILURE); }

    char line[1024];

    // skip comments / blank lines until header
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

    r = (int*) realloc(r, count * sizeof(int));
    c = (int*) realloc(c, count * sizeof(int));
    v = (float*) realloc(v, count * sizeof(float));

    *outM = M; *outN = N; *outNnz = count;
    *row = r; *col = c; *val = v;
    if (header_nnz > 0 && header_nnz != count) {
        fprintf(stderr, "Note: header nnz=%d but read %d entries. Using %d.\n", header_nnz, count, count);
    }
}

// CPU reference for COO SpMM (Y = A * B)
void cpu_coo_spmm(int M, int N, int nnz, const int *cooRow, const int *cooCol, const float *cooVal,
                  const float *B, int K, float *Y)
{
    // zero Y
    for (long long i = 0; i < (long long)M * K; ++i) Y[i] = 0.0f;
    for (int idx = 0; idx < nnz; ++idx) {
        int r = cooRow[idx];
        int c = cooCol[idx];
        float a = cooVal[idx];
        const float *brow = &B[(long long)c * K];
        float *yrow = &Y[(long long)r * K];
        for (int kk = 0; kk < K; ++kk) {
            yrow[kk] += a * brow[kk];
        }
    }
}

// GPU COO SpMM kernel (each thread processes one nonzero entry and atomically updates Y row)
__global__
void coo_spmm_atomic_kernel(int nnz, int K, const int *cooRow, const int *cooCol, const float *cooVal,
                            const float *B, float *Y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;
    int r = cooRow[idx];
    int c = cooCol[idx];
    float a = cooVal[idx];

    const float *brow = &B[(long long)c * K];
    float *yrow = &Y[(long long)r * K];

    // multiply the scalar 'a' with the entire B row and atomic-add into Y row
    for (int kk = 0; kk < K; ++kk) {
        float prod = a * brow[kk];
        // atomic add to avoid race between multiple nonzeros writing same yrow[kk]
        atomicAdd(&yrow[kk], prod);
    }
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("Usage: %s matrix.mtx [iters=10] [K=N]\n", argv[0]);
        return 0;
    }

    const char *matrix_file = argv[1];
    int ITERS = 10;
    if (argc >= 3) ITERS = atoi(argv[2]);

    int M, N, nnz;
    int *h_cooRow = NULL, *h_cooCol = NULL;
    float *h_cooVal = NULL;

    read_mtx_coo_robust(matrix_file, &M, &N, &nnz, &h_cooRow, &h_cooCol, &h_cooVal);
    printf("Loaded matrix %s : %d x %d with %d entries\n", matrix_file, M, N, nnz);

    int K = N;
    if (argc >= 4) K = atoi(argv[3]);
    printf("Running COO SpMM with K=%d, iters=%d\n", K, ITERS);

    // host dense B and outputs
    float *h_B = (float*) malloc((size_t)N * K * sizeof(float));
    for (long long i = 0; i < (long long)N * K; ++i) h_B[i] = 1.0f;

    float *h_Y = (float*) malloc((size_t)M * K * sizeof(float));
    float *h_Y_ref = (float*) malloc((size_t)M * K * sizeof(float));

    // CPU reference (single-threaded)
    cpu_coo_spmm(M, N, nnz, h_cooRow, h_cooCol, h_cooVal, h_B, K, h_Y_ref);

    // device allocations
    int *d_cooRow = NULL, *d_cooCol = NULL;
    float *d_cooVal = NULL, *d_B = NULL, *d_Y = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_cooRow, (size_t)nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_cooCol, (size_t)nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_cooVal, (size_t)nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, (size_t)N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, (size_t)M * K * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_cooRow, h_cooRow, (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cooCol, h_cooCol, (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cooVal, h_cooVal, (size_t)nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, (size_t)N * K * sizeof(float), cudaMemcpyHostToDevice));

    // warm-up
    int threads = 256;
    int blocks = (nnz + threads - 1) / threads;
    CUDA_CHECK(cudaMemset(d_Y, 0, (size_t)M * K * sizeof(float)));
    for (int w = 0; w < 2; ++w) {
        coo_spmm_atomic_kernel<<<blocks, threads>>>(nnz, K, d_cooRow, d_cooCol, d_cooVal, d_B, d_Y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < ITERS; ++it) {
        CUDA_CHECK(cudaMemset(d_Y, 0, (size_t)M * K * sizeof(float)));
        coo_spmm_atomic_kernel<<<blocks, threads>>>(nnz, K, d_cooRow, d_cooCol, d_cooVal, d_B, d_Y);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float msTotal = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msTotal, start, stop));
    float avg_ms = msTotal / ITERS;

    // copy result back
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

    // metrics
    double flops = 2.0 * (double)nnz * (double)K;
    double gflops = flops / (avg_ms / 1000.0) / 1e9;
    double bytes = (double)nnz * (sizeof(int) + sizeof(float)) + (double)N * K * sizeof(float) + (double)M * K * sizeof(float);
    double gbs = bytes / (avg_ms / 1000.0) / 1e9;
    double mem_coo_mb = ((double)nnz * (sizeof(int) + sizeof(float))) / 1e6;
    double mem_dense_mb = ((double)(N*K + M*K) * sizeof(float)) / 1e6;

    printf("\n===== COO Baseline SpMM Metrics =====\n");
    printf("Matrix: %s\n", matrix_file);
    printf("Dimensions: %d x %d\n", M, N);
    printf("NNZ: %d\n", nnz);
    printf("K: %d\n", K);
    printf("Avg Time: %.6f ms\n", avg_ms);
    printf("Performance: %.3f GFLOP/s\n", gflops);
    printf("Effective Bandwidth: %.3f GB/s\n", gbs);
    printf("Memory Footprint: COO = %.2f MB, Dense matrices = %.2f MB\n", mem_coo_mb, mem_dense_mb);
    printf("====================================\n");

    // cleanup
    cudaFree(d_cooRow);
    cudaFree(d_cooCol);
    cudaFree(d_cooVal);
    cudaFree(d_B);
    cudaFree(d_Y);

    free(h_cooRow);
    free(h_cooCol);
    free(h_cooVal);
    free(h_B);
    free(h_Y);
    free(h_Y_ref);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
