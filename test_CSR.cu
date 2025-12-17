// spmm_csr_baseline.cu
// Compile: nvcc -O3 spmm_csr_baseline.cu -lcusparse -lcudart -o spmm_csr_baseline
// Usage: ./spmm_csr_baseline matrix.mtx [iters=10] [K=N] [csv_out=stdout]
// Example: ./csr Mat_data/barth.mtx
// nvcc csr.cu -lcusparse -o csr

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <cuda_runtime.h>
#include <algorithm> 
#include <cusparse.h>

#define CUDA_CHECK(call) do {                                            \
    cudaError_t e = (call);                                              \
    if (e != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                               \
    } } while(0)

#define CUSPARSE_CHECK(call) do {                                         \
    cusparseStatus_t s = (call);                                          \
    if (s != CUSPARSE_STATUS_SUCCESS) {                                   \
        fprintf(stderr, "cuSPARSE error %s:%d: %d\n", __FILE__, __LINE__, (int)s); \
        exit(EXIT_FAILURE);                                               \
    } } while(0)

// Robust MatrixMarket COO reader (handles leading whitespace, comments, pattern format, inaccurate header nnz)
void read_mtx_coo_robust(const char* filename, int *outM, int *outN, int *outNnz,
                         int **row, int **col, float **val)
{
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error: cannot open %s\n", filename);
        exit(EXIT_FAILURE);
    }

    char line[1024];

    // skip comments and blank lines until first non-comment header
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
        // 1-based to 0-based
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

    r = (int*) realloc(r, count * sizeof(int));
    c = (int*) realloc(c, count * sizeof(int));
    v = (float*) realloc(v, count * sizeof(float));

    *outM = M; *outN = N; *outNnz = count;
    *row = r; *col = c; *val = v;
    if (header_nnz > 0 && header_nnz != count) {
        fprintf(stderr, "Note: header nnz=%d but read %d entries. Using %d.\n", header_nnz, count, count);
    }
}

// Simple CPU reference for Y = A (CSR MxN) * B (N x K) where B stored row-major with ld=K
void cpu_csr_spmm(int M, int N, int nnz, const int *csrRow, const int *csrCol, const float *csrVal,
                  const float *B, int K, float *Y)
{
    for (int i = 0; i < M; ++i) {
        int rstart = csrRow[i];
        int rend = csrRow[i+1];
        for (int kk = 0; kk < K; ++kk) Y[i*K + kk] = 0.0f;
        for (int idx = rstart; idx < rend; ++idx) {
            int c = csrCol[idx];
            float aval = csrVal[idx];
            const float *brow = &B[c*K];
            float *yrow = &Y[i*K];
            for (int kk = 0; kk < K; ++kk) {
                yrow[kk] += aval * brow[kk];
            }
        }
    }
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("Usage: %s matrix.mtx [iters=10] [K=N] [csv_out=stdout]\n", argv[0]);
        return 0;
    }

    const char* matrix_file = argv[1];
    int ITERS = 10;
    if (argc >= 3) ITERS = atoi(argv[2]);
    int M, N, nnz;
    int *h_cooRow = NULL, *h_cooCol = NULL;
    float *h_cooVal = NULL;

    read_mtx_coo_robust(matrix_file, &M, &N, &nnz, &h_cooRow, &h_cooCol, &h_cooVal);
    printf("Loaded matrix %s : %d x %d with %d entries\n", matrix_file, M, N, nnz);

    // Convert COO -> CSR
    int *h_csrRow = (int*) calloc((M + 1), sizeof(int));
    for (int i = 0; i < nnz; ++i) {
        int r = h_cooRow[i];
        if (r < 0 || r >= M) { fprintf(stderr, "Row index out of range: %d\n", r); exit(1); }
        h_csrRow[r + 1]++;
    }
    for (int i = 0; i < M; ++i) h_csrRow[i + 1] += h_csrRow[i];

    int *tempRow = (int*) malloc((M + 1) * sizeof(int));
    memcpy(tempRow, h_csrRow, (M + 1) * sizeof(int));

    int *h_csrCol = (int*) malloc((size_t)nnz * sizeof(int));
    float *h_csrVal = (float*) malloc((size_t)nnz * sizeof(float));
    for (int i = 0; i < nnz; ++i) {
        int r = h_cooRow[i];
        int dest = tempRow[r]++;
        h_csrCol[dest] = h_cooCol[i];
        h_csrVal[dest] = h_cooVal[i];
    }
    free(tempRow);

    int K = N;
    if (argc >= 4) K = atoi(argv[3]);
    
    printf("Using dense B: %d x %d (K=%d), iterations=%d\n", N, K, K, ITERS);

    // Host dense B (N x K) and host Y buffers
    float *h_B = (float*) malloc((size_t)N * K * sizeof(float));
    for (long long i = 0; i < (long long)N * K; ++i) h_B[i] = 1.0f;

    float *h_Y = (float*) malloc((size_t)M * K * sizeof(float));
    float *h_Y_ref = (float*) malloc((size_t)M * K * sizeof(float));

    // Device memory
    int *d_csrRow = NULL, *d_csrCol = NULL;
    float *d_csrVal = NULL, *d_B = NULL, *d_Y = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_csrRow, (M + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_csrCol, (size_t)nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_csrVal, (size_t)nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, (size_t)N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, (size_t)M * K * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_csrRow, h_csrRow, (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrCol, h_csrCol, (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrVal, h_csrVal, (size_t)nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, (size_t)N * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_Y, 0, (size_t)M * K * sizeof(float)));

    // cuSPARSE setup
    cusparseHandle_t handle = NULL;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    cusparseSpMatDescr_t matA = NULL;
    cusparseDnMatDescr_t matB = NULL, matY = NULL;

    CUSPARSE_CHECK(cusparseCreateCsr(&matA, M, N, nnz,
                                    d_csrRow, d_csrCol, d_csrVal,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnMat(&matB, N, K, K, d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CUSPARSE_CHECK(cusparseCreateDnMat(&matY, M, K, K, d_Y, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    const float alpha = 1.0f, beta = 0.0f;
    size_t bufferSize = 0;
    void *dBuffer = NULL;

    CUSPARSE_CHECK(cusparseSpMM_bufferSize(handle,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha, matA, matB, &beta, matY,
                                          CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
    if (bufferSize > 0) CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

    // warm-up
    for (int i = 0; i < 3; ++i) {
        CUSPARSE_CHECK(cusparseSpMM(handle,
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &alpha, matA, matB, &beta, matY,
                                   CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < ITERS; ++i) {
        CUSPARSE_CHECK(cusparseSpMM(handle,
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &alpha, matA, matB, &beta, matY,
                                   CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float msTotal = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msTotal, start, stop));
    float avg_ms = msTotal / ITERS;

    // copy result back
    CUDA_CHECK(cudaMemcpy(h_Y, d_Y, (size_t)M * K * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference and verification (OK since K and sizes moderate; for very large K this is slow)
    cpu_csr_spmm(M, N, nnz, h_csrRow, h_csrCol, h_csrVal, h_B, K, h_Y_ref);

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
    double bytes = (double)nnz * (sizeof(int) + sizeof(float)) + (double)(M+1) * sizeof(int) + (double)N * K * sizeof(float) + (double)M * K * sizeof(float);
    double gbs = bytes / (avg_ms / 1000.0) / 1e9;
    double mem_csr_mb = ((nnz * (sizeof(int) + sizeof(float)) + (M+1)*sizeof(int)))/1e6;
    double mem_dense_mb = ((double)(N*K + M*K)*sizeof(float))/1e6;

    printf("\n===== CSR Baseline SpMM Metrics =====\n");
    printf("Matrix: %s\n", matrix_file);
    printf("Dimensions: %d x %d\n", M, N);
    printf("NNZ: %d\n", nnz);
    printf("K: %d\n", K);
    printf("Avg Time: %.6f ms\n", avg_ms);
    printf("Performance: %.3f GFLOP/s\n", gflops);
    printf("Effective Bandwidth: %.3f GB/s\n", gbs);
    printf("Memory Footprint: CSR = %.2f MB, Dense matrices = %.2f MB\n", mem_csr_mb, mem_dense_mb);
    printf("====================================\n");

    // cleanup
    if (dBuffer) cudaFree(dBuffer);
    cudaFree(d_csrRow);
    cudaFree(d_csrCol);
    cudaFree(d_csrVal);
    cudaFree(d_B);
    cudaFree(d_Y);

    free(h_cooRow);
    free(h_cooCol);
    free(h_cooVal);
    free(h_csrRow);
    free(h_csrCol);
    free(h_csrVal);
    free(h_B);
    free(h_Y);
    free(h_Y_ref);

    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matY);
    cusparseDestroy(handle);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
