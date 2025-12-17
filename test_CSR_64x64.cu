// spmm_csr_tiled64.cu
// Compile: nvcc -O3 spmm_csr_tiled64.cu -o spmm_csr_tiled64
// Usage: ./spmm_csr_tiled64 matrix.mtx [iters=10] [K=N]
// nvcc test_CSR_64x64.cu -lcusparse -lcudart -o test_CSR_64x64
// ./test_CSR_64x64 Mat_data/barth.mtx

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

#define TILE_ROWS 64
#define TILE_COLS 64

// Robust COO reader (same as baseline)
void read_mtx_coo_robust(const char* filename, int *outM, int *outN, int *outNnz,
                         int **row, int **col, float **val)
{
    FILE *f = fopen(filename, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); exit(1); }

    char line[1024];
    do {
        if (!fgets(line, sizeof(line), f)) { fprintf(stderr, "Empty file\n"); exit(1);}
        char *p = line; while (isspace((unsigned char)*p)) p++;
        if (*p == '%' || *p == '\0') continue; break;
    } while(1);

    int M=0,N=0,header_nnz=0;
    if(sscanf(line,"%d %d %d",&M,&N,&header_nnz)<2){ fprintf(stderr,"Malformed header\n"); exit(1);}
    if(header_nnz<0) header_nnz=0;

    int cap = (header_nnz>0)?header_nnz:1024;
    int count =0;
    int *r=(int*)malloc(cap*sizeof(int)),*c=(int*)malloc(cap*sizeof(int));
    float *v=(float*)malloc(cap*sizeof(float));
    if(!r||!c||!v){ fprintf(stderr,"malloc failed\n"); exit(1); }

    int line_no=1;
    while(fgets(line,sizeof(line),f)){
        line_no++;
        char *p = line; while (isspace((unsigned char)*p)) p++;
        if(*p=='\0'||*p=='%') continue;
        int ri=-1,ci=-1; double vd=0.0;
        int nitems=sscanf(p,"%d %d %lf",&ri,&ci,&vd);
        if(nitems<2){ fprintf(stderr,"Warning unparsable line %d\n",line_no); continue; }
        ri--; ci--;
        float vf = (nitems>=3)?(float)vd:1.0f;
        if(count>=cap){
            int newcap = cap*2;
            r=(int*)realloc(r,newcap*sizeof(int));
            c=(int*)realloc(c,newcap*sizeof(int));
            v=(float*)realloc(v,newcap*sizeof(float));
            if(!r||!c||!v){ fprintf(stderr,"realloc failed\n"); exit(1);}
            cap=newcap;
        }
        r[count]=ri; c[count]=ci; v[count]=vf; count++;
    }
    fclose(f);
    r=(int*)realloc(r,count*sizeof(int));
    c=(int*)realloc(c,count*sizeof(int));
    v=(float*)realloc(v,count*sizeof(float));

    *outM=M; *outN=N; *outNnz=count;
    *row=r; *col=c; *val=v;
    if(header_nnz>0 && header_nnz!=count) fprintf(stderr,"Header nnz=%d but read %d\n",header_nnz,count);
}

// CPU reference SpMM (same as baseline)
void cpu_csr_spmm(int M, int N, int nnz, const int *csrRow, const int *csrCol, const float *csrVal,
                  const float *B, int K, float *Y)
{
    for(int i=0;i<M;i++){
        int rstart=csrRow[i], rend=csrRow[i+1];
        for(int kk=0;kk<K;kk++) Y[i*K+kk]=0.0f;
        for(int idx=rstart;idx<rend;idx++){
            int c = csrCol[idx];
            float aval = csrVal[idx];
            const float *brow = &B[c*K];
            float *yrow = &Y[i*K];
            for(int kk=0;kk<K;kk++) yrow[kk]+=aval*brow[kk];
        }
    }
}

// Tiled CSR x Dense kernel
// Each block corresponds to a tile (tileRow, tileCol)
// Each thread handles one row inside the tile (up to TILE_ROWS threads per block)
// Each thread accumulates TILE_COLS results in a small local array
__global__ void spmm_csr_tiled_kernel(int M, int N, int nnz,
                                      const int *csrRow, const int *csrCol, const float *csrVal,
                                      const float *B, int K, float *Y)
{
    const int tileRow = blockIdx.x;
    const int tileCol = blockIdx.y;
    const int tid = threadIdx.x; // 0..TILE_ROWS-1 expected
    const int rowBase = tileRow * TILE_ROWS;
    const int colBase = tileCol * TILE_COLS;

    const int row = rowBase + tid;
    if (tid >= TILE_ROWS) return;

    // compute actual tile width (handle incomplete tile at boundaries)
    int colEnd = colBase + TILE_COLS;
    if (colEnd > K) colEnd = K;
    const int tileW = colEnd - colBase;
    if (tileW <= 0) return;

    // local accumulator per thread
    float acc[TILE_COLS]; // TILE_COLS compile-time constant (64)
    #pragma unroll
    for (int i=0;i<TILE_COLS;i++) acc[i]=0.0f;

    if (row < M) {
        int rstart = csrRow[row];
        int rend = csrRow[row+1];
        for (int idx = rstart; idx < rend; ++idx) {
            int c = csrCol[idx];
            float aval = csrVal[idx];
            // pointer to B[c, colBase .. colEnd-1] (row-major)
            const float *bptr = &B[(long long)c * K + colBase];
            // multiply-accumulate for this tile's columns
            #pragma unroll 4
            for (int kk = 0; kk < tileW; ++kk) {
                acc[kk] += aval * bptr[kk];
            }
        }
        // write back to Y
        float *yptr = &Y[(long long)row * K + colBase];
        for (int kk=0; kk<tileW; ++kk) {
            yptr[kk] = acc[kk];
        }
    }
    // rows >= M do nothing
}

int main(int argc, char** argv)
{
    if(argc<2){ printf("Usage: %s matrix.mtx [iters=10] [K=N]\n",argv[0]); return 0;}
    const char* matrix_file = argv[1];
    int ITERS = (argc>=3)?atoi(argv[2]):10;
    int K_override = (argc>=4)?atoi(argv[3]):-1;

    int M,N,nnz;
    int *h_cooRow=NULL,*h_cooCol=NULL; float *h_cooVal=NULL;
    read_mtx_coo_robust(matrix_file,&M,&N,&nnz,&h_cooRow,&h_cooCol,&h_cooVal);
    printf("Loaded matrix %s : %d x %d with %d entries\n",matrix_file,M,N,nnz);

    // COO -> CSR
    int *h_csrRow=(int*)calloc(M+1,sizeof(int));
    for(int i=0;i<nnz;i++) {
        int r = h_cooRow[i];
        if (r < 0 || r >= M) { fprintf(stderr,"COO row out of range\n"); exit(1); }
        h_csrRow[r+1]++;
    }
    for(int i=0;i<M;i++) h_csrRow[i+1]+=h_csrRow[i];

    int *tempRow=(int*)malloc((M+1)*sizeof(int));
    memcpy(tempRow,h_csrRow,(M+1)*sizeof(int));

    int *h_csrCol=(int*)malloc(nnz*sizeof(int));
    float *h_csrVal=(float*)malloc(nnz*sizeof(float));
    for(int i=0;i<nnz;i++){
        int r=h_cooRow[i];
        int dest=tempRow[r]++;
        h_csrCol[dest]=h_cooCol[i];
        h_csrVal[dest]=h_cooVal[i];
    }
    free(tempRow);

    int K = (K_override>0)?K_override:N;
    printf("Dense B: %d x %d, iterations=%d\n",N,K,ITERS);

    // Host dense B and output
    float *h_B=(float*)malloc((size_t)N*(size_t)K*sizeof(float));
    if(!h_B) { fprintf(stderr,"malloc h_B failed\n"); return 1; }
    for(long long i=0;i<(long long)N*K;i++) h_B[i]=1.0f;

    float *h_Y=(float*)malloc((size_t)M*(size_t)K*sizeof(float));
    float *h_Y_ref=(float*)malloc((size_t)M*(size_t)K*sizeof(float));
    if(!h_Y || !h_Y_ref){ fprintf(stderr,"malloc h_Y failed\n"); return 1; }

    // Device memory
    int *d_csrRow=NULL,*d_csrCol=NULL;
    float *d_csrVal=NULL,*d_B=NULL,*d_Y=NULL;
    CUDA_CHECK(cudaMalloc(&d_csrRow,(M+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrCol,nnz*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrVal,nnz*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B,(size_t)N*(size_t)K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y,(size_t)M*(size_t)K*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_csrRow,h_csrRow,(M+1)*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrCol,h_csrCol,nnz*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrVal,h_csrVal,nnz*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B,h_B,(size_t)N*(size_t)K*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_Y,0,(size_t)M*(size_t)K*sizeof(float)));

    // Kernel launch configuration
    int numRowTiles = (M + TILE_ROWS - 1) / TILE_ROWS;
    int numColTiles = (K + TILE_COLS - 1) / TILE_COLS;
    dim3 grid(numRowTiles, numColTiles);
    dim3 block(TILE_ROWS); // one thread per row in tile

    // Warm-up
    for(int i=0;i<3;i++){
        spmm_csr_tiled_kernel<<<grid,block>>>(M,N,nnz,d_csrRow,d_csrCol,d_csrVal,d_B,K,d_Y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // timing
    cudaEvent_t start,stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for(int i=0;i<ITERS;i++){
        // zero output before each iteration
        CUDA_CHECK(cudaMemset(d_Y,0,(size_t)M*(size_t)K*sizeof(float)));
        spmm_csr_tiled_kernel<<<grid,block>>>(M,N,nnz,d_csrRow,d_csrCol,d_csrVal,d_B,K,d_Y);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float msTotal=0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msTotal,start,stop));
    float avg_ms = msTotal / ITERS;

    CUDA_CHECK(cudaMemcpy(h_Y,d_Y,(size_t)M*(size_t)K*sizeof(float),cudaMemcpyDeviceToHost));

    // CPU reference verification
    cpu_csr_spmm(M,N,nnz,h_csrRow,h_csrCol,h_csrVal,h_B,K,h_Y_ref);
    int mismatches=0; double max_abs_err=0.0;
    for(long long i=0;i<(long long)M*K;i++){
        float a=h_Y_ref[i],b=h_Y[i];
        float err=fabsf(a-b);
        if(err>1e-3f){ if(mismatches<10) printf("Mismatch idx %lld: cpu=%f gpu=%f err=%f\n",i,a,b,err); mismatches++; }
        if(err>max_abs_err) max_abs_err=err;
    }
    if(mismatches==0) printf("Verification: PASSED (max err=%.6f)\n",max_abs_err);
    else printf("Verification: FAILED (%d mismatches, max err=%.6f)\n",mismatches,max_abs_err);

    // Basic metrics
    double flops = 2.0*(double)nnz*(double)K;
    double gflops = flops / (avg_ms/1000.0)/1e9;
    double bytes = (double)nnz*(sizeof(int)+sizeof(float)) + (double)(M+1)*sizeof(int) + (double)(N*(size_t)K + M*(size_t)K)*sizeof(float);
    double gbs = bytes / (avg_ms/1000.0)/1e9;
    double mem_csr_mb = ((nnz*(sizeof(int)+sizeof(float)) + (M+1)*sizeof(int)))/1e6;
    double mem_dense_mb = ((double)(N*(size_t)K + M*(size_t)K)*sizeof(float))/1e6;

    // GPU info
    cudaDeviceProp devProp; int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&devProp,dev));
    int totalSM = devProp.multiProcessorCount;
    int maxThreadsPerSM = devProp.maxThreadsPerMultiProcessor;

    // Sparsity metrics
    double avg_nnz_per_row = (double)nnz/M;
    double row_variance=0.0;
    for(int i=0;i<M;i++){
        double count = h_csrRow[i+1]-h_csrRow[i];
        row_variance += (count - avg_nnz_per_row)*(count - avg_nnz_per_row);
    }
    row_variance /= M;

    // --- Load Balancing / Parallel Efficiency ---
    double warp_size = 32.0;
    double thread_efficiency = 0.0;
    for(int i=0;i<M;i++){
        double nnz_in_row = h_csrRow[i+1]-h_csrRow[i];
        double row_eff = fmin(nnz_in_row, warp_size)/warp_size;
        thread_efficiency += row_eff;
    }
    thread_efficiency /= M;

    // Approximate occupancy (very rough)
    int max_threads_per_block = devProp.maxThreadsPerBlock;
    int warp_per_sm = devProp.maxThreadsPerMultiProcessor / (int)warp_size;
    double occupancy = (double)(warp_per_sm * warp_size) / max_threads_per_block;

    printf("\n===== Tiled CSR SpMM (64x64) Metrics =====\n");
    printf("Matrix: %s\nDimensions: %d x %d, NNZ: %d, K=%d\n",matrix_file,M,N,nnz,K);
    printf("Grid: (%d row-tiles) x (%d col-tiles) -> %d blocks\n",numRowTiles,numColTiles,numRowTiles*numColTiles);
    printf("Block threads: %d (one thread per row in tile)\n",TILE_ROWS);
    printf("Avg Time: %.6f ms\nGFLOPS: %.3f\nEffective Bandwidth: %.3f GB/s\n",avg_ms,gflops,gbs);
    printf("Memory Footprint: CSR=%.2f MB, Dense matrices=%.2f MB\n",mem_csr_mb,mem_dense_mb);
    printf("GPU: %s, SMs=%d, MaxThreads/SM=%d\n",devProp.name,totalSM,maxThreadsPerSM);
    printf("Total FLOPs: %.0f\n",flops);
    printf("Avg nnz/row: %.2f, Variance: %.2f\n",avg_nnz_per_row,row_variance);
    printf("Thread Efficiency (approx): %.3f\n", thread_efficiency);
    printf("Approximate Occupancy (very rough): %.3f\n", occupancy);
    printf("==========================================\n");

    // cleanup
    cudaFree(d_csrRow); cudaFree(d_csrCol); cudaFree(d_csrVal);
    cudaFree(d_B); cudaFree(d_Y);
    free(h_cooRow); free(h_cooCol); free(h_cooVal);
    free(h_csrRow); free(h_csrCol); free(h_csrVal);
    free(h_B); free(h_Y); free(h_Y_ref);
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
