#include <iostream>
#include <algorithm>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <cuda_runtime.h>

using namespace std;


uint THREADS_PER_BLOCK = 512;
const uint N = 1<<20; 

void initialize(int* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 10000;
    }
}

bool verify(int* arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) return false;
    }
    return true;
}

void comp_and_swap(int arr[], int i, int j, int direction) {
    if ((direction == 1 && arr[i] > arr[j]) || (direction == 0 && arr[i] < arr[j])) {
        swap(arr[i], arr[j]);
    }
}

void bitonic_merge(int arr[], int low, int cnt, int direction) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++) {
            comp_and_swap(arr, i, i + k, direction);
        }
        bitonic_merge(arr, low, k, direction);
        bitonic_merge(arr, low + k, k, direction);
    }
}

void bitonic_sort(int arr[], int low, int cnt, int direction) {
    if (cnt > 1) {
        int k = cnt / 2;
        bitonic_sort(arr, low, k, 1);
        bitonic_sort(arr, low + k, k, 0);
        bitonic_merge(arr, low, cnt, direction);
    }
}

void cpu_sort_array(int arr[], int n) {
    int up = 1;
    bitonic_sort(arr, 0, n, up);
}

__global__ void bitonic_sort_step(int *dev_values, int j, int k) {
    uint i = threadIdx.x + blockDim.x * blockIdx.x;
    uint ixj = i | j;

    if (ixj != i) {
        bool is_ascending = ((i & k) == 0);
        
        int a = dev_values[i];
        int b = dev_values[ixj];

        int min_val = (a < b) ? a : b;
        int max_val = (a < b) ? b : a;

        dev_values[i]   = is_ascending ? min_val : max_val;
        dev_values[ixj] = is_ascending ? max_val : min_val;
    }
}

void gpu_sort_array(int* d_data) {
    dim3 blocks(N / THREADS_PER_BLOCK, 1);
    dim3 threads(THREADS_PER_BLOCK, 1);

    
    for (int k = 2; k <= N; k <<= 1) {      // Bitonic sequence length
        for (int j = k >> 1; j > 0; j >>= 1) { // Comparison distance
            bitonic_sort_step<<<blocks, threads>>>(d_data, j, k);
        }
    }
    
    cudaDeviceSynchronize(); 
}

void printResults(double cpu_time, double gpu_time, bool cpu_valid, bool gpu_valid, int n) {
    const int lw = 26; // label width
    const int vw = 26; // value width
    char cpu_str[32], gpu_str[32], speed_str[32];
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    sprintf(cpu_str, "%.6f s", cpu_time);
    sprintf(gpu_str, "%.6f s", gpu_time);
    sprintf(speed_str, "%.2f x", cpu_time / gpu_time);

    printf("\n");
    printf("+----------------------------+----------------------------+\n");
    printf("| %-*s | %-*s |\n", lw, "Configuration Metric", vw, "Value");
    printf("+----------------------------+----------------------------+\n");
    
    printf("| %-*s | %-*s |\n", lw, "GPU Model", vw, prop.name);
    printf("| %-*s | %-26d |\n", lw, "Threads Per Block", THREADS_PER_BLOCK);
    printf("| %-*s | %-26d |\n", lw, "Total Blocks", n / THREADS_PER_BLOCK);
    printf("| %-*s | %-26d |\n", lw, "Array Size (N)", n);
    
    printf("+----------------------------+----------------------------+\n");
    printf("| %-*s | %-*s |\n", lw, "Performance Metric", vw, "Value");
    printf("+----------------------------+----------------------------+\n");
    
    printf("| %-*s | %-*s |\n", lw, "CPU Execution Time", vw, cpu_str);
    printf("| %-*s | %-*s |\n", lw, "GPU Execution Time", vw, gpu_str);
    printf("| %-*s | %-*s |\n", lw, "SpeedUp (CPU/GPU)", vw, speed_str);
    
    printf("+----------------------------+----------------------------+\n");
    printf("| %-*s | %-*s |\n", lw, "Validation Status", vw, "Result");
    printf("+----------------------------+----------------------------+\n");
    
    printf("| %-*s | %-*s |\n", lw, "CPU Sort Status", vw, cpu_valid ? "SUCCESS" : "FAILED");
    printf("| %-*s | %-*s |\n", lw, "GPU Sort Status", vw, gpu_valid ? "SUCCESS" : "FAILED");
    printf("+----------------------------+----------------------------+\n");
    printf("\n");
}

int main(int argc, char** argv) {
    if (argc > 1) {
        THREADS_PER_BLOCK = atoi(argv[1]);
    }
    
    int *h_arr = (int*)malloc(N * sizeof(int));
    int *h_cpu_arr = (int*)malloc(N * sizeof(int));
    
    initialize(h_arr, N);
    
    memcpy(h_cpu_arr, h_arr, N * sizeof(int));

    clock_t start_cpu = clock();
    cpu_sort_array(h_cpu_arr, N);
    clock_t end_cpu = clock();

    int *d_data;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    clock_t start_gpu = clock();
    gpu_sort_array(d_data);
    clock_t end_gpu = clock();
    
    cudaMemcpy(h_arr, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    bool cpu_valid = verify(h_cpu_arr, N);
    bool gpu_valid = verify(h_arr, N);
    double cpu_time = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
    double gpu_time = (double)(end_gpu - start_gpu) / CLOCKS_PER_SEC;

    printResults(cpu_time, gpu_time, cpu_valid, gpu_valid, N);
    
    free(h_arr);
    free(h_cpu_arr);
    cudaFree(d_data);

    return 0;
}