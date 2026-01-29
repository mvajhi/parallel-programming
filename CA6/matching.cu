#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <limits>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace cv;
using namespace std;

#define SOURCE_IMG_PATH   "/kaggle/input/ut-image/source.jpg"
#define TEMPLATE_IMG_PATH "/kaggle/input/ut-image/template.jpg"
#define MAX_TEMPLATE_SIZE 1024 * 16 * 4

__constant__ uchar c_tpl[MAX_TEMPLATE_SIZE];

double apply_template(int tpl_height, int tpl_width, int row, int src_width, int col, const uchar *src_ptr, const uchar *tpl_ptr)
{
    double result = 0.0;
    for (int t_row = 0; t_row < tpl_height; ++t_row)
    {
        for (int t_col = 0; t_col < tpl_width; ++t_col)
        {
            int src_idx = (row + t_row) * src_width + (col + t_col);

            int tpl_idx = (t_row * tpl_width) + t_col;

            int img_val = (int)src_ptr[src_idx];
            int tpl_val = (int)tpl_ptr[tpl_idx];

            int diff = img_val - tpl_val;
            result += (diff * diff);
            // result += abs(diff);
        }
    }

    return result;
}

Mat compute_ssd_cpu(const Mat& source_img, const Mat& template_img) {
    int src_height = source_img.rows;
    int src_width  = source_img.cols;
    int tpl_height = template_img.rows;
    int tpl_width  = template_img.cols;

    int out_rows = src_height - tpl_height + 1;
    int out_cols = src_width - tpl_width + 1;

    Mat result_map = Mat::zeros(out_rows, out_cols, CV_32FC1);

    cout << "Starting Serial SSD computation..." << endl;

    const uchar* src_ptr = source_img.data;
    const uchar* tpl_ptr = template_img.data;
    float* result_ptr = (float*)result_map.data;

    for (int row = 0; row < out_rows; ++row) {
        for (int col = 0; col < out_cols; ++col) {
            int res_idx = row * out_cols + col;
            result_ptr[res_idx] = (float)apply_template(tpl_height, tpl_width, row, src_width, col, src_ptr, tpl_ptr);
        }
    }

    return result_map;
}

__global__ void ssd_kernel(const uchar* d_src, float* d_result,
                           int src_width, int src_height, 
                           int tpl_width, int tpl_height,
                           int out_width, int out_height) {
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < out_width && row < out_height) {
        
        double sum_sq_diff = 0.0;

        for (int t_row = 0; t_row < tpl_height; ++t_row) {
            for (int t_col = 0; t_col < tpl_width; ++t_col) {
                
                int src_idx = (row + t_row) * src_width + (col + t_col);
                int tpl_idx = (t_row * tpl_width) + t_col;

                int img_val = d_src[src_idx];
                int tpl_val = c_tpl[tpl_idx];
                
                int diff = img_val - tpl_val;
                sum_sq_diff += (double)(diff * diff);
            }
        }
        
        int res_idx = row * out_width + col;
        d_result[res_idx] = (float)sum_sq_diff;
    }
}

Mat compute_ssd_cuda(const Mat& source_img, const Mat& template_img) {
    int src_h = source_img.rows;
    int src_w = source_img.cols;
    int tpl_h = template_img.rows;
    int tpl_w = template_img.cols;
    int out_h = src_h - tpl_h + 1;
    int out_w = src_w - tpl_w + 1;

    Mat result_map = Mat::zeros(out_h, out_w, CV_32FC1);

    size_t src_sz = src_w * src_h * sizeof(uchar);
    size_t tpl_sz = tpl_w * tpl_h * sizeof(uchar);
    size_t res_sz = out_w * out_h * sizeof(float);

    uchar *d_src;
    float *d_res;

    cudaMalloc((void**)&d_src, src_sz);
    cudaMalloc((void**)&d_res, res_sz);

    cudaMemcpy(d_src, source_img.data, src_sz, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_tpl, template_img.data, tpl_sz);

    dim3 block(32, 32);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);

    ssd_kernel<<<grid, block>>>(d_src, d_res, src_w, src_h, tpl_w, tpl_h, out_w, out_h);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(result_map.data, d_res, res_sz, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_res);

    return result_map;
}

bool compare_results(const Mat& cpu_res, const Mat& gpu_res) {
    if (cpu_res.size() != gpu_res.size()) return false;
    int errors = 0;
    const float epsilon = 0.1f; 
    const float* cpu_ptr = (float*)cpu_res.data;
    const float* gpu_ptr = (float*)gpu_res.data;
    int total = cpu_res.rows * cpu_res.cols;

    for (int i = 0; i < total; ++i) {
        if (abs(cpu_ptr[i] - gpu_ptr[i]) > epsilon) errors++;
    }

    if (errors == 0) cout << ">>> VERIFICATION PASSED." << endl;
    else cout << ">>> VERIFICATION FAILED: " << errors << " mismatches." << endl;
    return errors == 0;
}

void save_result_images(Mat& color_src, const Mat& result_map, int tpl_w, int tpl_h) {
    double min_val; Point min_loc;
    minMaxLoc(result_map, &min_val, nullptr, &min_loc, nullptr);
    
    rectangle(color_src, min_loc, Point(min_loc.x + tpl_w, min_loc.y + tpl_h), Scalar(0, 255, 0), 2);
    imwrite("output_match.png", color_src);
    
    Mat display;
    normalize(result_map, display, 0, 255, NORM_MINMAX, CV_8UC1);
    imwrite("output_heatmap.png", display);
}

void print_results(double cpu_time, double gpu_time, bool verification_passed, 
                  int src_w, int src_h, int tpl_w, int tpl_h, 
                  dim3 grid, dim3 block) {
    const int lw = 26; // label width
    const int vw = 26; // value width
    char cpu_str[32], gpu_str[32], speed_str[32];
    char img_res_str[32], tpl_res_str[32], grid_str[32], block_str[32];

    // Get GPU Name
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Format Times
    sprintf(cpu_str, "%.6f s", cpu_time);
    sprintf(gpu_str, "%.6f s", gpu_time);
    if (gpu_time > 0)
        sprintf(speed_str, "%.2f x", cpu_time / gpu_time);
    else
        sprintf(speed_str, "Inf");

    // Format Configuration Info
    sprintf(img_res_str, "%dx%d", src_w, src_h);
    sprintf(tpl_res_str, "%dx%d", tpl_w, tpl_h);
    sprintf(grid_str, "%dx%d (%d blocks)", grid.x, grid.y, grid.x * grid.y);
    sprintf(block_str, "%dx%d (%d threads)", block.x, block.y, block.x * block.y);

    printf("\n");
    printf("+----------------------------+----------------------------+\n");
    printf("| %-*s | %-*s |\n", lw, "Configuration Metric", vw, "Value");
    printf("+----------------------------+----------------------------+\n");
    
    printf("| %-*s | %-*s |\n", lw, "GPU Model", vw, prop.name);
    printf("| %-*s | %-*s |\n", lw, "Image Size", vw, img_res_str);
    printf("| %-*s | %-*s |\n", lw, "Template Size", vw, tpl_res_str);
    printf("| %-*s | %-*s |\n", lw, "Grid Dimensions", vw, grid_str);
    printf("| %-*s | %-*s |\n", lw, "Block Dimensions", vw, block_str);
    
    printf("+----------------------------+----------------------------+\n");
    printf("| %-*s | %-*s |\n", lw, "Performance Metric", vw, "Value");
    printf("+----------------------------+----------------------------+\n");
    
    printf("| %-*s | %-*s |\n", lw, "CPU Execution Time", vw, cpu_str);
    printf("| %-*s | %-*s |\n", lw, "GPU Execution Time", vw, gpu_str);
    printf("| %-*s | %-*s |\n", lw, "SpeedUp (CPU/GPU)", vw, speed_str);
    
    printf("+----------------------------+----------------------------+\n");
    printf("| %-*s | %-*s |\n", lw, "Validation Status", vw, "Result");
    printf("+----------------------------+----------------------------+\n");
    
    printf("| %-*s | %-*s |\n", lw, "Verification (GPU match)", vw, verification_passed ? "PASSED" : "FAILED");
    printf("+----------------------------+----------------------------+\n");
    printf("\n");
}


int main() {
    Mat src_color = imread(SOURCE_IMG_PATH);
    Mat tpl_color = imread(TEMPLATE_IMG_PATH);

    Mat src_gray, tpl_gray;
    cvtColor(src_color, src_gray, COLOR_BGR2GRAY);
    cvtColor(tpl_color, tpl_gray, COLOR_BGR2GRAY);

    clock_t start_cpu = clock();
    Mat cpu_res = compute_ssd_cpu(src_gray, tpl_gray);
    clock_t end_cpu = clock();
    double cpu_seconds = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;

    
    clock_t start_gpu = clock();
    Mat gpu_res = compute_ssd_cuda(src_gray, tpl_gray);
    clock_t end_gpu = clock();
    double gpu_seconds = (double)(end_gpu - start_gpu) / CLOCKS_PER_SEC;

    bool is_valid = compare_results(cpu_res, gpu_res);

    int out_h = src_gray.rows - tpl_gray.rows + 1;
    int out_w = src_gray.cols - tpl_gray.cols + 1;
    dim3 block(32, 32);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);
    print_results(cpu_seconds, gpu_seconds, is_valid,
                  src_gray.cols, src_gray.rows, 
                  tpl_gray.cols, tpl_gray.rows, 
                  grid, block);

    save_result_images(src_color, gpu_res, tpl_gray.cols, tpl_gray.rows);

    return 0;
}