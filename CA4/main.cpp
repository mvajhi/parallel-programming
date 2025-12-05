#include <omp.h>
#include <cmath>
#include <queue>
#include <chrono>
#include <iostream>
#include <pthread.h>
#include <sys/time.h>
#include <x86intrin.h>
#include <immintrin.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

class Sobel
{
protected:
    const int kernel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    const int kernel_y[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };

public:
    Sobel() {}
    ~Sobel() {}

    virtual cv::Mat apply(const cv::Mat &in_img) = 0;
};

class SobelSerial : public Sobel
{
protected:
    unsigned char apply_for_one_pixel(const unsigned char *in_image, int row, unsigned int NCOLS, int col)
    {
        int gx = 0;
        int gy = 0;

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                int pixel_val = in_image[(row + i - 1) * NCOLS + col + j - 1];
                if (kernel_x[i][j] != 0)
                    gx += kernel_x[i][j] * pixel_val;
                if (kernel_y[i][j] != 0)
                    gy += kernel_y[i][j] * pixel_val;
            }

        double magnitude = sqrt(static_cast<double>(gx * gx) + static_cast<double>(gy * gy));

        return cv::saturate_cast<unsigned char>(magnitude);
    }

public:
    cv::Mat apply(const cv::Mat &in_img) override
    {
        unsigned int NCOLS = in_img.cols;
        unsigned int NROWS = in_img.rows;

        cv::Mat out_img(in_img.rows, in_img.cols, CV_8U);

        const unsigned char *in_image = (const unsigned char *)in_img.data;
        unsigned char *out_image = (unsigned char *)out_img.data;

        for (int row = 1; row < NROWS - 1; row++)
            for (int col = 1; col < NCOLS - 1; col++)
                out_image[row * NCOLS + col] = apply_for_one_pixel(in_image, row, NCOLS, col);
        return out_img;
    }
};

class SobelSIMD : public Sobel
{
protected:
    inline void accumulate_kernel_product(int kernel_value, const __m256i &p_16_low, const __m256i &p_16_high, __m256i &acc_low, __m256i &acc_high)
    {
        if (kernel_value != 0)
        {
            __m256i k_value = _mm256_set1_epi16((short int)kernel_value);
            __m256i multi_low = _mm256_mullo_epi16(k_value, p_16_low);
            acc_low = _mm256_add_epi16(multi_low, acc_low);

            __m256i multi_high = _mm256_mullo_epi16(k_value, p_16_high);
            acc_high = _mm256_add_epi16(multi_high, acc_high);
        }
    }

    inline void apply_sobel_kernels(const unsigned char *in_image, int row, unsigned int NCOLS, int col, __m256i &gx_16_low, __m256i &gx_16_high, __m256i &gy_16_low, __m256i &gy_16_high)
    {
        __m256i p_8, p_16_low, p_16_high;
        __m256i zero = _mm256_setzero_si256();

        gx_16_low = zero;
        gx_16_high = zero;
        gy_16_low = zero;
        gy_16_high = zero;

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                p_8 = _mm256_loadu_si256((__m256i const *)&in_image[(row + i - 1) * NCOLS + col + j - 1]);

                p_16_low = _mm256_unpacklo_epi8(p_8, zero);
                p_16_high = _mm256_unpackhi_epi8(p_8, zero);

                accumulate_kernel_product(kernel_x[i][j], p_16_low, p_16_high, gx_16_low, gx_16_high);
                accumulate_kernel_product(kernel_y[i][j], p_16_low, p_16_high, gy_16_low, gy_16_high);
            }
        }
    }

    inline void calculate_magnitude(const __m256i &gx_16_low, const __m256i &gx_16_high, const __m256i &gy_16_low, const __m256i &gy_16_high, __m256i &result_a, __m256i &result_b, __m256i &result_c, __m256i &result_d)
    {
        __m256i gx_32_a, gx_32_b, gx_32_c, gx_32_d;
        __m256i gy_32_a, gy_32_b, gy_32_c, gy_32_d;
        __m256i gx_squared_a, gx_squared_b, gx_squared_c, gx_squared_d;
        __m256i gy_squared_a, gy_squared_b, gy_squared_c, gy_squared_d;
        __m256i sum_a, sum_b, sum_c, sum_d;
        __m256 sqrt_a, sqrt_b, sqrt_c, sqrt_d;

        convert_gradients_16bit_to_32(gx_32_a, gx_16_low, gx_32_b, gx_32_c, gx_16_high, gx_32_d, gy_32_a, gy_16_low, gy_32_b, gy_32_c, gy_16_high, gy_32_d);
        square_gradients(gx_squared_a, gx_32_a, gx_squared_b, gx_32_b, gx_squared_c, gx_32_c, gx_squared_d, gx_32_d, gy_squared_a, gy_32_a, gy_squared_b, gy_32_b, gy_squared_c, gy_32_c, gy_squared_d, gy_32_d);
        sum_gx_gy(sum_a, gx_squared_a, gy_squared_a, sum_b, gx_squared_b, gy_squared_b, sum_c, gx_squared_c, gy_squared_c, sum_d, gx_squared_d, gy_squared_d);
        sqrt_g_sum(sqrt_a, sum_a, sqrt_b, sum_b, sqrt_c, sum_c, sqrt_d, sum_d);
        convert_float_to_int(result_a, sqrt_a, result_b, sqrt_b, result_c, sqrt_c, result_d, sqrt_d);
    }

    inline void convert_float_to_int(__m256i &result_a, const __m256 &sqrt_a, __m256i &result_b, const __m256 &sqrt_b, __m256i &result_c, const __m256 &sqrt_c, __m256i &result_d, const __m256 &sqrt_d)
    {
        result_a = _mm256_cvtps_epi32(sqrt_a);
        result_b = _mm256_cvtps_epi32(sqrt_b);
        result_c = _mm256_cvtps_epi32(sqrt_c);
        result_d = _mm256_cvtps_epi32(sqrt_d);
    }

    inline void sqrt_g_sum(__m256 &sqrt_a, const __m256i &sum_a, __m256 &sqrt_b, const __m256i &sum_b, __m256 &sqrt_c, const __m256i &sum_c, __m256 &sqrt_d, const __m256i &sum_d)
    {
        sqrt_a = _mm256_sqrt_ps(_mm256_cvtepi32_ps(sum_a));
        sqrt_b = _mm256_sqrt_ps(_mm256_cvtepi32_ps(sum_b));
        sqrt_c = _mm256_sqrt_ps(_mm256_cvtepi32_ps(sum_c));
        sqrt_d = _mm256_sqrt_ps(_mm256_cvtepi32_ps(sum_d));
    }

    inline void sum_gx_gy(__m256i &sum_a, const __m256i &gx_squared_a, const __m256i &gy_squared_a, __m256i &sum_b,
                          const __m256i &gx_squared_b, const __m256i &gy_squared_b, __m256i &sum_c, const __m256i &gx_squared_c,
                          const __m256i &gy_squared_c, __m256i &sum_d, const __m256i &gx_squared_d, const __m256i &gy_squared_d)
    {
        sum_a = _mm256_add_epi32(gx_squared_a, gy_squared_a);
        sum_b = _mm256_add_epi32(gx_squared_b, gy_squared_b);
        sum_c = _mm256_add_epi32(gx_squared_c, gy_squared_c);
        sum_d = _mm256_add_epi32(gx_squared_d, gy_squared_d);
    }

    inline void square_gradients(__m256i &gx_squared_a, const __m256i &gx_32_a, __m256i &gx_squared_b,
                                 const __m256i &gx_32_b, __m256i &gx_squared_c, const __m256i &gx_32_c, __m256i &gx_squared_d,
                                 const __m256i &gx_32_d, __m256i &gy_squared_a, const __m256i &gy_32_a, __m256i &gy_squared_b,
                                 const __m256i &gy_32_b, __m256i &gy_squared_c, const __m256i &gy_32_c, __m256i &gy_squared_d, const __m256i &gy_32_d)
    {
        gx_squared_a = _mm256_mullo_epi32(gx_32_a, gx_32_a);
        gx_squared_b = _mm256_mullo_epi32(gx_32_b, gx_32_b);
        gx_squared_c = _mm256_mullo_epi32(gx_32_c, gx_32_c);
        gx_squared_d = _mm256_mullo_epi32(gx_32_d, gx_32_d);

        gy_squared_a = _mm256_mullo_epi32(gy_32_a, gy_32_a);
        gy_squared_b = _mm256_mullo_epi32(gy_32_b, gy_32_b);
        gy_squared_c = _mm256_mullo_epi32(gy_32_c, gy_32_c);
        gy_squared_d = _mm256_mullo_epi32(gy_32_d, gy_32_d);
    }

    inline void convert_gradients_16bit_to_32(__m256i &gx_32_a, const __m256i &gx_16_low, __m256i &gx_32_b, __m256i &gx_32_c,
                                              const __m256i &gx_16_high, __m256i &gx_32_d, __m256i &gy_32_a, const __m256i &gy_16_low,
                                              __m256i &gy_32_b, __m256i &gy_32_c, const __m256i &gy_16_high, __m256i &gy_32_d)
    {
        gx_32_a = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gx_16_low, 0));
        gx_32_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gx_16_low, 1));
        gx_32_c = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gx_16_high, 0));
        gx_32_d = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gx_16_high, 1));

        gy_32_a = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gy_16_low, 0));
        gy_32_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gy_16_low, 1));
        gy_32_c = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gy_16_high, 0));
        gy_32_d = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gy_16_high, 1));
    }

    inline void pack_and_store_results(__m256i &result_a, __m256i &result_b, __m256i &result_c, __m256i &result_d, unsigned char *out_image, int row, unsigned int NCOLS, int col)
    {
        clamp_results(result_a, result_b, result_c, result_d);

        __m256i reordered;
        reduce_and_reorder_results(result_a, result_b, result_c, result_d, reordered);

        _mm256_storeu_si256((__m256i_u *)&out_image[row * NCOLS + col], reordered);
    }

    inline void reduce_and_reorder_results(__m256i &result_a, __m256i &result_b, __m256i &result_c, __m256i &result_d, __m256i &reordered)
    {
        __m256i packed_ab, packed_cd, final_packed;

        packed_ab = _mm256_packus_epi32(result_a, result_b);
        packed_cd = _mm256_packus_epi32(result_c, result_d);

        packed_ab = _mm256_permute4x64_epi64(packed_ab, (3 << 6) | (1 << 4) | (2 << 2) | 0);
        packed_cd = _mm256_permute4x64_epi64(packed_cd, (3 << 6) | (1 << 4) | (2 << 2) | 0);

        final_packed = _mm256_packus_epi16(packed_ab, packed_cd);

        reordered = _mm256_permute4x64_epi64(final_packed, (3 << 6) | (2 << 4) | (1 << 2) | 0);
    }

    inline void clamp_results(__m256i &result_a, __m256i &result_b, __m256i &result_c, __m256i &result_d)
    {
        __m256i zero = _mm256_setzero_si256();
        __m256i max255 = _mm256_set1_epi32(255);

        result_a = _mm256_min_epi32(result_a, max255);
        result_a = _mm256_max_epi32(result_a, zero);

        result_b = _mm256_min_epi32(result_b, max255);
        result_b = _mm256_max_epi32(result_b, zero);

        result_c = _mm256_min_epi32(result_c, max255);
        result_c = _mm256_max_epi32(result_c, zero);

        result_d = _mm256_min_epi32(result_d, max255);
        result_d = _mm256_max_epi32(result_d, zero);
    }

    inline void apply_for_32_pixel(const unsigned char *in_image, int row, unsigned int NCOLS, int col, unsigned char *out_image)
    {
        __m256i gx_16_low, gx_16_high, gy_16_low, gy_16_high;
        apply_sobel_kernels(in_image, row, NCOLS, col, gx_16_low, gx_16_high, gy_16_low, gy_16_high);
        __m256i result_a, result_b, result_c, result_d;
        calculate_magnitude(gx_16_low, gx_16_high, gy_16_low, gy_16_high, result_a, result_b, result_c, result_d);
        pack_and_store_results(result_a, result_b, result_c, result_d, out_image, row, NCOLS, col);
    }

public:
    cv::Mat apply(const cv::Mat &in_img) override
    {
        unsigned int NCOLS = in_img.cols;
        unsigned int NROWS = in_img.rows;

        cv::Mat out_img(in_img.rows, in_img.cols, CV_8U);

        const unsigned char *in_image = (const unsigned char *)in_img.data;
        unsigned char *out_image = (unsigned char *)out_img.data;

        for (int row = 1; row < NROWS - 1; row++)
            for (int col = 1; col < NCOLS - 1; col += 32)
            {
                apply_for_32_pixel(in_image, row, NCOLS, col, out_image);

                if (col == 1) {
                    out_image[(row * NCOLS) + 0] = 0;
                }
                else if (!(col + 32 < NCOLS - 1)) {
                    out_image[(row * NCOLS) + (NCOLS - 1)] = 0;
                }
            }

        out_image[((NROWS-1) * (NCOLS)) + 0] = 0;

        return out_img;
    }
};

class SobelSIMD_OMP : public SobelSIMD
{
public:
    cv::Mat apply(const cv::Mat &in_img) override
    {
        unsigned int NCOLS = in_img.cols;
        unsigned int NROWS = in_img.rows;

        cv::Mat out_img(in_img.rows, in_img.cols, CV_8U);

        const unsigned char *in_image = (const unsigned char *)in_img.data;
        unsigned char *out_image = (unsigned char *)out_img.data;

        #pragma omp parallel for
        for (int row = 1; row < NROWS - 1; row++)
            for (int col = 1; col < NCOLS - 1; col += 32)
            {
                apply_for_32_pixel(in_image, row, NCOLS, col, out_image);

                if (col == 1)
                    out_image[(row * NCOLS) + 0] = 0;
                else if (!(col + 32 < NCOLS - 1))
                {
                    out_image[(row * NCOLS) + (NCOLS - 1)] = 0;
                    out_image[((row + 1) * NCOLS) + 0] = 0;
                }
            }

        out_image[((NROWS - 1) * (NCOLS)) + 0] = 0;

        return out_img;
    }
};

class SobelSIMD_Pthread : public SobelSIMD
{
private:
    struct ThreadData {
        SobelSIMD_Pthread* instance;
        const unsigned char* in_image;
        unsigned char* out_image;
        int start_row;
        int end_row;
        unsigned int NCOLS;
    };

    static void* worker(void* arg) {
        ThreadData* data = static_cast<ThreadData*>(arg);

        const unsigned char* in_image = data->in_image;
        unsigned char* out_image = data->out_image;
        unsigned int NCOLS = data->NCOLS;
        SobelSIMD_Pthread* instance = data->instance;

        for (int row = data->start_row; row < data->end_row; row++) {
            for (int col = 1; col < NCOLS - 1; col += 32) {
                instance->apply_for_32_pixel(in_image, row, NCOLS, col, out_image);

                if (col == 1) {
                    out_image[(row * NCOLS) + 0] = 0;
                }
                else if (!(col + 32 < NCOLS - 1)) {
                    out_image[(row * NCOLS) + (NCOLS - 1)] = 0;
                    out_image[((row + 1) * NCOLS) + 0] = 0; 
                }
            }
        }
        return NULL;
    }

public:
    cv::Mat apply(const cv::Mat &in_img) override
    {
        unsigned int NCOLS = in_img.cols;
        unsigned int NROWS = in_img.rows;

        cv::Mat out_img(in_img.rows, in_img.cols, CV_8U);

        const unsigned char *in_image = (const unsigned char *)in_img.data;
        unsigned char *out_image = (unsigned char *)out_img.data;

        int num_threads = 8;
        pthread_t threads[num_threads];
        ThreadData thread_data[num_threads];
        int total_work_rows = NROWS - 2; 
        int rows_per_thread = total_work_rows / num_threads;
        int start_index = 1;

        for (int i = 0; i < num_threads; i++) {
            thread_data[i].instance = this;
            thread_data[i].in_image = in_image;
            thread_data[i].out_image = out_image;
            thread_data[i].NCOLS = NCOLS;
            thread_data[i].start_row = start_index;
            
            if (i == num_threads - 1) {
                thread_data[i].end_row = NROWS - 1;
            } 
            else {
                thread_data[i].end_row = start_index + rows_per_thread;
            }
            start_index = thread_data[i].end_row;
            int rc = pthread_create(&threads[i], NULL, worker, (void*)&thread_data[i]);
        }

        for (int i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }

        out_image[((NROWS - 1) * (NCOLS)) + 0] = 0;
        return out_img;
    }
};

class FrameQueue {
private:
    queue<cv::Mat> frame_queue;
    int capacity;
    bool finished;

    pthread_mutex_t mutex;
    pthread_cond_t cond_not_full;
    pthread_cond_t cond_not_empty;

public:
    FrameQueue(int cap) : capacity(cap), finished(false) {
        pthread_mutex_init(&mutex, NULL);
        pthread_cond_init(&cond_not_full, NULL);
        pthread_cond_init(&cond_not_empty, NULL);
    }

    ~FrameQueue() {
        pthread_mutex_destroy(&mutex);
        pthread_cond_destroy(&cond_not_full);
        pthread_cond_destroy(&cond_not_empty);
    }

    bool push(const cv::Mat& frame) {
        pthread_mutex_lock(&mutex);

        while (frame_queue.size() >= capacity && !finished) {
            pthread_cond_wait(&cond_not_full, &mutex);
        }

        if (finished) {
            pthread_mutex_unlock(&mutex);
            return false;
        }

        frame_queue.push(frame);
        pthread_cond_signal(&cond_not_empty);
        pthread_mutex_unlock(&mutex);
        return true;
    }

    bool pop(cv::Mat& frame) {
        pthread_mutex_lock(&mutex);

        while (frame_queue.empty() && !finished) {
            pthread_cond_wait(&cond_not_empty, &mutex);
        }

        if (frame_queue.empty() && finished) {
            pthread_mutex_unlock(&mutex);
            return false;
        }

        frame = frame_queue.front();
        frame_queue.pop();
        pthread_cond_signal(&cond_not_full);
        pthread_mutex_unlock(&mutex);
        return true;
    }

    void shutdown() {
        pthread_mutex_lock(&mutex);
        finished = true;
        pthread_cond_broadcast(&cond_not_empty);
        pthread_cond_broadcast(&cond_not_full);
        pthread_mutex_unlock(&mutex);
    }
};

struct PipelineData {
    FrameQueue* inputQueue;
    FrameQueue* outputQueue;
    double cameraFPS;
};

void* input_thread_func(void* arg) {
    PipelineData* data = (PipelineData*)arg;
    #ifdef _WIN32
        cv::VideoCapture cap(0, cv::CAP_DSHOW);        
    #else
        cv::VideoCapture cap(0);
    #endif

    if (!cap.isOpened()) {
        cerr << "Error: Could not open camera" << endl;
        data->inputQueue->shutdown();
        return NULL;
    }

    cap.set(cv::CAP_PROP_FPS, 30);
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30;
    data->cameraFPS = fps;

    cout << "[Input Thread] Camera started." << endl;
    cv::Mat frame;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        if (!data->inputQueue->push(frame.clone())) {
            break;
        }
    }

    cout << "[Input Thread] Finished" << endl;
    data->inputQueue->shutdown();
    return NULL;
}


void* processing_thread_func(void* arg) {
    PipelineData* data = (PipelineData*)arg;
    SobelSIMD_Pthread sobelFilter;
    cv::Mat frame, gray, processed;

    cout << "[Processing Thread] Started.\n";

    while (data->inputQueue->pop(frame)) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        processed = sobelFilter.apply(gray);
        data->outputQueue->push(processed);
    }

    cout << "[Processing Thread] Finished." << endl;
    data->outputQueue->shutdown();
    return NULL;
}


void* output_thread_func(void* arg) {
    PipelineData* data = (PipelineData*)arg;
    cv::Mat frame;
    cv::VideoWriter writer;
    
    int frame_count = 0;
    auto start_time = chrono::steady_clock::now();

    cout << "[Output Thread] Started." << endl;

    while (data->outputQueue->pop(frame)) {
        if (!writer.isOpened()) {
            writer.open("output.avi",  cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),  data->cameraFPS,  frame.size(),  false);
        }

        writer.write(frame);
        cv::imshow("Processed Output", frame);
        
        frame_count++;
        if (frame_count % 30 == 0) {
            auto end_time = chrono::steady_clock::now();            
            chrono::duration<double> elapsed_seconds = end_time - start_time;            
            double fps = 30 / elapsed_seconds.count();
            
            cout << "Instant FPS: " << fps << "\r" << flush;
            start_time = end_time; 
        }

        if (cv::waitKey(1) == 'q') {
            cout << endl;
            cout << "[Output Thread] Quit signal received." << endl;
            break;
        }
    }

    cout << "[Output Thread] Finished." << endl;
    return NULL;
}

int main() {
    FrameQueue q1(10);
    FrameQueue q2(10);

    PipelineData data;
    data.inputQueue = &q1;
    data.outputQueue = &q2;

    pthread_t t_in, t_proc, t_out;

    pthread_create(&t_in, NULL, input_thread_func, &data);
    pthread_create(&t_proc, NULL, processing_thread_func, &data);
    pthread_create(&t_out, NULL, output_thread_func, &data);

    pthread_join(t_out, NULL);
    q1.shutdown(); 
    pthread_join(t_in, NULL);
    pthread_join(t_proc, NULL);

    cout << "[Main Thread] Pipeline shutdown complete." << endl;
    return 0;
}




