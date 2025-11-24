#include "stdio.h"
#include "x86intrin.h"
#include "stdlib.h"
#include "time.h"
#include <sys/time.h>
#include "immintrin.h"
#include "omp.h"

#define	VECTOR_SIZE	16000000

int main() {
    struct timeval start, end;
	long seconds, micros, time_serial, time_simd, time_omp, time_omp_simd;

    float sum_temp[8];
    float value_temp;
	float mean_serial, var_serial;
	float mean_simd, var_simd;
	float mean_omp, var_omp;
	float mean_omp_simd, var_omp_simd;

    float *vector;
	vector = new float [VECTOR_SIZE];

    if (!vector) {
		printf("Memory allocation error!\n");
		return 1;
	}

    srand(10);
	for (long i = 0; i < VECTOR_SIZE; i++) {
		vector[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/100.0));
	}


    // SERIAL PART:

    gettimeofday(&start, NULL);
    sum_temp[0] = sum_temp[1] = sum_temp[2] = sum_temp[3] = 0.0;
    sum_temp[4] = sum_temp[5] = sum_temp[6] = sum_temp[7] = 0.0;

    for (long i = 0; i < VECTOR_SIZE; i+=8) {
		sum_temp[0] += vector[i+0];
        sum_temp[1] += vector[i+1];
        sum_temp[2] += vector[i+2];
        sum_temp[3] += vector[i+3];
        sum_temp[4] += vector[i+4];
        sum_temp[5] += vector[i+5];
        sum_temp[6] += vector[i+6];
        sum_temp[7] += vector[i+7];
    }

    sum_temp[0] = sum_temp[0] + sum_temp[4];
    sum_temp[1] = sum_temp[1] + sum_temp[5];
    sum_temp[2] = sum_temp[2] + sum_temp[6];
    sum_temp[3] = sum_temp[3] + sum_temp[7];
    mean_serial = ((sum_temp[0] + sum_temp[1]) + (sum_temp[2] + sum_temp[3])) / VECTOR_SIZE;


    sum_temp[0] = sum_temp[1] = sum_temp[2] = sum_temp[3] = 0.0;
    sum_temp[4] = sum_temp[5] = sum_temp[6] = sum_temp[7] = 0.0;

    for (long i = 0; i < VECTOR_SIZE; i+=8) {
        value_temp = vector[i+0] - mean_serial;
		sum_temp[0] += value_temp * value_temp;

        value_temp = vector[i+1] - mean_serial;
        sum_temp[1] += value_temp * value_temp;

        value_temp = vector[i+2] - mean_serial;
        sum_temp[2] += value_temp * value_temp;

        value_temp = vector[i+3] - mean_serial;
        sum_temp[3] += value_temp * value_temp;

        value_temp = vector[i+4] - mean_serial;
        sum_temp[4] += value_temp * value_temp;

        value_temp = vector[i+5] - mean_serial;
        sum_temp[5] += value_temp * value_temp;

        value_temp = vector[i+6] - mean_serial;
        sum_temp[6] += value_temp * value_temp;

        value_temp = vector[i+7] - mean_serial;
        sum_temp[7] += value_temp * value_temp;
    }

    sum_temp[0] = sum_temp[0] + sum_temp[4];
    sum_temp[1] = sum_temp[1] + sum_temp[5];
    sum_temp[2] = sum_temp[2] + sum_temp[6];
    sum_temp[3] = sum_temp[3] + sum_temp[7];
    var_serial = ((sum_temp[0] + sum_temp[1]) + (sum_temp[2] + sum_temp[3])) / VECTOR_SIZE;

    gettimeofday(&end, NULL);
	seconds = (end.tv_sec - start.tv_sec);
	time_serial = micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);


    // SIMD PARALLEL PART:

    gettimeofday(&start, NULL);
    __m256 vec, diff, squared;
    __m256 sum_vec = _mm256_set1_ps(0.0);
    for (long i = 0; i < VECTOR_SIZE; i += 8) {
        vec = _mm256_load_ps(&vector[i]);
        sum_vec = _mm256_add_ps(sum_vec, vec);
    }
    __m128 sum_vec_low = _mm256_castps256_ps128(sum_vec);
    __m128 sum_vec_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_vec_128 = _mm_add_ps(sum_vec_low, sum_vec_high);
    sum_vec_128 = _mm_hadd_ps(sum_vec_128, sum_vec_128);
    sum_vec_128 = _mm_hadd_ps(sum_vec_128, sum_vec_128);
    mean_simd = _mm_cvtss_f32(sum_vec_128) / VECTOR_SIZE;

    __m256 mean_vec = _mm256_set1_ps(mean_simd);
    __m256 var_vec = _mm256_set1_ps(0.0);
    for (long i = 0; i < VECTOR_SIZE; i += 8) {
        vec = _mm256_load_ps(&vector[i]);
        diff = _mm256_sub_ps(vec, mean_vec);
        squared = _mm256_mul_ps(diff, diff);
        var_vec = _mm256_add_ps(var_vec, squared);
    }

    __m128 var_vec_low = _mm256_castps256_ps128(var_vec);
    __m128 var_vec_high = _mm256_extractf128_ps(var_vec, 1);
    __m128 var_vec_128 = _mm_add_ps(var_vec_low, var_vec_high);
    var_vec_128 = _mm_hadd_ps(var_vec_128, var_vec_128);
    var_vec_128 = _mm_hadd_ps(var_vec_128, var_vec_128);
    var_simd = _mm_cvtss_f32(var_vec_128) / VECTOR_SIZE;

    gettimeofday(&end, NULL);
	seconds = (end.tv_sec - start.tv_sec);
	time_simd = micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);


    // OMP PARALLEL PART:

    gettimeofday(&start, NULL);
    sum_temp[0] = sum_temp[1] = sum_temp[2] = sum_temp[3] = 0.0;
    sum_temp[4] = sum_temp[5] = sum_temp[6] = sum_temp[7] = 0.0;

    #pragma omp parallel for reduction(+:sum_temp[:8])
    for (long i = 0; i < VECTOR_SIZE; i+=8) {
		sum_temp[0] += vector[i+0];
        sum_temp[1] += vector[i+1];
        sum_temp[2] += vector[i+2];
        sum_temp[3] += vector[i+3];
        sum_temp[4] += vector[i+4];
        sum_temp[5] += vector[i+5];
        sum_temp[6] += vector[i+6];
        sum_temp[7] += vector[i+7];
    }

    sum_temp[0] = sum_temp[0] + sum_temp[4];
    sum_temp[1] = sum_temp[1] + sum_temp[5];
    sum_temp[2] = sum_temp[2] + sum_temp[6];
    sum_temp[3] = sum_temp[3] + sum_temp[7];
    mean_omp = ((sum_temp[0] + sum_temp[1]) + (sum_temp[2] + sum_temp[3])) / VECTOR_SIZE;

    sum_temp[0] = sum_temp[1] = sum_temp[2] = sum_temp[3] = 0.0;
    sum_temp[4] = sum_temp[5] = sum_temp[6] = sum_temp[7] = 0.0;

    #pragma omp parallel for reduction(+:sum_temp[:8]) private(value_temp)
    for (long i = 0; i < VECTOR_SIZE; i+=8) {
        value_temp = vector[i+0] - mean_omp;
		sum_temp[0] += value_temp * value_temp;

        value_temp = vector[i+1] - mean_omp;
        sum_temp[1] += value_temp * value_temp;

        value_temp = vector[i+2] - mean_omp;
        sum_temp[2] += value_temp * value_temp;

        value_temp = vector[i+3] - mean_omp;
        sum_temp[3] += value_temp * value_temp;

        value_temp = vector[i+4] - mean_omp;
        sum_temp[4] += value_temp * value_temp;

        value_temp = vector[i+5] - mean_omp;
        sum_temp[5] += value_temp * value_temp;

        value_temp = vector[i+6] - mean_omp;
        sum_temp[6] += value_temp * value_temp;

        value_temp = vector[i+7] - mean_omp;
        sum_temp[7] += value_temp * value_temp;
    }

    sum_temp[0] = sum_temp[0] + sum_temp[4];
    sum_temp[1] = sum_temp[1] + sum_temp[5];
    sum_temp[2] = sum_temp[2] + sum_temp[6];
    sum_temp[3] = sum_temp[3] + sum_temp[7];
    var_omp = ((sum_temp[0] + sum_temp[1]) + (sum_temp[2] + sum_temp[3])) / VECTOR_SIZE;

    gettimeofday(&end, NULL);
    seconds = (end.tv_sec - start.tv_sec);
    time_omp = micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);


    // SIMD OMP PARALLEL PART:

    gettimeofday(&start, NULL);
    
    float global_sum_omp_simd = 0;

    #pragma omp parallel 
    {
        __m256 vec_local_sum = _mm256_setzero_ps();

        #pragma omp for
        for (long i = 0; i < VECTOR_SIZE; i += 8) {
            __m256 vec_chunk = _mm256_load_ps(&vector[i]);
            vec_local_sum = _mm256_add_ps(vec_local_sum, vec_chunk);
        }

        __m128 low = _mm256_castps256_ps128(vec_local_sum);
        __m128 high = _mm256_extractf128_ps(vec_local_sum, 1);
        __m128 combined = _mm_add_ps(low, high);
        combined = _mm_hadd_ps(combined, combined);
        combined = _mm_hadd_ps(combined, combined);
        float scalar_sum = _mm_cvtss_f32(combined);

        #pragma omp atomic
        global_sum_omp_simd += scalar_sum;
    }
    mean_omp_simd = global_sum_omp_simd / VECTOR_SIZE;


    float global_var_sum_omp_simd = 0;
    
    #pragma omp parallel
    {
        __m256 mean_vec_local = _mm256_set1_ps(mean_omp_simd);
        __m256 var_vec_local = _mm256_setzero_ps();
        
        #pragma omp for
        for (long i = 0; i < VECTOR_SIZE; i += 8) {
            __m256 vec_chunk = _mm256_load_ps(&vector[i]);
            __m256 diff_chunk = _mm256_sub_ps(vec_chunk, mean_vec_local);
            __m256 sq_chunk = _mm256_mul_ps(diff_chunk, diff_chunk);
            var_vec_local = _mm256_add_ps(var_vec_local, sq_chunk);
        }

        __m128 low = _mm256_castps256_ps128(var_vec_local);
        __m128 high = _mm256_extractf128_ps(var_vec_local, 1);
        __m128 combined = _mm_add_ps(low, high);
        combined = _mm_hadd_ps(combined, combined);
        combined = _mm_hadd_ps(combined, combined);
        float scalar_var = _mm_cvtss_f32(combined);

        #pragma omp atomic
        global_var_sum_omp_simd += scalar_var;
    }
    var_omp_simd = global_var_sum_omp_simd / VECTOR_SIZE;


    gettimeofday(&end, NULL);
    seconds = (end.tv_sec - start.tv_sec);
    time_omp_simd = micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);


    // Results:

    printf("The Serial Result              = %.4f\n", var_serial);
	printf("The SIMD parallel Result       = %.4f\n", var_simd);
	printf("The OMP Parallel Result        = %.4f\n", var_omp);
	printf("The OMP & SIMD Parallel Result = %.4f\n", var_omp_simd);

    printf("====================================================\n");
    
	printf("Serial Run time                = %ld \n", time_serial);
	printf("SIMD Parallel Run time         = %ld \n", time_simd);
	printf("OMP Parallel Run time          = %ld \n", time_omp);
	printf("OMP & SIMD Parallel Run time   = %ld \n", time_omp_simd);

    printf("====================================================\n");

	printf("Speedup (SIMD vs Serial)       = %f\n", (double) (time_serial)/(double) time_simd);
	printf("Speedup (OMP vs Serial)        = %f\n", (double) (time_serial)/(double) time_omp);
	printf("Speedup (OMP & SIMD vs Serial) = %f\n", (double) (time_serial)/(double) time_omp_simd);
	printf("Speedup (OMP & SIMD vs SIMD)   = %f\n", (double) (time_simd)/(double) time_omp_simd);
	printf("Speedup (OMP & SIMD vs OMP)    = %f\n", (double) (time_omp)/(double) time_omp_simd);

    delete[] vector;
	return 0;
}

