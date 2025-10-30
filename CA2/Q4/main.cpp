#include "stdio.h"
#include "ipp.h"
#include "x86intrin.h"
#include <iostream>
#include <iomanip>

using namespace std;

const int INPUT_SIZE = 8;
const int HIDDEN_SIZE = 16;
const int OUTPUT_SIZE = 8;

void print_list(float *list, int size)
{
	std::cout << std::fixed;
	std::cout << std::setprecision(3);
	cout << "[";
	for (int i = 0; i < size; i++)
		cout << list[i] << " ,";
	cout << "]\n";
}

void print_matrix(float **matrix, int row, int col)
{
	cout << "{\n";
	for (int i = 0; i < row; i++)
		print_list(matrix[i], col);
	cout << "}\n";
}

void free_mat(float **matrix, int row)
{
	for (int i = 0; i < row; i++)
		free(matrix[i]);
}

float *create_list(int size, float base = 0, float coef = 0)
{
	float *list = (float *)_mm_malloc(sizeof(float) * size, sizeof(float) * 8);
	for (int i = 0; i < size; i++)
		list[i] = base + (i + 1) * coef;
	return list;
}

float **create_matrix(int row, int col)
{
	float **mat = (float **)_mm_malloc(sizeof(float *) * row, sizeof(float *) * 8);
	for (int i = 0; i < row; i++)
		mat[i] = create_list(col, 0, 0.1);
	return mat;
}

class Layer
{
protected:
	float **weights;
	float bias;
	int input_size;
	int output_size;
	float *out;

public:
	Layer(int input_size, int output_size) : input_size(input_size), output_size(output_size), weights(nullptr), out(nullptr)
	{
		this->out = (float *)_mm_malloc(sizeof(float) * output_size, sizeof(float) * 8);
	}

	void set_weights(float **weights, float bias)
	{
		this->weights = weights;
		this->bias = bias;
	}

	virtual float *forward(float *x) = 0;
};

class SerialLayer : public Layer
{
private:
	float ReLU(float x)
	{
		return (0.0 < x) ? x : 0.0;
	}

public:
	SerialLayer(int input_size, int output_size)
		: Layer(input_size, output_size) {}

	float *forward(float *x)
	{
		for (int i = 0; i < output_size; i++)
			out[i] = bias;

		for (int i = 0; i < output_size; i++)
			for (int j = 0; j < input_size; j++)
				out[i] += x[j] * weights[i][j];

		for (int i = 0; i < output_size; i++)
			out[i] = ReLU(out[i]);

		return out;
	}
};

class ParallelLayer : public Layer
{
private:
public:
	ParallelLayer(int input_size, int output_size)
		: Layer(input_size, output_size) {}

	float *forward(float *x) override
	{
		__m256 mac, x_tmp, w_tmp;
		__m128 sum, high, low, relu;
		for (int i = 0; i < output_size; i++)
		{
			mac = _mm256_set1_ps(0.0);
			for (int j = 0; j < input_size; j += 8)
			{
				x_tmp = _mm256_loadu_ps(&x[j]);
				w_tmp = _mm256_loadu_ps(&weights[i][j]);
				mac = _mm256_fmadd_ps(x_tmp, w_tmp, mac);
			}

			// mac = [a0, a1, a2, a3, a4, a5, a6, a7]
			low = _mm256_castps256_ps128(mac);
			high = _mm256_extractf128_ps(mac, 1);
			// sum = [a0+a4, a1+a5, a2+a6, a3+a7]
			sum = _mm_add_ps(low, high);

			// sum = [a0+a4+a1+a5, a2+a6+a3+a7, a0+a4+a1+a5, a2+a6+a3+a7]
			sum = _mm_hadd_ps(sum, sum);
			// sum = [sum(0,7), sum(0,7), sum(0,7), sum(0,7)]
			sum = _mm_hadd_ps(sum, sum);
			// sum = [sum(0,7)+bias, sum(0,7), sum(0,7), sum(0,7)]
			sum = _mm_add_ss(sum, _mm_set_ss(bias));

			relu = _mm_max_ss(sum, _mm_setzero_ps());

			out[i] = _mm_cvtss_f32(relu);
		}

		return out;
	}
};

int main(void)
{
	float *input = create_list(INPUT_SIZE, 0, 0.1);
	cout << "input: ";
	print_list(input, INPUT_SIZE);

	float **weights_l1 = create_matrix(HIDDEN_SIZE, INPUT_SIZE);
	cout << "weight layer 1: \n";
	print_matrix(weights_l1, HIDDEN_SIZE, INPUT_SIZE);

	float **weights_l2 = create_matrix(HIDDEN_SIZE, HIDDEN_SIZE);
	cout << "weight layer 2: \n";
	print_matrix(weights_l2, HIDDEN_SIZE, HIDDEN_SIZE);

	float **weights_l3 = create_matrix(OUTPUT_SIZE, HIDDEN_SIZE);
	cout << "weight layer 3: \n";
	print_matrix(weights_l3, OUTPUT_SIZE, HIDDEN_SIZE);

	float bais_l1 = 1;
	float bais_l2 = 2;
	float bais_l3 = 3;
	{
		float *x = input;

		SerialLayer l1(INPUT_SIZE, HIDDEN_SIZE);
		l1.set_weights(weights_l1, bais_l1);
		x = l1.forward(x);
		cout << "out_l1: ";
		print_list(x, HIDDEN_SIZE);

		SerialLayer l2(HIDDEN_SIZE, HIDDEN_SIZE);
		l2.set_weights(weights_l2, bais_l2);
		x = l2.forward(x);
		cout << "out_l2: ";
		print_list(x, HIDDEN_SIZE);

		SerialLayer l3(HIDDEN_SIZE, OUTPUT_SIZE);
		l3.set_weights(weights_l3, bais_l3);
		x = l3.forward(x);
		cout << "out_l3: ";
		print_list(x, OUTPUT_SIZE);
	}

	{
		float *x = input;

		ParallelLayer l1(INPUT_SIZE, HIDDEN_SIZE);
		l1.set_weights(weights_l1, bais_l1);
		x = l1.forward(x);
		cout << "out_l1: ";
		print_list(x, HIDDEN_SIZE);

		ParallelLayer l2(HIDDEN_SIZE, HIDDEN_SIZE);
		l2.set_weights(weights_l2, bais_l2);
		x = l2.forward(x);
		cout << "out_l2: ";
		print_list(x, HIDDEN_SIZE);

		ParallelLayer l3(HIDDEN_SIZE, OUTPUT_SIZE);
		l3.set_weights(weights_l3, bais_l3);
		x = l3.forward(x);
		cout << "out_l3: ";
		print_list(x, OUTPUT_SIZE);
	}

	return 0;
}