#include 	"stdio.h"
#include 	"ipp.h"
#include 	"x86intrin.h"
#include	<iostream>
#include	<iomanip>

using namespace std;

const int INPUT_SIZE = 8;
const int HIDDEN_SIZE = 16;
const int OUTPUT_SIZE = 8;

void print_list(float* list, int size)
{
	std::cout << std::fixed;      
    std::cout << std::setprecision(3);
	cout << "[";
	for (int i = 0; i < size; i++)
		cout << list[i] << " ,";
	cout << "]\n";
}

void print_matrix(float** matrix, int row, int col)
{
	cout << "{\n";
	for (int i = 0; i < row; i++)
		print_list(matrix[i], col);
	cout << "}\n";
}

void free_mat(float** matrix, int row)
{
	for (int i = 0; i < row; i++)
		free(matrix[i]);
}

float* create_list(int size, float base=0, float coef = 0)
{
	float *list = new float[size];
	for (int i = 0; i < size; i++)
		list[i] = base + (i+1) * coef;
	return list;
}

float** create_matrix(int row, int col)
{
	float **mat = new float*[row];
	for (int i = 0; i < row; i++)
		mat[i] = create_list(col, 0, 0.1);
	return mat;
}

class Layer
{
private:
	float** weights;
	float bais;
	int input_size;
	int output_size;
	float* out;

public:
	Layer(int input_size, int output_size)
	{
		this->input_size = input_size;
		this->output_size = output_size;
		this->out = new float[output_size];
	}

	~Layer()
	{
		free(out);
	}

	void set_weights(float** weights, float bais)
	{
		this->weights = weights;
		this->bais = bais;
	}

	float ReLU(float x)
	{
		return (0 < x) ? x : 0;
	}

	float* forward(float* x)
	{
		for (int i = 0; i < output_size; i++)
			out[i] = bais;
			
		for (int i = 0; i < output_size; i++)
			for (int j = 0; j < INPUT_SIZE; j++)
				out[i] += x[j] * weights[i][j];

		for (int i = 0; i < output_size; i++)
			out[i] = ReLU(out[i]);

		return out;
	}
};

int main(void)
{
	// auto x = generate_random_input(INPUT_SIZE);
	// x = layer(x, HIDDEN_SIZE);
	// x = layer(x, HIDDEN_SIZE);
	// x = layer(x, OUTPUT_SIZE);

	float *input = create_list(INPUT_SIZE, 0, 0.1);
	cout << "input: ";	
	print_list(input, INPUT_SIZE);
	
	float** weights_l1 = create_matrix(HIDDEN_SIZE, INPUT_SIZE);
	cout << "weight layer 1: \n";	
	print_matrix(weights_l1, HIDDEN_SIZE, INPUT_SIZE);
	
	float** weights_l2 = create_matrix(HIDDEN_SIZE, HIDDEN_SIZE);
	cout << "weight layer 2: \n";	
	print_matrix(weights_l2, HIDDEN_SIZE, HIDDEN_SIZE);
	
	float** weights_l3 = create_matrix(OUTPUT_SIZE, HIDDEN_SIZE);
	cout << "weight layer 3: \n";	
	print_matrix(weights_l3, OUTPUT_SIZE, HIDDEN_SIZE);
	
	// float *out_l1 = create_list(HIDDEN_SIZE);
	// float *out_l2 = create_list(HIDDEN_SIZE);
	// float *out_l3 = create_list(OUTPUT_SIZE);
	
	float bais_l1 = 1;
	float bais_l2 = 2;
	float bais_l3 = 3;
	
	
	// for (int i = 0; i < HIDDEN_SIZE; i++)
	// 	out_l1[i] = bais_l1;
		
	// for (int i = 0; i < HIDDEN_SIZE; i++)
	// 	for (int j = 0; j < INPUT_SIZE; j++)
	// 		out_l1[i] += input[j] * weights_l1[i][j];

	// cout << "out_l1: ";	
	// print_list(out_l1, HIDDEN_SIZE);
	
	Layer l1(INPUT_SIZE, HIDDEN_SIZE);
	l1.set_weights(weights_l1, bais_l1);
	float* out_l1 = l1.forward(input);
	cout << "out_l1: ";
	print_list(out_l1, HIDDEN_SIZE);
	
	Layer l2(HIDDEN_SIZE, HIDDEN_SIZE);
	l2.set_weights(weights_l2, bais_l2);
	float* out_l2 = l2.forward(out_l1);
	cout << "out_l2: ";
	print_list(out_l2, HIDDEN_SIZE);

	Layer l3(HIDDEN_SIZE, OUTPUT_SIZE);
	l3.set_weights(weights_l3, bais_l3);
	float* out_l3 = l3.forward(out_l2);
	cout << "out_l3: ";
	print_list(out_l3, OUTPUT_SIZE);
	
	free(input);
	free_mat(weights_l1, HIDDEN_SIZE);
	free_mat(weights_l2, HIDDEN_SIZE);
	free_mat(weights_l3, OUTPUT_SIZE);
	return 0;
}