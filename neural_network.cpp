#include "neural_network.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <exception>
#include "MatrixManipulation.h"
#include "MatrixManipulation.cpp"
using namespace std;

void display(vector<vector<double>> a)
{
    for (unsigned int i = 0; i < a.size(); i++)
    {
        for (unsigned int j = 0; j < a[i].size(); j++)
        {
            cout << i << ": " <<  a[i][j] * 100 << "%";
        }
        cout << endl;
    }
}
    // HELPER FUNCTIONS

    // Calculates error between the target and output layer
Vec find_error(Matrx m, Vec target)
{
    Vec err;
    for (unsigned int i = 0; i < target.size(); i++)
    {
        err.push_back(target[i] - m[i][0]); // error = target - output
    }
    return err;
}

    // After reading MINIST file line by line as a string, pass a line and rescalre target
Vec rescale_target(string a_set)
{
    Vec new_target;
    unsigned int target;
    stringstream ss(a_set);
    ss >> target;
    for (unsigned int i = 0; i < 10; i++)
    {
        // One-hot encoding
        if (i != target)
        {
            new_target.push_back(0.01);
        }
        else
        {
            new_target.push_back(0.99); // Fire 0.99 for the actuall target
        }
    }
    return new_target;
}
    // After reading MINIST file line by line as a string, pass a line and rescalre input
Matrx rescale_input(string train_set)
{
    Matrx m;
    stringstream ss(train_set);
    double num;
    ss >> num;
    ss.ignore();    // Ignore the actual target (first column), we only need the input.
    const int NUMB_RANGE = 255;
    const double RANGE = 0.99;
    while(ss >> num)
    {
        ss.ignore();
        num = (((double) (num / NUMB_RANGE)) * RANGE) + 0.01; // Rescale
        Vec vec = {num};
        m.push_back(vec);
    }
    return m;
}
    // Creates a random Matrix
Matrx new_random_matrix(unsigned int row, unsigned int col)
{
    Matrx m;
    for (unsigned int i = 0; i < row; i++)
    {
        m.push_back({});
        for (unsigned int j = 0; j < col; j++)
        {
            double high = 1 / sqrt(row); // The upper limit of random (decided by number of neurons = row)
            double low = high * -1;     // The lower limit of random
            double temp = static_cast<double> (rand()) * (high - low) / static_cast<double>(RAND_MAX) + low;
            m[i].push_back( temp );
        }
    }
    return m;
}
    // Sets up the weight layers
Layer set_up_layer_weight(int in, int hid, int out, int layer)
{
    Layer weight_layer;
    srand(time(NULL));
    unsigned int numb_of_weight_Matrx = layer - 1;
    int row = hid;
    int col = in;
    for (unsigned int i = 0; i < numb_of_weight_Matrx - 1; i++)
    {
        weight_layer.push_back(new_random_matrix(row, col)); // Weights for hidden layers
        col = row;
        row = hid / (layer - i);        // Self-created formula, used to determine how many neurons in layers
        row = (row > out ? row : out);  // although it only works best for 3 layers (input, hidden and output)
    }
    int len = weight_layer.size();
    row = weight_layer[len-1].size();
    weight_layer.push_back(new_random_matrix(out, row)); // Weight for input layer
    return weight_layer;
}

    // Read file line by line as string
vector<string> read_file(string file_name)
{
    string line;
    vector<string> vec;
    ifstream myfile(file_name);
    if (myfile.is_open())
    {
        while (getline(myfile, line))
        {
            vec.push_back(line);
        }
        myfile.close();
    }
    return vec;
}

    // Private functions

    // Sigmoid function
double neuralNetwork::sigmoid(double x)
{
    double ex = exp(x);
    double n = 1;
    return n / ( n + (n / ex ));
}
    // Applys sigmoid function for all of the element of a matrx
void neuralNetwork::update_sigmoid(Matrx& m)
{
    for (unsigned int i = 0; i < m.size(); i++)
    {
        for (unsigned int j = 0; j < m[i].size(); j++)
        {
            m[i][j] = sigmoid(m[i][j]);
        }
    }
}
    // Feed forward by calculating dot_product: Weight * layers
Layer neuralNetwork::feed_forward(Matrx an_input_set)
{
    Layer v;
    unsigned int i = 0;
    Matrx temp;
    temp = weight_layer[i] * an_input_set; // User * operator for dot_product
    update_sigmoid(temp);  // Applys sigmoid function after dot_product
    v.push_back(temp);
    i++;
    for (; i < weight_layer.size(); i++)
    {
        temp = weight_layer[i] * temp; // Takes output of hidden layer as input for the nextlayer
        update_sigmoid(temp);
        v.push_back(temp);
    }
    return v;
}

    // Calculates error after feed forward
Layer neuralNetwork::update_error(Vec target, Layer output_layer)
{
    Layer error_layer;
    int len = output_layer.size();
    Vec err = find_error(output_layer[len-1], target); // Finds error between actual target and output
    error_layer.push_back(MatrixManipulation<double>().tranpose(err));
    for (unsigned int i = len - 1; i > 0; i--)
    {
            // Hidden Error layer = (Weight hidden)^T * erro_output_layer
        Matrx error = MatrixManipulation<double>().tranpose(weight_layer[i]) * error_layer[0];
        error_layer.insert(error_layer.begin(), error);
    }
    return error_layer;
}

    // Backpropagation
Matrx neuralNetwork::backpropagate(Matrx E, Matrx S, Matrx O, double learn_rate)
{
    Matrx m;
    for (unsigned int i = 0; i < S.size(); i++)
    {
            // (t - o) * sigmoid (1 - sigmoid) * learning_rate
        double temp = S[i][0] * (1 - S[i][0]) * E[i][0] * learn_rate;
        Vec v = {temp};
        m.push_back(v);

    }
        // delta W = E * sigmoid (1 - sigmoid) * o^T * learning_rate
    m = m * MatrixManipulation<double>().tranpose(O);
    return m;
}
    // Calculates backward and refine weight
void neuralNetwork::backward(Vec target, Matrx input_layer, Layer output_layer, Layer error_layer)
{
    Matrx m;
    for (unsigned int i = weight_layer.size() - 1; i > 0; i--)
    {
        m = backpropagate(error_layer[i], output_layer[i], output_layer[i-1], this->learn_rate);
        weight_layer[i] = weight_layer[i] + m; // Weight = weight + delta weight

    }
    m = backpropagate(error_layer[0], output_layer[0], input_layer, this->learn_rate);
    weight_layer[0] = weight_layer[0] + m; // Weight = weight + delta weight

}
    // After reading the MINIST file line by line as string, pass it in and begin learning
void neuralNetwork::learning(vector<string> train_sets)
{
    for (unsigned int i = 0; i < train_sets.size(); i++)
    {
        Layer output_layer;
        Layer error_layer;
        Matrx input_layer = rescale_input(train_sets[i]);
        Vec target = rescale_target(train_sets[i]);
        output_layer = feed_forward(input_layer);
        error_layer = update_error(target, output_layer);
        backward(target, input_layer, output_layer, error_layer);
    }
 }

    //Public functions

    // Constructor
neuralNetwork::neuralNetwork(int in_nodes, int hid_nodes, int out_nodes, int layer, int epochs, double lr)
{
    initialize(in_nodes, hid_nodes, out_nodes, layer, epochs, lr);
}

    // Initializing
void neuralNetwork::initialize(int in_nodes, int hid_nodes, int out_nodes, int layer, int epochs, double lr)
{
    this->in_nodes = in_nodes;
    this->hid_nodes = hid_nodes;
    this->out_nodes = out_nodes;
    this->layers = layer;
    this->learn_rate = lr;
    this->epoch = epochs;
    weight_layer = set_up_layer_weight(in_nodes, hid_nodes, out_nodes, layer);

}

    // Takes the name of a train set and begin to train
void neuralNetwork::train(string train_set_file_name)
{
    vector<string> train_sets = read_file(train_set_file_name);
    for (unsigned int i = 0; i < this->epoch; i++)
    {
        learning(train_sets);
    }
}

    // Takes the name of a test set and begin predict and evaluate
void neuralNetwork::query(string test_set)
{

    Matrx m;
    vector<string> a_set = read_file(test_set);
    double accuracy = 0;
    for (unsigned int i = 0; i < a_set.size(); i++)
    {
        int index = a_set[i][0] - '0';
        m = rescale_input(a_set[i]);
        Layer output_layer = feed_forward(m);
        int len = output_layer.size();
        accuracy += output_layer[len-1][index][0];
    }
    cout << "Accuracy: " << accuracy * 100 / a_set.size() << "%";

}
neuralNetwork::~neuralNetwork()
{

}
