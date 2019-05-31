#ifndef NEURAL_NETWORK_H_INCLUDED
#define NEURAL_NETWORK_H_INCLUDED

#include <vector>
#include <string>
#include <fstream>

    //Alias
using Vec = std::vector<double>;
using Matrx = std::vector<Vec>;
using Layer = std::vector<Matrx>;

class neuralNetwork
{
private:
    // Variables of the class

    unsigned int in_nodes, hid_nodes, out_nodes, layers, epoch;
    double learn_rate;
    Layer weight_layer;

    // Private methods

        // Takes an input(Matrix of 28x28), and feed forward
    Layer feed_forward(Matrx input);

        // Calculates error and return error layer
    Layer update_error(Vec target, Layer output_layer);

        // Backpropagate, return the delta weight matrix
    Matrx backpropagate(Matrx E, Matrx S, Matrx O, double learn_rate);

        // Calculates backward and refine weight
    void backward(Vec target, Matrx a_train_set, Layer output, Layer error);

        // Signmoid function
    double sigmoid(double x);

        // Given a matrix, apply sigmoid function for all of its elements
    void update_sigmoid(Matrx& m);

        // After reading the MINIST file line by line as string, pass it in and begin learning
    void learning(std::vector<std::string> train_sets);

public:
        // Constructor
    neuralNetwork(int in_nodes, int hid_nodes, int out_nodes, int layer, int epochs, double lr);

        // Initialize all the variables
    void initialize(int in_nodes, int hid_nodes, int out_nodes, int layer, int epochs, double lr);

        // Takes the name of a train set and begin to train
    void train(std::string train_set_file_name);

        // Takes the name of a test_set and begin to predict
    void query(std::string test_set_file_name);

        // Destructor
    ~neuralNetwork();

};


#endif // NEURAL_NETWORK_H_INCLUDED
