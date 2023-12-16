#ifndef NNETWORK_H
#define NNETWORK_H

#include <vector>

#define M_e 2.718281

// class to hold an artificial neuron, along with its weights
// for the sake of simplicity, the bias will always be the last member of the weights array
class neuron
{
private:
    // stores the weights and bias for the neuron
    std::vector<float> weights;

    // stores the neurons current output
    float output;

    // stores error for a neuron, to be used in backpropogation (also referred to as "delta")
    float error = 0;

public:
    // neuron constructor that randomly defines weights
    neuron(int numweights);

    // neuron constructor when given predetermined weights
    neuron(std::vector<float> predeterminedWeights);

    // returns the vector of weights
    std::vector<float>& getWeights();

    // returns the activation of a neuron given an input
    float activation(std::vector<float> input);

    // returns the stored output
    float getOutput();

    // sets the next output
    void setOutput(float output);

    // calculates and returns transfer derivative
    float transferDerivative();

    // returns the stored error of a neuron
    float getError();

    // sets the error of a given neuron
    void setError(float error);

    // calculates the error based on the passed expected value
    void calculateError(float expected);
};

/**
 * Layer Class to hold layers of neurons
 */
class layer
{
private:
    // stores the neurons for a given layer
    std::vector<neuron> neurons;

public:
    // constructor for layer that randomly generates neurons
    layer(int numneurons, int numweights);

    // creates layer from neurons
    layer(std::vector<neuron> neurons) : neurons(neurons) {}

    // returns the neurons array
    std::vector<neuron>& getNeurons();

    // returns the number of neurons in the layer
    int numNeurons() const;

    // sets the current layer's neurons
    void setNeurons(std::vector<neuron> newNeurons);
};

/**
 * Neural Network Class
 * - Will store all nodes
 */
class neuralnetwork
{
private:
    std::vector<layer> network;

    // calculates the transfer value when given the activation value of a neuron
    float transfer(float activation);

public:
    // initalizes the neuralnetwork
    neuralnetwork(int numinputs, int numhidden, int numoutputs);

    // initializes network when given layers
    neuralnetwork(std::vector<layer> layers);

    // initializes network when given vector of vector of neurons
    neuralnetwork(std::vector<std::vector<neuron>> neurons);

    // prints the neural network
    void print();

    // propogates inputs from input layer to output layer, and returns their output
    std::vector<float> forwardPropogate(std::vector<float> inputs);

    // takes in the expected response for the network, and backpropogates the error to adjust the neurons
    void backwardPropogateError(std::vector<float> expected);
};

#endif