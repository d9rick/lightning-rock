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
public:
    // neuron constructor that randomly defines weights
    neuron(int numweights);

    // neuron constructor when given predetermined weights
    neuron(std::vector<float> predeterminedWeights);

    // returns the vector of weights
    std::vector<float> getWeights();

    // returns the activation of a neuron given an input
    float activation(std::vector<float> input);

    // sets the next output
    void setOutput(float output);

    float getOutput();
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
    layer(std::vector<neuron> neurons) : neurons(neurons){}
    
    // returns the neurons array
    std::vector<neuron> getNeurons();

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

public:
    // initalizes the neuralnetwork
    neuralnetwork(int numinputs, int numhidden, int numoutputs);

    // initializes network when given layers
    neuralnetwork(std::vector<layer> layers);

    // initializes network when given vector of vector of neurons
    neuralnetwork(std::vector<std::vector<neuron>> neurons);

    // prints the neural network
    void print();

    // calculates the transfer value when given the activation value of a neuron
    float transfer(float activation);

    // propogates inputs from input layer to output layer, and returns their output
    std::vector<float> forwardPropogate(std::vector<float> inputs);
};

#endif