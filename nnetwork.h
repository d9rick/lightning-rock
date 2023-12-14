#ifndef NNETWORK_H
#define NNETWORK_H

#include <vector>
#define private public

// class to hold a neuron, along with its weights
class neuron
{
private:
    std::vector<int> weights;

public:
    neuron(int numweights);
};

/**
 * Layer Class to hold layers of neurons
*/
class layer
{
private:
    std::vector<neuron> neurons;

public:
    layer(int numneurons, int numweights);
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
};

#endif