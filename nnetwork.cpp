#include "nnetwork.h"
#include <random>
#include <iostream>

/*
-------------- Neuron Implementation --------------
*/

neuron::neuron(int numweights)
{
    // seed random number generator
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(0, 1);

    // add weights
    for(int i = 0; i < numweights; i++)
    {
        weights.push_back(dist(mt));
    }
}

neuron::neuron(std::vector<float> newWeights)
{
    weights = newWeights;
}

// returns the weights vector
std::vector<float> neuron::getWeights()
{
    return weights;
}

// calculates and returns the activation of a neuron based on its input
// activation = sum(weights[i] * input[i]) + bias (for all i)
float neuron::activation(std::vector<float> input)
{
    // get bias
    float bias = weights[weights.size() - 1];

    // sum the weights*inputs
    float activation = bias;
    for(size_t i = 0; i < input.size(); i++)
    {
        activation += weights[i] * input[i];
    }

    return activation; // left off here
}

// sets the output of a neuron
void neuron::setOutput(float output)
{
    this->output = output;
}

// returns the output of a neuron
float neuron::getOutput()
{
    return output;
}

/*
-------------- Layer Implementation --------------
*/

// constructor for layer that randomly generates neurons
layer::layer(int numneurons, int numweights)
{
    // create hidden layer
    for(int i = 0; i < numneurons; i++)
    {  
        neurons.push_back(neuron(numweights)); // add weights for each node + bias
    }
}

// returns the neurons array
std::vector<neuron> layer::getNeurons()
{
    return neurons;
}

void layer::setNeurons(std::vector<neuron> newNeurons)
{
    this->neurons = neurons;
}

/*
-------------- Neural Network Implementation --------------
*/

neuralnetwork::neuralnetwork(int numinput, int numhidden, int numoutput)
{
    // create hidden layer
    layer hiddenLayer(numhidden, numinput + 1);

    // create output layer
    layer outputLayer(numoutput, numhidden + 1);

    // add em to network
    network.push_back(hiddenLayer);
    network.push_back(outputLayer);
}

// creates network from given layers (if you have those for whatever reason lol)
neuralnetwork::neuralnetwork(std::vector<layer> layers)  : network(layers){}

// create network from vector of vector of neurons
// note : must be stored in order from input to output layer
neuralnetwork::neuralnetwork(std::vector<std::vector<neuron>> neurons)
{
    // construct layer from vector of neurons
    for(std::vector<neuron> neuralLayer : neurons)
    {
        // create layer
        layer newLayer(neuralLayer);

        // add to network
        network.push_back(newLayer);
    }
}

// tabularly prints the current network
void neuralnetwork::print()
{
    int layernum = 0;
    for(layer l : network)
    {
        std::cout << "Layer: " << layernum << std::endl;
        layernum++;
        int neuronnum = 0;
        for(neuron n : l.getNeurons())
        {
            std::cout << "  Neuron: " << neuronnum << std::endl;
            neuronnum++;
            for(size_t i = 0; i < n.getWeights().size(); i++)
            {
                std::cout << "    " << n.getWeights()[i] << std::endl;
            }
        }
    }
}

// transfers an activation function using the sigmoid function
// sigmoid : output = 1 / (1 + e^(-activation))
float neuralnetwork::transfer(float activation)
{
    return 1.0 / (1.0 + std::pow(M_e, -1*activation));
}

std::vector<float> neuralnetwork::forwardPropogate(std::vector<float> inputs)
{
    // start propogation from layer 1
    for(layer currLayer : network)
    {
        std::vector<float> nextInputs;
        for(neuron currNeuron : currLayer.getNeurons())
        {
            // calculate the transfer for the given neuron
            float transferInput = transfer(currNeuron.activation(inputs));

            // store neuron output for the backpropogation algo
            currNeuron.setOutput(transferInput);

            // add to next layer of inputs
            nextInputs.push_back(transferInput);
        }
        // make inputs = the new ones
        inputs = nextInputs;

        // clear nextInputs so we can rewrite it next time
        nextInputs.clear();
    }
    
    // after going thru all layers, inputs should be the transfer of the last layer of neurons
    return inputs;
}