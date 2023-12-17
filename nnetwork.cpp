#include "nnetwork.h"
#include <random>
#include <iostream>
#include <cmath>
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
    for (int i = 0; i < numweights; i++)
    {
        weights.push_back(dist(mt));
    }
}

neuron::neuron(std::vector<float> newWeights)
{
    weights = newWeights;
}

// returns the weights vector
std::vector<float> &neuron::getWeights()
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
    for (size_t i = 0; i < input.size() - 1; i++)
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
calculates transfer derivative
 since we are using the sigmoid function, derivative is calculated like:
 derivative = output * (1.0 - output)
*/
float neuron::transferDerivative()
{
    return output * (1.0 - output);
}

// gets stored error and returns it
float neuron::getError()
{
    return error;
}

// changes the stored error to the passed one
void neuron::setError(float newError)
{
    error = newError;
}
/*
    Calculates error based on passed expected output
error = (output - expected) * transfer_derivative(output)
*/
void neuron::calculateError(float expected)
{
    error = (output - expected) * transferDerivative();
}

/*
-------------- Layer Implementation --------------
*/

// constructor for layer that randomly generates neurons
layer::layer(int numneurons, int numweights)
{
    // create hidden layer
    for (int i = 0; i < numneurons; i++)
    {
        neurons.push_back(neuron(numweights)); // add weights for each node + bias
    }
}

// returns the neurons array
std::vector<neuron> &layer::getNeurons()
{
    return neurons;
}

void layer::setNeurons(std::vector<neuron> newNeurons)
{
    this->neurons = newNeurons;
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
neuralnetwork::neuralnetwork(std::vector<layer> layers) : network(layers) {}

// create network from vector of vector of neurons
// note : must be stored in order from input to output layer
neuralnetwork::neuralnetwork(std::vector<std::vector<neuron>> neurons)
{
    // construct layer from vector of neurons
    for (std::vector<neuron> neuralLayer : neurons)
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
    for (layer l : network)
    {
        std::cout << "Layer: " << layernum << std::endl;
        layernum++;
        int neuronnum = 0;
        for (neuron n : l.getNeurons())
        {
            std::cout << "  Neuron: " << neuronnum << std::endl;
            neuronnum++;
            for (size_t i = 0; i < n.getWeights().size(); i++)
            {
                std::cout << "    " << n.getWeights()[i] << std::endl;
            }
            std::cout << "    Error: " << n.getError() << std::endl;
            std::cout << "    Output: " << n.getOutput() << std::endl;
            std::cout << "    Transfer: " << n.transferDerivative() << std::endl;
        }
    }
}

// transfers an activation function using the sigmoid function
// sigmoid : output = 1 / (1 + e^(-activation))
float neuralnetwork::transfer(float activation)
{
    return 1.0f / (1.0f + std::exp(-activation));
}

std::vector<float> neuralnetwork::forwardPropogate(std::vector<float> inputs)
{
    // start propogation from layer 1
    for (size_t i = 0; i < network.size(); i++)
    {
        // get current layer
        std::vector<float> nextInputs;
        std::vector<neuron> &currNeurons = network[i].getNeurons();

        // iterate thru neurons in current layer
        for (size_t j = 0; j < currNeurons.size(); j++)
        {
            // calculate the transfer for the given neuron
            float transferInput = transfer(currNeurons[j].activation(inputs));

            // store neuron output for the backpropogation algo
            currNeurons[j].setOutput(transferInput);

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

/*
    Calculates the error for each neuron in each layer of the network
error of output neuron = (output - expected) * transfer_derivative(output)
error of hidden neuron = (weight_k * error_j) * transfer_derivative(output)
*/
void neuralnetwork::backwardPropogateError(std::vector<float> expected)
{
    // iterate through the layers backwards
    for (int i = network.size() - 1; i >= 0; i--)
    {
        // get current operating layer
        std::vector<neuron> &currLayer = network[i].getNeurons();

        // declare error array
        std::vector<float> errors = {};

        // calculate error for hidden layers
        if (i != (int)network.size() - 1)
        {
            for (size_t j = 0; j < currLayer.size(); j++)
            {
                float error = 0.0;
                // iterate thru next layer and get error based on this layer's weight
                for (size_t k = 0; k < network[i + 1].getNeurons().size(); k++)
                {
                    error += (network[i + 1].getNeurons()[k].getWeights()[j] * network[i + 1].getNeurons()[k].getError());
                }
                // add error to total error array
                errors.push_back(error);
            }
        }
        // calculate error for output layer
        else
        {
            for (size_t j = 0; j < currLayer.size(); j++)
            {
                // just add diff between expected and output for output layer
                errors.push_back(currLayer[j].getOutput() - expected[j]);
            }
        }

        // update errors in neurons
        for (size_t j = 0; j < currLayer.size(); j++)
        {
            // error = errors[j] * currLayer[j].transferDerivative()
            currLayer[j].setError(errors[j] * currLayer[j].transferDerivative());
        }
    }
}

void neuralnetwork::updateWeights(std::vector<float> inputs, float learningRate)
{
    // remove last term of inputs (it should not affect bias)
    inputs.pop_back();

    // loop thru network
    for (size_t i = 0; i < network.size(); i++)
    {
        std::vector<neuron> &currLayer = network[i].getNeurons();

        // input of currlayer = output of prev layer iff currLayer != input layer
        if (i != 0)
        {
            std::vector<neuron> &prevLayer = network[i - 1].getNeurons();
            for (size_t j = 0; j < prevLayer.size(); j++)
            {
                // set output
                inputs[i] = prevLayer[i].getOutput();
            }
        }

        // update the weights of the current layer
        for (size_t j = 0; j < currLayer.size(); j++)
        {
            // get the weights of the current neuron
            std::vector<float> &currWeights = currLayer[j].getWeights();

            // get the current learning rate * error to speed up calculations
            float learnError = learningRate * currLayer[j].getError();

            // loop through all inputs
            for (size_t k = 0; k < inputs.size(); k++)
            {
                // update weights
                currWeights[k] -= learnError * inputs[k];
            }

            // update the bias (the last weight in the weights array)
            currWeights[currWeights.size() - 1] -= learnError;
        }
    }
}

// function to train the neural network
void neuralnetwork::trainNetwork(std::vector<std::vector<float>> trainingData, float learnRate, int numEpoch, int numOutputs)
{
    // run the training sequence epoch number of times
    for(int e = 0; e < numEpoch; e++)
    {
        // set up sum error
        float sumerror = 0;

        // loop through inputs in dataset
        for(std::vector<float>& inputs : trainingData)
        {
            // forward propogate and get output
            std::vector<float> outputs = forwardPropogate(inputs);

            // get the expected outputs
            std::vector<float> expected(numOutputs, 0.0);
            expected[static_cast<int>(inputs.back())] = 1.0;

            // calculate error and update the error sum
            for(size_t i = 0; i < expected.size(); i++)
            {
                sumerror += std::pow(expected[i] - outputs[i], 2);
            }

            // backprop the neural network to update the errors
            backwardPropogateError(expected);

            // update the weights
            updateWeights(inputs, learnRate);
        }

        std::cout << ">epoch= " << e+1 << ", l-rate= " << learnRate << ", error=" << sumerror << std::endl;
    }
}
