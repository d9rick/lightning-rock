#include "nnetwork.h"
#include <random>

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

/*
-------------- Layer Implementation --------------
*/

layer::layer(int numneurons, int numweights)
{
    // create hidden layer
    for(int i = 0; i < numneurons; i++)
    {  
        neurons.push_back(neuron(numweights)); // add weights for each node + bias
    }
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