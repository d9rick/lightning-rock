#include "nnetwork.h"
#include <iostream>

// for testing
const int numinput  = 2;
const int numhidden = 1;
const int numoutput = 2;

neuralnetwork testNetwork(numinput, numhidden, numoutput);

int main(int argc, char** argv)
{
    // hidden layer
    std::vector<neuron> hiddenneurons = {neuron(std::vector<float>{0.13436424411240122, 0.8474337369372327, 0.763774618976614})};
    layer hiddenLayer(hiddenneurons);

    // output layer
    std::vector<neuron> outputneurons = {neuron(std::vector<float>{0.2550690257394217, 0.49543508709194095}), neuron(std::vector<float>{0.4494910647887381, 0.651592972722763})};
    layer outputLayer(outputneurons);

    // create network
    neuralnetwork network(std::vector<layer>{hiddenLayer,outputLayer});

    // test inputs
    std::vector<float> inputs = {1.0, 0.0};

    // get outputs
    std::vector<float> outputs = network.forwardPropogate(inputs);

    // print network
    std::cout << "Network:" << std::endl;
    network.print();

    // print outputs
    std::cout << "Outputs: ";
    for(float output : outputs)
    {
        std::cout << output << " ";
    }

    return 0;
}
