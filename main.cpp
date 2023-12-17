#include "nnetwork.h"
#include <iostream>
#include <fstream>
#include <set>
#include <sstream>

// for testing
const int numinput  = 2;
const int numhidden = 1;
const int numoutput = 2;

// takes inputs in from a file
// note: assumes last value on line is expected, rest are inputs
std::vector<std::vector<float>> getTrainingData(char* filename, int &numinputs, int &numoutputs)
{
    // init dataset
    std::vector<std::vector<float>> dataset;

    // open file
    std::ifstream file(filename);
    if(!file.good())
    {
        std::cout << "File no worky" << std::endl;
        return dataset;
    }

    // init outputs
    std::set<float> outputs;

    // init string stream and parse loop
    std::string buffer;
    while(std::getline(file, buffer))
    {
        // create row in dataset for this row of data
        dataset.push_back(std::vector<float>());

        // create string stream to feed data into dataset
        std::stringstream sstream(buffer);
        float temp;

        // put all inputs & expected into dataset
        while(sstream >> temp)
        {
            dataset.back().push_back(temp);
        }

        // try adding last element to the output set
        // len of the output set at the end will be num of unique outputs ie output neurons
        outputs.insert(temp);
    }

    // update numoutputs and numinputs
    numoutputs = outputs.size();
    if(!dataset.empty())
    {
        numinputs = dataset[0].size() - 1;
    }
    else
    {
        numinputs = 0;
    }

    return dataset;
}


void oldTest()
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

    // test expected
    std::vector<float> expected = {0.0, 1.0};

    // get outputs
    std::vector<float> outputs = network.forwardPropogate(inputs);

    // print network
    std::cout << "Network:" << std::endl;
    network.print();

    // backpropogate
    std::cout << "Backpropogating..." << std::endl;
    network.backwardPropogateError(expected);

    // print network
    std::cout << "Network:" << std::endl;
    network.print();
}

int main(int argc, char** argv)
{   
    if(argc != 2)
    {
        std::cout << "Please input a file." << std::endl;
        return -1;
    }

    // init network values
    int numinput = 0, numoutput = 0;

    // user set variables
    const int numhidden = 4;
    const float learningRate = .5;
    const int epoch = 20;

    // call file parser
    std::vector<std::vector<float>> trainingdata = getTrainingData(argv[1], numinput, numoutput);

    // check training data
    if(trainingdata.empty())
    {
        std::cerr << "YO there isn't anything in this data set bro??" << std::endl;
        return -1;
    }
    
    //print data vals
    std::cout << "Training data contains...\n# Inputs: " << numinput << "\n# Outputs: " << numoutput << std::endl << std::endl;


    // create network
    neuralnetwork network(numinput, numhidden, numoutput);

    // print network
    std::cout   << "Created NeuralNotwork with...\n# Inputs: " << numinput
                << "\n# Hidden: " << numhidden << "\n# Outputs: " << numoutput << std::endl << std::endl;
    //network.print();

    // train network
    std::cout << "Training... " << std::endl;
    network.trainNetwork(trainingdata, learningRate, epoch, numoutput);

    return 0;
}