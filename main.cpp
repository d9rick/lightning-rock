#include "nnetwork.h"
#include <iostream>
#include <fstream>
#include <set>
#include <sstream>

void normalizeDataSet(std::vector<std::vector<float>>& dataset, std::vector<std::pair<float, float>> minmax);
std::vector<std::pair<float, float>> datasetMinMax(std::vector<std::vector<float>> dataset);

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
    const int numhidden = 5;
    const float learningRate = .3;
    const int epoch = 500;

    // call file parser
    std::vector<std::vector<float>> trainingdata = getTrainingData(argv[1], numinput, numoutput);

    // check training data
    if(trainingdata.empty())
    {
        std::cerr << "YO there isn't anything in this data set bro??" << std::endl;
        return -1;
    }

    // normalize dataset
    std::vector<std::pair<float, float>> minmax = datasetMinMax(trainingdata);
    normalizeDataSet(trainingdata, minmax);
    
    //print data vals
    std::cout << "Training data contains...\n# Inputs: " << numinput << "\n# Outputs: " << numoutput << std::endl << std::endl;


    // create network
    neuralnetwork network(numinput, numhidden, numoutput);

    // print network
    std::cout   << "Created NeuralNotwork with...\n# Inputs: " << numinput
                << "\n# Hidden: " << numhidden << "\n# Outputs: " << numoutput << std::endl << std::endl;
    network.forwardPropogate(trainingdata[0]);
    network.print();

    // train network
    std::cout << "\nTraining... " << std::endl;
    network.trainNetwork(trainingdata, learningRate, epoch, numoutput);

    // set up for predictions
    std::cout << "\nChecking network accuracy..." << std::endl;

    // set up accuracy vars
    int numTrials = 0;
    int numCorrect = 0;

    // run trials
    for(std::vector<float>& input : trainingdata)
    {
        // get expected
        int expected = input.back();
        std::cout << "Expected:  " << expected << ", ";

        // get predicted
        int predicted = network.predict(input);
        std::cout << "Predicted: " << predicted << std::endl;

        // check if expected matches predictions
        if(expected == predicted)
        {
            numCorrect++;
        }

        // increment trials
        numTrials++;
    }

    // check accuracy & print results
    float accuracy = float(numCorrect)/float(numTrials);
    std::cout << std::endl << "RESULTS:\n";
    std::cout << "# Trials:  " << numTrials << std::endl;
    std::cout << "# Correct: " << numCorrect << std::endl;
    std::cout << "Accuracy:  " << accuracy*100 << '%' << std::endl;

    return 0;
}

// ------------------------ Data Helper Functions ------------------------

// returns a vector of pairs of min and max values for each column in the dataset
std::vector<std::pair<float, float>> datasetMinMax(std::vector<std::vector<float>> dataset)
{
    // init minmax
    std::vector<std::pair<float, float>> minmax;

    // init minmax to first row of dataset
    for(size_t i = 0; i < dataset[0].size(); i++)
    {
        minmax.push_back(std::pair<float, float>(dataset[0][i], dataset[0][i]));
    }

    // loop through dataset
    for(std::vector<float> row : dataset)
    {
        // loop through row
        for(size_t i = 0; i < row.size(); i++)
        {
            // update minmax
            if(row[i] < minmax[i].first)
            {
                minmax[i].first = row[i];
            }
            else if(row[i] > minmax[i].second)
            {
                minmax[i].second = row[i];
            }
        }
    }

    return minmax;
}

// normalizes the dataset
void normalizeDataSet(std::vector<std::vector<float>>& dataset, std::vector<std::pair<float, float>> minmax)
{
    // loop through dataset
    for(std::vector<float>& row : dataset)
    {
        // loop through row
        for(size_t i = 0; i < row.size(); i++)
        {
            // normalize
            row[i] = (row[i] - minmax[i].first) / (minmax[i].second - minmax[i].first);
        }
    }
}