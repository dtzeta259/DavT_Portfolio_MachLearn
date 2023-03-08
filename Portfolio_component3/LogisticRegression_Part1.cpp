/**
 * 
 * Portfolio Component 3
 * Logistic Regression
 * 
 * C++ program that will read in the titanic data file and will perform logistic
 * regression, predicting the number of survived based on sex. The data will be divided,
 * using the first 800 observations as the train data, to then output the coefficients.
 * The remaining data will then be used to predict values. Functions for calculating accuracy,
 * sensitivity, and specificity will also be called. Once done, the test metrics and run time
 * for the algorithm will be printed out.s 
 * 
 * Created by David Teran on February 20, 2023 for CS4375 Intro to Machine Learning
 * 
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <math.h>
#include <float.h>
#include <iomanip>
#include <chrono>
#include <utility>

using std::cin;
using std::cout;
using std::endl;
using std::vector;
using std::setw;
using std::right;
using std::setfill;

//struct used for data split
struct dataSplit
{
    int survived;
    int sex;
};


//Function Prototypes
vector<vector<double>> transposeMat(vector<vector<double>> matrix1);
vector<double> sigmoidLR(vector<double> datamat_product);
vector<double> multiplyMat(vector<vector<double>> matrixData, vector<double> weightVect);

int main(){

    //Variables used for reading in the file
    const int MAX_NUM = 1500;
    std::ifstream dataInFile;
    std::string csvFile = "titanic.csv";
    vector<double> survived (MAX_NUM);
    vector<double> sex(MAX_NUM);
    std::string headingLine;
    std::string surv_in, sex_in;

    vector<dataSplit> train_data;
    vector<dataSplit> test_data;
    vector<vector<double>> dataMatrix;
    vector<double> weights{1,1};
    double learn_Rate = 0.001;

    //Open the file and read in the headings for the data
    cout << "Opening file: " << csvFile << " for data exploration.\n";

    dataInFile.open(csvFile);
    if(!dataInFile.is_open()){
        cout << "File not found! Cannot open file: " << csvFile << endl;
        return 1;
    }

    cout << "Headings: ";
    std::getline(dataInFile, headingLine);
    cout << headingLine << endl;

    //Printed Headings, now reading in the data being used for logistic regression

    int counterData = 0;
    while(dataInFile.good())
    {
        //Remove the first element, as its not being used
        std::string clean;

        getline(dataInFile, clean, ',');

        //Read in the survived and sex data columns
        getline(dataInFile, surv_in, ',');
        getline(dataInFile, sex_in, ',');

        survived.at(counterData) = stof(surv_in);
        sex.at(counterData) = stof(sex_in);

        //Remove the rest of the unused data
        getline(dataInFile, clean, '\n');

        counterData++;
    }

    survived.resize(counterData);
    sex.resize(counterData);
    
    //closing file, now calculating and printing data stats
    dataInFile.close();

    //Create the testing and training data for logistic regression
    for(int i = 0; i < 800; i++){
        dataSplit train = {static_cast<int>(survived.at(i)), static_cast<int>(sex.at(i))};
        train_data.push_back(train);
    }

    for(int i = 800; i < survived.size(); i++){
        dataSplit test = {static_cast<int>(survived.at(i)), static_cast<int>(sex.at(i))};
        test_data.push_back(test);
    }

    //Create data matrix
    for(int i = 0; i < train_data.size(); i++){
        vector<double> data{1, (double)train_data.at(i).survived };
        dataMatrix.push_back(data);
    }

    //creating transpose matrix before running algorithm, due to constants
    vector<vector<double>> transposeMatrix = transposeMat(dataMatrix);

    //Training the model, time starts
    auto startTime = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 1000; i++){
        //Calculate product of data matrix and weight vector. Weights change with each iteration
        vector<double> matrixProd = multiplyMat(dataMatrix, weights);
        vector<double> probVector = sigmoidLR(matrixProd);

        vector<double> errorVec(probVector.size());

        //Calculate error by taking difference of train sex 
        for(int i = 0; i < train_data.size(); i++){
            errorVec.at(i) = train_data.at(i).sex - probVector.at(i);
        }

        //Calculate transpose of data matrix
        vector<double> transposeProd = multiplyMat(transposeMatrix, errorVec);

        for(int i = 0; i < transposeProd.size(); i++){
            transposeProd.at(i) *= learn_Rate;
        }

        weights.at(0) += transposeProd.at(0);
        weights.at(1) += transposeProd.at(1);
    }

    auto stopTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stopTime - startTime);

    vector<vector<double>> dataMatrix2;
    for(int i = 0; i < test_data.size(); i++){
        vector<double> dataVect{1, (double)test_data.at(i).survived};
        dataMatrix2.push_back(dataVect);
    }

    //Form Probabilities from test data
    vector<double> predict = multiplyMat(dataMatrix2, weights);
    vector<double> probably;
    double exponent_pred = 0;

    for (int i = 0; i < dataMatrix2.size(); i++)
    {
        double exp_predict = exp(predict.at(i));
        probably.push_back(exp_predict / (1 + exp_predict));
    }

    vector<double> prediction;
    for (int i = 0; i < probably.size(); i++)
    {
        if(probably.at(i) > 0.5)
            prediction.push_back(1);
        else
            prediction.push_back(0);
    }

    //Treating 1 as positive class
    double acc = 0;
    int tn = 0, tp = 0, fn = 0, fp = 0;
    for (int i = 0; i < prediction.size(); i++)
    {
        int var = prediction.at(i);
        if(var == test_data.at(i).sex){
            acc++;
            if(var == 0)
                tn++;
            else
                tp++;
        }
        else{
            if(var == 0)
                fn++;
            else
                fp++;
        }
    }

    acc /= prediction.size();
    double sensitivity = tp / ((double)tp + fn);
	double specificity = tn / ((double)tn + fp);
	//Model information
	cout << "--- Model Coefficients and Data Output ---" << endl << endl;
	cout << "Weight 1: " << weights.at(0) << endl;
	cout << "Weight 2: " << weights.at(1) << endl << endl;;
	cout << "Accuracy: " << acc << endl << endl;

	cout << "Training time after 1k iterations: " << duration.count() << " ms\n" << endl;

	cout << "--- Confusion Matrix Result ---\n" << endl;
	cout << "Treating values 1 'sex' as the positive class\n" << endl;
	


	cout << setw(4) << setfill(' ') << right <<"pred" << setw(4) <<"0" << setw(4) << "1" << endl;
	cout << setfill(' ') << setw(4) << right << '0' << setw(4) << tn << setw(4) << fn << endl;
	cout << setfill(' ') << setw(4) << right << '1' << setw(4) << fp << setw(4) << tp << endl << endl;

	cout << "Sensitivity: " << sensitivity << endl;
	cout << "Specificity: " << specificity << endl;
    
    

    return 0;

}

//Functions used in this program

//transposeMat: function that takes a matrix and implements the 
//transposition of the matrix

vector<vector<double>> transposeMat(vector<vector<double>> matrix1){

    vector<vector<double>> matrix2;
    vector<double> row1;
    vector<double> row2;

    for(int i = 0; i < matrix1.size(); i++){
        row1.push_back(matrix1.at(i).at(0));
        row2.push_back(matrix1.at(i).at(1));
    }

    matrix2.push_back(row1);
    matrix2.push_back(row2);


    return matrix2;

}

//sigmoidLR: function that calculates the sigmoid for use in logistic regression

vector<double> sigmoidLR(vector<double> datamat_product){
    vector<double> sigResult(datamat_product.size());
    for(int i = 0; i < datamat_product.size(); i++){
        double result = 1.0 / (1 + exp(-(datamat_product.at(i))));
        sigResult.at(i) = result;
    }
    return sigResult;
}

//multiplyMat: function that calculates the product of the matrix
//and the weight vector
vector<double> multiplyMat(vector<vector<double>> matrixData, vector<double> weightVect){
    vector<double> resultProd(matrixData.size());

    for(int i = 0; i < matrixData.size(); i++){
        vector<double> row1 = matrixData.at(i);
        double prod = 0;
        for(int j = 0; j < row1.size(); j++){
            prod += row1.at(j) * weightVect.at(j);
        }

        resultProd.at(i) = prod;
    }
    return resultProd;
}
