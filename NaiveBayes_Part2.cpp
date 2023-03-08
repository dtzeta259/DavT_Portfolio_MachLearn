/**
 * 
 * Portfolio Component 3
 * Naive Bayes
 * 
 * C++ program for implementing Naive Bayes algorithm on the titanic data using
 * age, pclass, and sex as predictors for predicting survival, using the first 800
 * observations for the train data, the remaining as test data. The program will
 * output the coefficients and metrics, as well as run times for the algorithm.
 * 
 * Created by David Teran and Huy Nguyen on February 20, 2023 for CS4375 Intro to Machine Learning
 * 
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <math.h>
#include <chrono>

#ifndef ValPI
#define ValPI 3.14159265358979323846
#endif

using std::cin;
using std::cout;
using std::endl;
using std::vector;

//Function Prototypes
static double vectorSum(vector<double> vect);
static double vectorMean(vector<double> vect);
static double covariance(vector<double> vect1, vector<double> vect2);
static double calcAgeLike(double val, double meanVect, double varVect);

//Main Program
int main(){

    //Variables to be used for reading in the program
    const int TEST_MAX = 1500;
    const int TRAIN_MAX = 800;
    std::ifstream dataInFile;
    std::string csvFile = "titanic.csv";
    std::string headingLine;
    std::string val_in, pclass_in, surv_in, sex_in, age_in;

    //Train and Test Vectors
    vector<double> val(TRAIN_MAX);
    vector<double> pclass(TRAIN_MAX);
    vector<double> survived(TRAIN_MAX);
    vector<double> sex(TRAIN_MAX);
    vector<double> age(TRAIN_MAX);

    vector<double> val_test(TEST_MAX);
    vector<double> pclass_test(TEST_MAX);
    vector<double> surv_test(TEST_MAX);
    vector<double> sex_test(TEST_MAX);
    vector<double> age_test(TEST_MAX);

    //Opening and reading in the csv file

    cout << "Opening file " << csvFile << " for data exploration.\n";

    dataInFile.open(csvFile);
    if(!dataInFile.is_open()){
        cout << "File not found! Cannot open file: " << csvFile << endl;
        return 1;
    }

    cout << "Headings: ";
    getline(dataInFile, headingLine);
    cout << headingLine << endl;

    //Collecting the data into their respective vectors, separating the
    //first 800 observation as train data, the remainder as test data

    int counterObserve = 0;
    while (dataInFile.good() && counterObserve < TRAIN_MAX)
    {
        //Read in the data of each row and split by the column data type
        getline(dataInFile, val_in, ',');
        getline(dataInFile, pclass_in, ',');
        getline(dataInFile, surv_in, ',');
        getline(dataInFile, sex_in, ',');
        getline(dataInFile, age_in, '\n');

        pclass.at(counterObserve) = stof(pclass_in);
        survived.at(counterObserve) = stof(surv_in);
        sex.at(counterObserve) = stof(sex_in);
        age.at(counterObserve) = stof(age_in);

        counterObserve++;
    }

    int testObserve = 0;
    while (dataInFile.good() && testObserve < TEST_MAX)
    {
        //Read in the data of each row and split by the column data type
        getline(dataInFile, val_in, ',');
        getline(dataInFile, pclass_in, ',');
        getline(dataInFile, surv_in, ',');
        getline(dataInFile, sex_in, ',');
        getline(dataInFile, age_in, '\n');

        pclass_test.at(counterObserve) = stof(pclass_in);
        surv_test.at(counterObserve) = stof(surv_in);
        sex_test.at(counterObserve) = stof(sex_in);
        age_test.at(counterObserve) = stof(age_in);

        testObserve++;
    }

    //Close the file, since data has been read in already
    dataInFile.close();
    
    //Begin the timer for training duration
    auto algStart = std::chrono::high_resolution_clock::now();

    //Calculate prior probability
    double prior [2] = {};
    for(int i = 0; i < TRAIN_MAX; i++){
        if(survived.at(i) == 0){
            prior[0]++;
        }
        if(survived.at(i) == 1){
            prior[1]++;
        }
    }

    prior[0] /= TRAIN_MAX;
    prior[1] /= TRAIN_MAX;

    //Likelihood of qualitative data
    //Survived Likelihood
    int surCount[2] = {};
    for (int i = 0; i < TRAIN_MAX; i++){
        if(survived.at(i) == 0){
            surCount[0]++;
        }
        if(survived.at(i) == 1){
            surCount[1]++;
        }
    }
    
    //Pclass Likelihood
    double pclassProb[2][3] = {};
    int sv[] = {0,1};
    int pclassHard [] = {1,2,3};
    for (int s: sv){
        for(int p: pclassHard){
            int counter = 0;
            for(int i = 0; i < TRAIN_MAX; i++){
                if(survived.at(i)==s && pclass.at(i)==p){
                    counter++;
                }
            }
            pclassProb[s][p-1] = ((double) counter) / ((double) surCount[s]);
        }
    }

    //Sex Likelihood
    double sexProb[2][2] = {};
    int sexVal[] = {0,1};
    for (int s: sv){
        for(int sx: sexVal){
            int counter = 0;
            for (int i = 0; i < TRAIN_MAX; i++){
                if(survived.at(i) == s && sex.at(i) == sx){
                    counter++;
                }
            }
            sexProb[s][sx] = ((double)counter) / ((double)surCount[s]);
        }
    }
    
    //Age Likelihood
    double meanAge[2] = {};
    double variaAge[2] = {};
    for(int s: sv){
        vector<double> surAge;
        for(int i = 0; i < TRAIN_MAX; i++){
            if(survived.at(i) == s){
                surAge.push_back(age.at(i));
            }
        }
        meanAge[s] = vectorMean(surAge);
        variaAge[s] = covariance(surAge,surAge);
    }

    //Training Finish
    auto algStop = std::chrono::high_resolution_clock::now();
    auto algDuration = std::chrono::duration_cast<std::chrono::milliseconds>(algStop - algStart);
    
    //Start Test Set
    double results[TEST_MAX] = {};
    for(int i = 0; i < TEST_MAX; i++){
        double probably[2] = {};

        double numSurv = pclassProb[1][(int)(pclass_test.at(i) - 1)] * 
            sexProb[1][(int)sex_test.at(i)] * prior[1] * calcAgeLike(age_test.at(i), meanAge[1], variaAge[1]);

        double numProb = pclassProb[0][(int)(pclass_test.at(i) - 1)] * 
            sexProb[0][(int)sex_test.at(i)] * prior[0] * calcAgeLike(age_test.at(i), meanAge[0], variaAge[0]);

        double denomin = numSurv + numProb;
        probably[0] = numSurv / denomin;
        probably[1] = numProb / denomin;

        results[i] = (probably[0] >= probably[1]) ? 1 : 0;
    }

        //Coefficients output
    cout << "Coefficient Output: " << "\n\n";
    cout << "Prior Probabilities: "<< endl;
    cout << "0: " << prior[0] << " 1: " << prior[1] << "\n\n";

        //Print pclass Probabiltiy
    cout << "Pclass Probability:" << endl;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            cout << pclassProb[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;

    //Print Sex Probability
    cout << "Sex Probability:" << endl;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            cout << sexProb[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;

    //Print age stats
    cout << "Age Statistics: " << endl;
    cout << "Mean: " << meanAge[0] << ", Variance: " << variaAge[0] << endl;
    cout << "Mean: " << meanAge[1] << ", Variance: " << variaAge[1] << endl;
    cout << endl;


    //Data Accuracy
    double accurate = 0;
    for (int i = 0; i < TEST_MAX; i++) {
        if (surv_test.at(i) == results[i]) {
            accurate++;
        }
    }
    accurate /= TEST_MAX;
    cout << "Training Algorithm Duration (Milliseconds): " << algDuration.count() << endl;
    cout << "Accuracy: " << accurate << endl;
    
    //Sensitivity and Specificity
    double tp = 0, fn = 0, tn = 0, fp = 0;
    for (int i = 0; i < TEST_MAX; i++) {
        if (surv_test.at(i) == results[i] && results[i]==1) {
            tp++;
        } else if (surv_test.at(i) != results[i] && results[i] == 0) {
            fn++;
        }
        else if (surv_test.at(i) == results[i] && results[i] == 0) {
            tn++;
        }
        else if (surv_test.at(i) != results[i] && results[i] == 1) {
            fp++;
        }
    }

    cout << "Sensitivity: " << tp / (tp + fn) << endl;
    cout << "Specificity: " << tn / (tn+fp) << endl;

    return 0;
}

//Functions used for this program

static double vectorSum(vector<double> vect) {
    double sum = 0;
    for (int i = 0; i < vect.size(); i++) {
        sum += vect.at(i);
    }
    return sum;
}

static double vectorMean(vector<double> vect) {
    double sum = vectorSum(vect);
    return sum / vect.size();
}

static double covariance(vector<double> vect1, vector<double> vect2) {
    double covarSum = 0;
    double vect1Avg = vectorMean(vect1);
    double vect2Avg = vectorMean(vect2);
    for (int i = 0; i < vect2.size(); i++) {
        covarSum += (vect1.at(i) - vect1Avg) * (vect2.at(i) - vect2Avg);
    }
    covarSum /= (vect1.size() - 1);

    return covarSum;
}

static double calcAgeLike(double val, double meanVect, double varVect) {
    double likelihood = 1 / sqrt(2* ValPI * varVect);
    double secTerm = ((-1 * pow((val - meanVect), 2))/(2*varVect));
    double raisedToE = exp(secTerm);
    likelihood *= raisedToE;
    return likelihood;
}