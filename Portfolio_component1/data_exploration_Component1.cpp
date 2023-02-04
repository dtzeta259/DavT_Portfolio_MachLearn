/*

    Data Exploration Program
    Program will read in the csv file Boston.csv and read the data into two vectors.
    Once read, the program will calculate the data stats of the vectors, including
    the sum, mean, median, and range. Program will also calculate the covariance and
    correlation of both vectors. When the data stats are calculated, the results
    are printed out on the console.

    Created by David Teran (dxt180025) for CS 4375.004

*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <math.h>

using std::cin;
using std::cout;

//Prototypes of function used in the program
double getSum(std::vector<double> vecList, int counter);
double getMean(std::vector<double> vecList, int counter);
double getMedian(std::vector<double> vecList, int counter);
double getRange(std::vector<double> vecList, int counter);
double getCovariance(std::vector<double> vecList1, std::vector<double> vecList2,int counter);
double getCorrelation(std::vector<double> vecList1, std::vector<double> vecList2,int counter);
void printDataStats(std::vector<double> rm, std::vector<double> medv,int counter);

//Main program
int main(int argc, char** argv){
    
    //Variables needed for program
    const int MAX_NUM = 1000;
    std::ifstream inDataFile;
    std::ofstream textResult;
    std::string csvFile = "Boston.csv";
    std::vector<double> rm(MAX_NUM);
    std::vector<double> medv(MAX_NUM);
    std::string headingLine;
    std::string rmIn, medvIn;

    //First, open the csv file for reading and read the first line containing the headings
 
    cout << "Opening file: " << csvFile << " for data exploration.\n";

    inDataFile.open(csvFile);
    if(!inDataFile.is_open()){
        cout << "File not found! Cannot open file: " << csvFile << std::endl;
        return 1;
    }

    cout << "Headings: ";
    std::getline(inDataFile, headingLine);
    cout << headingLine << std::endl;

    //Headings read, now to collect the data from each respective column

    int counterData = 0;
    while(inDataFile.good())
    {
        getline(inDataFile, rmIn, ',');
        getline(inDataFile, medvIn, '\n');

        rm.at(counterData) = stof(rmIn);
        medv.at(counterData) = stof(medvIn);

        counterData++;
    }

    rm.resize(counterData);
    medv.resize(counterData);
    
    inDataFile.close();
    
    printDataStats(rm, medv, counterData);
    
    cout << "\nProgram Terminated! Have a great day!";

    return 0;
}

//Function to print out data statistics

void printDataStats(std::vector<double> rm, std::vector<double> medv,int counter){
    cout << "\nRecords of data: " << counter << std::endl;
    
    cout << "    rm \n";
    cout << "Sum: " << getSum(rm, counter) << "\n";
    cout << "Mean: " << getMean(rm, counter) << "\n";
    cout << "Median: " << getMedian(rm, counter) << "\n";
    cout << "Range: " << getRange(rm, counter) << "\n";
    
    cout << "\n    medv \n";
    cout << "Sum: " << getSum(medv, counter) << "\n";
    cout << "Mean: " << getMean(medv, counter) << "\n";
    cout << "Median: " << getMedian(medv, counter) << "\n";
    cout << "Range: " << getRange(medv, counter) << "\n";

    cout << "\nCovariance of rm and medv: " << getCovariance(rm, medv, counter) << std::endl;
    cout << "Correlation of rm and medv: " << getCorrelation(rm, medv, counter) << std::endl;
}

//Functions to find the sum, mean, median, and range of numeric vector

double getSum(std::vector<double> vecList, int counter){
    double sumNum = 0;

    for(int i = 0; i < counter; i++){
        sumNum = sumNum + vecList[i];
    }

    return sumNum;
}

double getMean(std::vector<double> vecList, int counter){
    double sumNum = 0;
    double averageVec = 0;

    sumNum = getSum(vecList, counter);

    averageVec = sumNum / counter;
    
    return averageVec;
}


double getMedian(std::vector<double> vecList, int counter){

    double medianVec = 0;
    double end1, end2;

    std::sort(vecList.begin(), vecList.end());

    if(counter % 2){
        medianVec = (vecList[counter / 2 - 1] + vecList[counter/2]) / 2;
    }
    else{
        medianVec = vecList[counter / 2];
    }
    

    return medianVec;
}

double getRange(std::vector<double> vecList, int counter){
    double rangeVec = 0;
    double minNum = 0;
    double maxNum = 0;

    minNum = vecList.at(0);
    maxNum = vecList.at(0);

    for(int i = 0; i < counter; i++){
        if(vecList[i] < minNum){
            minNum = vecList[i];
        }

        if(vecList[i] > maxNum){
            maxNum = vecList[i];
        }
    }

    rangeVec = maxNum - minNum;

    return rangeVec;
}

//Functions to compute covariance and correlation between both vectors

double getCovariance(std::vector<double> vecList1, std::vector<double> vecList2, int counter){
    double meanList1, meanList2;
    double calculate = 0;
    double covariance = 0;

    meanList1 = getMean(vecList1, counter);
    meanList2 = getMean(vecList2, counter);

    for(int i = 0; i < counter; i++){

        calculate += (vecList1[i] - meanList1) * (vecList2[i] - meanList2);

    }

    covariance = calculate / (counter - 1);

    return covariance;
}

double getCorrelation(std::vector<double> vecList1, std::vector<double> vecList2,int counter){

    double preCorrelation = 0;
    double calculate1 = 0; 
    double calculate2 = 0;
    double calculate3 = 0;
    double correlation = 0;
    double covariance;
    
   for(int i = 0; i < counter; i++){


        calculate1 += vecList1[i] * vecList2[i]; 
        calculate2 += pow(vecList1[i], 2);
        calculate3 += pow(vecList2[i], 2);
    }

    calculate1 = (counter * calculate1) - (getSum(vecList1, counter) * getSum(vecList2, counter));
    preCorrelation = sqrtf(((counter * calculate2) - pow(getSum(vecList1, counter), 2))  * ((counter * calculate3) - pow(getSum(vecList2, counter), 2)));


    correlation = (calculate1/ preCorrelation);

    return correlation;
}
