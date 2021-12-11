#include <iostream>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <time.h>


using namespace std;

int MAX_INT = 2147483647;

struct node{
    vector<double> features;
    int classLabel;
};

double crossValidate(vector<node*> data, vector<int> currentFeatureSet, int feature, int choice){
    if (choice == 1) { //for forward selection
        currentFeatureSet.push_back(feature);
    }
    else{ //for backward elimination
        int index = currentFeatureSet.size() - 1;
        for (int j = 0; j < currentFeatureSet.size(); j++) {
            if (currentFeatureSet.at(j) == feature) {
                index = j;
            }
        }
        currentFeatureSet.at(index) = currentFeatureSet.at(currentFeatureSet.size() - 1);
        currentFeatureSet.pop_back();
    }

    int correct = 0; //number of correctly classified
    double nnDist; //distance of nearest neighbor
    int nnLoc; //index of nearest neighbor
    double distance; //distance between two elements
    node* objectToClassify; //element to be classified in the test
    for (int i = 0; i < data.size(); i++) {
        objectToClassify = data.at(i); //will classify all instances using all other instances at each loop
        nnDist = MAX_INT;
        nnLoc = MAX_INT;
        for (int j = 0; j < data.size(); j++) { //loop through to find nearest neighbor
            if (i != j) {
                double sum = 0;
                double temp = 0;
                for (int i = 0; i < currentFeatureSet.size(); i++) { //loop through all features being looked at
                    temp = (data.at(j)->features.at(currentFeatureSet.at(i) - 1) - objectToClassify->features.at(currentFeatureSet.at(i) - 1));
                    temp = temp * temp;
                    sum = temp + sum;
                }
                distance = sqrt(sum);
                //update nn if distance is smaller than current nn
                if (distance < nnDist) {
                    nnDist = distance;
                    nnLoc = j;
                }
            }
        }
        //check if the class labels are the same
        if (data.at(nnLoc)->classLabel == objectToClassify->classLabel) {
            correct++;   //if yes, then increment correct
        }
    }
    //return #correct/size of data
    return ((correct)*(1.0) / data.size());

}

void forwardSearch(vector<node*> data){
    int numFeatures = data.at(0)->features.size();
    vector<int> currSet; //stores the feature choice at each level
    vector<int> bestFeatures; //stores list of features with the highest accuracy
    int featureToAdd; //used to find feature with best accuracy at each level
    double accuracy; //accuracy of the features at each level
    double bestAcc; //best accuracy of feature list at each level
    double totalAcc; //total best feature list accuracy

    cout << "Beginning search." << endl << endl;

    for (int i = 0; i < numFeatures; ++i) {
        bestAcc = 0;
        for (int j = 0; j < numFeatures; ++j) { //check all features at each level
            if(find(currSet.begin(), currSet.end(), j+1) == currSet.end()){ //checks if j+1 is not in currSet
                accuracy = crossValidate(data, currSet, j+1, 1) * 100;
                cout << "   Using feature(s) {" << j+1;
                for (int k = 0; k < currSet.size(); ++k) {
                    cout << "," << currSet.at(k);
                }
                cout << "} accuracy is " << accuracy << "%." << endl;

                if (accuracy > bestAcc) { //update bestAcc and featureToAdd if accuracy is better than best accuracy
                    bestAcc = accuracy;
                    featureToAdd = j + 1;
                }
            }
        }

        currSet.push_back(featureToAdd); //add best feature to currSet
        cout << endl;

        if (bestAcc > totalAcc) { //if we find a list with better accuracy then use that
            totalAcc = bestAcc;
            bestFeatures = currSet;
        }
        else{
            cout << "Accuracy decreased." << endl;
        }

        cout << "Feature set {";
        for (int j = 0; j < currSet.size(); ++j) { //used to see what feature set we're on
            if (j < currSet.size() - 1) {
                cout << currSet.at(j) << ",";
            }
            else {
                cout << currSet.at(j);
            }
        }
        cout << "} was best, accuracy is " << bestAcc << "%." << endl;
    }

    cout << "Finished search!! The best feature subset is {";
    for (int k = 0; k < bestFeatures.size(); k++) {
        if (k < bestFeatures.size() - 1) {
            cout << bestFeatures.at(k) << ",";
        }
        else {
            cout << bestFeatures.at(k);
        }
    }

    cout << "}, which has an accuracy of " << totalAcc << "%." << endl;

}

void backwardSearch(vector<node*> data){
    int numFeatures = data.at(0)->features.size();
    vector<int> currSet; //stores the feature choice at each level
    vector<int> bestFeatures; //stores list of features with the highest accuracy
    int featureToRemove; //used to find feature with worst accuracy at each level
    double accuracy; //accuracy of the features at each level
    double bestAcc; //best accuracy of feature list at each level
    double totalAcc; //total best feature list accuracy

    cout << "Beginning search." << endl << endl;

    for(int i = 0; i < numFeatures; ++i){
        currSet.push_back(i+1);
    }

    for (int i = 0; i < numFeatures-1; ++i) {
        bestAcc = 0;
        for (int j = 0; j < numFeatures; ++j) { //checks all features potential to remove at each level
            if (find(currSet.begin(), currSet.end(), j+1) != currSet.end()) { //check if we already removed the feature
                accuracy = crossValidate(data, currSet, j + 1, 2)*100;
                cout << "   Using feature(s) {";
                if (currSet.size() >= 2) {
                    if (currSet.at(0) == j+1) {
                        cout << currSet.at(1);
                    }
                    else if(currSet.at(1) == j+1) {
                        cout << currSet.at(0);
                    }
                    else {
                        cout << currSet.at(0) << "," << currSet.at(1);
                    }
                }
                else{
                    cout << currSet.at(0);
                }

                for (int k = 2; k < currSet.size(); ++k) {
                    if (currSet.at(k) != j + 1) {
                        cout << "," << currSet.at(k);
                    }
                }
                cout << "} accuracy is " << accuracy << "%." << endl;

                if (accuracy >= bestAcc) { //update bestAcc and featureToRemove if accuracy is better than bestAcc
                    bestAcc = accuracy;
                    featureToRemove = j + 1;
                }
            }
        }

        int in; //find index of feature to remove
        in = currSet.size() - 1;
        for (int j = 0; j < currSet.size(); ++j) {
            if (currSet.at(j) == featureToRemove) {
                in = j;
            }
        }

        swap(currSet.at(in), currSet.at(currSet.size()-1)); //swap the index of feature to remove w last feature
        currSet.pop_back();
        cout << endl;

        if (bestAcc >= totalAcc) { //update total accuracy and best features if best accuracy is better than total acc
            totalAcc = bestAcc;
            bestFeatures = currSet;
        }
        else {
            cout << "Accuracy decreased." << endl;
        }


        cout << "Feature set {"; //shows what feature set it's on
        for (int k = 0; k < currSet.size(); ++k) {
            if (k < currSet.size() - 1) {
                cout << currSet.at(k) << ",";
            }
            else {
                cout << currSet.at(k);
            }
        }
        cout << "} was best, accuracy is " << bestAcc << "%." << endl;
    }


    cout << "Finished search!! The best feature subset is {";
    for (int k = 0; k < bestFeatures.size(); ++k) {
        if (k < bestFeatures.size() - 1) {
            cout << bestFeatures.at(k) << ",";
        }
        else {
            cout << bestFeatures.at(k);
        }
    }
    cout << "}, which has an accuracy of " << totalAcc << "%." << endl ;
}




int main(int argc, char* argv[]) {
    clock_t tStart = clock();
    vector<node*> data;
    stringstream ss;
    string str;
    node* temp;
    double num;

    if(argc != 2){
        cout << "Usage error: expected <executable> <input>" << endl;
        exit(1);
    }

    ifstream inFS;
    inFS.open(argv[1]);
    if (!inFS.is_open()){
        cout << "Error: could not open file." << endl;
        exit(1);
    }

    while(getline(inFS, str)){ //gets the features and class labels from the file
        temp = new node;
        ss << str;
        ss >> num;
        temp->classLabel = num;
        while(ss >> num){
            temp->features.push_back(num);
        }
        ss.clear();
        data.push_back(temp);
    }

    int choice = 0;
    cout << "Pick 1 for forward selection or 2 for backwards selection" << endl;
    cin >> choice;

    if(choice == 1){
        forwardSearch(data);
    }
    else if(choice == 2){
        backwardSearch(data);
    }
    else{
        exit(1);
    }


    printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

    return 0;
}
