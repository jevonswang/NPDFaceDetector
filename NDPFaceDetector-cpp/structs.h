#pragma once

#include <armadillo>

// the struct of NPD model
class NPDModel{
public:
	int objSize;
	int numStages;
	int numBranchNodes;
	int numLeafNodes;
	arma::mat stageThreshold;
	arma::mat treeRoot;
	arma::mat pixel1;
	arma::mat pixel2;
	arma::mat cutpoint;
	arma::mat leftChild;
	arma::mat rightChild;
	arma::mat fit;
	arma::mat npdTable;
	double scaleFactor;
	int numScales;
	arma::mat winSize;
};
