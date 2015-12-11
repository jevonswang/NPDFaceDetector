#pragma once

#include <opencv2/opencv.hpp>
#include <armadillo>

// the struct of NPD model
struct NPDModel{
	int objSize;
	int numStages;
	int numBranchNodes;
	int numLeafNodes;
	arma::vec stageThreshold; // okay
	arma::uvec  treeRoot;
	arma::umat  pixel1; // okay
	arma::umat  pixel2; // okay
	arma::umat  cutpoint;
	arma::uvec  leftChild;
	arma::uvec  rightChild;
	arma::vec  fit;
	arma::umat  npdTable;
	double scaleFactor;
	int numScales;
	arma::uvec  winSize;
};
