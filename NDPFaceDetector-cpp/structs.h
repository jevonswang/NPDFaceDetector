#pragma once

#include <opencv2/opencv.hpp>


// the struct of NPD model
struct NPDModel{
	int objSize;
	int numStages;
	int numBranchNodes;
	int numLeafNodes;
	cv::Mat stageThreshold;
	cv::Mat  treeRoot;
	cv::Mat  pixel1;
	cv::Mat  pixel2;
	cv::Mat  cutpoint;
	cv::Mat  leftChild;
	cv::Mat  rightChild;
	cv::Mat  fit;
	cv::Mat  npdTable;
	double scaleFactor;
	int numScales;
	cv::Mat  winSize;
};

struct BoostOpt{
	int treeLevel; // the maximal depth of the DQT trees to be learned
	int maxNumWeaks; // maximal number of weak classifiers to be learned
	double minDR; // minimal detection rate required
	double maxFAR; // maximal FAR allowed; stop the training if reached
	int minSamples; // minimal samples required to continue training. 
	                // 1000 is preferred in practice.
};

struct Options{
	int objSize; // size of the face detection template
	double negRatio; // factor of bootstrap nonface samples. 
					 // For example,negRatio=2 means bootstrapping two times 
	                 // of nonface samples w.r.t face samples.
	int finalNegs; // the minimal number of bootstrapped nonface samples.
				   // The training will be stopped if there is no enough nonface 
	               // samples in the final stage. This is also to avoid overfitting.
	int numFaces; // the number of face samples to be used for training.
				  // Inf means to use all face samples.
	int numThreads; // the number of computing threads for bootstrapping

	BoostOpt boostOpt; // see the comments above
};