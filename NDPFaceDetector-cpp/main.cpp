#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "structs.h"
#include "DetectFace.h"
#include "TrainDetector.h"
#include "LoadMat.h"

using namespace std;

void runTrain(){
	Options options;

	options.objSize = 20; // size of the face detection template
	options.negRatio = 1; // factor of bootstrap nonface samples.For example,
	// negRatio = 2 means bootstrapping two times of nonface samples w.r.t face samples.
	options.finalNegs = 100; // the minimal number of bootstrapped nonface samples.
	// The training will be stopped if there is no enough nonface samples in the
	// final stage.This is also to avoid overfitting.
	options.numFaces = INT_MAX; // the number of face samples to be used for training.
	// Inf means to use all face samples.
	options.numThreads = 24; // the number of computing threads for bootstrapping

	options.boostOpt.treeLevel = 4; // the maximal depth of the DQT trees to be learned
	options.boostOpt.maxNumWeaks = 4000; // maximal number of weak classifiers to be learned
	options.boostOpt.minDR = 1.0; // minimal detection rate required
	options.boostOpt.maxFAR = 1e-16; // maximal FAR allowed; stop the training if reached
	options.boostOpt.minSamples = 100; // minimal samples required to continue training. 1000 is preferred in practice
	// for other options to control the learning, please see LearnGAB.m.

	string faceDBFile = "..\\data\\FaceDB.mat";
	string nonfaceDBFile = "..\\data\\NonfaceDB.mat";
	string outFile = "..\\result.mat";

	NPDModel npdModel = TrainDetector(faceDBFile, nonfaceDBFile, outFile, options);
}

void runDetect(){
	string modelFile = "model_frontal.txt";
	string imgFile = "lena.jpg";

	NPDModel npdModel;
	loadModelFile(modelFile, npdModel);

	cv::Mat img = cv::imread(imgFile);
	vector<cv::Rect> rects;
	DetectFace(rects, npdModel, img);

	int numFaces = rects.size();
	printf("%d faces detected.\n", numFaces);

	if (numFaces > 0){
		int border = round(img.cols * 1.0 / 300);
		if (border < 2){
			border = 2;
		}

		for (int j = 0; j < numFaces; j++){
			cv::rectangle(img, rects[j], cv::Scalar::all(0), border);
		}
	}

	cv::imshow("result", img);

}

int main(int argc, char* argv[]){
	
	//string pathFaceDB = "";
	//cv::Mat FaceDB;
	NPDModel npdModel;
	string modelFile = "model_frontal.txt";
	loadModelFile(modelFile, npdModel);

	/*
	cout << "Press t to train model, press d to detect face.\n" << endl;
	char ch;
	cin >> ch;
	if (ch == 't'){
		runTrain();
	}
	else if (ch == 'd'){
		runDetect();
	}
	else{
		cout << "bad input." << endl;
	}
	*/
	return 0;
}