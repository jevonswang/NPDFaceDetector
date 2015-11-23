#include <iostream>
#include <vector>
#include <armadillo>
#include <opencv2/opencv.hpp>
#include "structs.h"
#include "DetectFace.h"

using namespace std;

bool loadModelFile(NPDModel &npdModel, const string &filepath){
	


	return true;
}


int main(){
	string modelFile = "model_frontal.mat";
	string imgFile = "lena.jpg";

	NPDModel npdModel;
	loadModelFile(npdModel,modelFile);

	cv::Mat img = cv::imread(imgFile);
	vector<cv::Rect> rects;
	DetectFace(rects, npdModel, img);

	int numFaces = rects.size();
	printf("%d faces detected.\n",numFaces);

	if (numFaces > 0){
		int border = round(img.cols * 1.0 / 300);
		if (border < 2){
			border = 2;
		}

		for (int j = 0; j < numFaces; j++){
			cv::rectangle(img, rects[j], cv::Scalar::all(0),border);
		}
	}

	cv::imshow("result",img);
	return 0;
}