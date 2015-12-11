#include <iostream>
#include <vector>
#include <string>
#include <armadillo>
#include <opencv2/opencv.hpp>
#include "f2c.h"
#include "clapack.h"
#include "structs.h"
#include "DetectFace.h"
#include "LoadMat.h"

using namespace std;

void runDetect(){

	string modelFile = "F:\\NDPData\\model\\model_frontal.txt";
	//string modelFile = "F:\\NDPData\\model\\model_unconstrain.txt";
	string imgFile = "F:\\NDPData\\images\\lena.jpg";

	NPDModel npdModel;
	loadModelFile(modelFile, npdModel);

	cv::Mat img = cv::imread(imgFile);
	vector<cv::Rect> rects;

	clock_t start, end;
	start = clock();
	DetectFace(rects, npdModel, img);
	end = clock();
	double dur = (double)(end - start);
	printf("detect time:%f s\n", (dur / CLOCKS_PER_SEC));

	int numFaces = rects.size();
	printf("%d faces detected.\n", numFaces);

	if (numFaces > 0){
		int border = round(img.cols * 1.0 / 300);
		if (border < 2){
			border = 2;
		}
		
		for (int j = 0; j < numFaces; j++){
			cv::rectangle(img, rects[j], cv::Scalar(0,255,0), border);
		}
		
	}
	cv::imshow("result", img);
	cv::waitKey();
}

int main(int argc, char* argv[]){
	
	runDetect();
	return 0;
}