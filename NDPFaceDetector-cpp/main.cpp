#include <iostream>
#include <vector>
#include <string>
#include <armadillo>
#include <opencv2/opencv.hpp>
#include "ShapePredictor.h"

#include "f2c.h"
#include "clapack.h"
#include "structs.h"
#include "DetectFace.h"
#include "LoadMat.h"


//using namespace std;
//using namespace Eigen;

void PlotShape(cv::Mat& im, vector<Eigen::Vector2f> s){
	for (int i = 0; i<s.size(); i++)
		circle(im, cv::Point2f(s[i](0), s[i](1)), 3, CV_RGB(255, 0, 0));
}

void runDetect(){

	string modelFile = "F:\\NDPData\\model\\model_frontal.txt";
	//string modelFile = "F:\\NDPData\\model\\model_unconstrain.txt";
	string imgFile = "F:\\NDPData\\images\\lena.jpg";
	string spFile = "F:\\NDPData\\model\\sp68.dat";

	// load npdModel
	NPDModel npdModel;
	loadModelFile(modelFile, npdModel);

	// load shape_predictor
	shape_predictor sp;
	std::ifstream fin(spFile, std::ios::binary);
	deserialize(sp, fin);

	// load image and turn it to gray
	cv::Mat img = cv::imread(imgFile);
	cv::Mat grayImg(img.size(), CV_8UC1);
	//cvtColor(img,grayImg,cv::COLOR_RGB2GRAY);
	for (int i = 0; i < img.rows; i++){
		for (int j = 0; j < img.cols; j++){
			double value = 0.2989 * img.at<cv::Vec3b>(i, j)[2]
				+ 0.5870 * img.at<cv::Vec3b>(i, j)[1]
				+ 0.1140 * img.at<cv::Vec3b>(i, j)[0];
			grayImg.at<uchar>(i, j) = (uchar)round(value);
		}
	}


	// detect faces
	vector<cv::Rect> rects;
	clock_t start, end;
	start = clock();
	DetectFace(rects, npdModel, grayImg);
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
			obj shape = sp(grayImg, rects[j]);
			PlotShape(img, shape.parts);
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