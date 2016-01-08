#include <iostream>
#include <vector>
#include <string>
#include <io.h>
#include <armadillo>
#include <opencv2/opencv.hpp>


//#include "f2c.h"
//#include "clapack.h"
#include "structs.h"
#include "DetectFace.h"
#include "LoadMat.h"
#include "ShapePredictor.h"

//using namespace std;
//using namespace Eigen;

void PlotShape(cv::Mat& im, vector<Eigen::Vector2f> s){
	for (int i = 0; i<s.size(); i++)
		circle(im, cv::Point2f(s[i](0), s[i](1)), 3, CV_RGB(255, 0, 0));
}

// get all files of some certain type in the directory
void LoadImages(string dirName, vector<cv::Mat> &images){

	_finddata_t fileDir;
	if (dirName[dirName.size() - 1] != '\\') dirName += "\\";
	string fullPath = dirName + "*.jpg";
	long lfDir;
	string imgName;
	cv::Mat image;

	if ((lfDir = _findfirst(fullPath.c_str(), &fileDir)) == -1l){
		cout << "No files founded." << endl;
		return;
	}
	else{
		//cout << "File in \"" << dirName << "\" with type " << type << ":" << endl;
		do{
			cout << fileDir.name << endl;
			imgName = dirName + string(fileDir.name);
			image = cv::imread(imgName);
			images.push_back(image);
		} while (_findnext(lfDir, &fileDir) == 0);
	}
	_findclose(lfDir);
}

void runDetect(){

	string modelFile = "F:\\NDPData\\model\\model_frontal.txt";
	//string modelFile = "F:\\NDPData\\model\\model_unconstrain.txt";
	string imgFile = "F:\\NDPData\\images\\";
	string spFile = "F:\\NDPData\\model\\sp68.dat";

	// load npdModel
	NPDModel npdModel;
	loadModelFile(modelFile, npdModel);

	// load shape_predictor
	shape_predictor sp;
	std::ifstream fin(spFile, std::ios::binary);
	deserialize(sp, fin);

	vector<cv::Mat> images;
	LoadImages(imgFile, images);

	cv::VideoCapture videoCapture(0);

	while (1){
	//for (int i = 0; i < images.size(); i++){

		cv::Mat img;
		videoCapture >> img;
		
		// load image and turn it to gray
		//cv::Mat img = images[i];
		cv::Mat grayImg(img.size(), CV_8UC1);
		//cvtColor(img, grayImg, cv::COLOR_RGB2GRAY);
		
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
				start = clock();
				obj shape = sp(grayImg, rects[j]);
				PlotShape(img, shape.parts);
				end = clock();
				printf("alignment time:%f s\n", (dur / CLOCKS_PER_SEC));
				cv::rectangle(img, rects[j], cv::Scalar(0, 255, 0), border);
			}
		}

		cv::imshow("result", img);
		cv::waitKey(10);
	}
}

int main(int argc, char* argv[]){
	
	runDetect();
	return 0;
}