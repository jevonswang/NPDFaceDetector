#include "DetectFace.h"

bool DetectFace(
	vector<cv::Rect> &rects, 
	NPDModel &npdModel, 
	cv::Mat &img){

	// initialize parameters
	int minFace = 20;
	int maxFace = 4000;
	double overlappingThreshold = 0.5;
	int numThreads = 24;

	// trun the img to gray
	cv::Mat grayImg;
	grayImg.create(img.size(),img.type());
	cvtColor(img,grayImg,cv::COLOR_RGB2GRAY);

	// convert cv::Mat to arma::Mat<uchar>
	arma::Mat<uchar> armaImg(grayImg.data, grayImg.rows, grayImg.cols);
	arma::inplace_trans(armaImg);

	// compare whether I equals grayImg
	//I(0,0,arma::size(10,10)).print("I=");
	//cout << grayImg(cv::Range(0, 10), cv::Range(0, 10)) << endl;

	// get candidate rects
	vector<Candi_rects> candi_rects;
	NPDScan(candi_rects,npdModel, armaImg, minFace, maxFace, numThreads);

	
	int numCandidates = rects.size();
	if (0 == numCandidates){
		rects.clear();
		return true;
	}

	arma::mat predicate = arma::eye<arma::mat>(numCandidates,numCandidates);
	// i and j belong to the same group if predicate(i,j) = 1

	int h, w, s;
	for (int i = 0; i < numCandidates; i++){
		for (int j = 0; j < numCandidates; j++){

		}
	}


	return true;
}