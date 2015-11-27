#include "DetectFace.h"

bool Logistic(arma::uvec &candi_rects_score, arma::uvec &weight){
	for (int i = 0; i < candi_rects_score.n_elem; i++){
		weight[i] = log(1 + exp(candi_rects_score[i]));
	}
}

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
	arma::mat candi_rects;// col[0]:row,col[1]:col,col[2]:size,col[3]:score 
	NPDScan(candi_rects,npdModel, armaImg, minFace, maxFace, numThreads);

	
	int numCandidates = rects.size();
	if (0 == numCandidates){
		rects.clear();
		return true;
	}

	arma::mat predicate = arma::eye<arma::mat>(numCandidates,numCandidates);
	// i and j belong to the same group if predicate(i,j) = 1

	// mark nearby detections
	int h, w, s;
	for (int i = 0; i < numCandidates; i++){
		for (int j = 0; j < numCandidates; j++){
			
			h = min(candi_rects(i,0) + candi_rects(i,2), candi_rects(j,0) + candi_rects(j,2))
				- max(candi_rects(i,0), candi_rects(j,0));
			w = min(candi_rects(i,1) + candi_rects(i,2), candi_rects(j,1) + candi_rects(j,2))
				- max(candi_rects(i,1), candi_rects(j,1));
			s = max(h, 0) * max(w, 0);

			if ((s / candi_rects(i,2) * candi_rects(i,2) + candi_rects(j,2) * candi_rects(j,2) - s) >= overlappingThreshold){
				predicate(i, j) = 1;
				predicate(j, i) = 1;
			}
		}
	}

	// merge nearby detections
	arma::mat label;
	Partition(predicate,label,numCandidates);

	arma::uvec rects_row, rects_col, rects_size, rects_score, rects_neighbors;
	rects_row.zeros(numCandidates, 1);
	rects_col.zeros(numCandidates, 1);
	rects_size.zeros(numCandidates, 1);
	rects_score.zeros(numCandidates, 1);
	rects_neighbors.zeros(numCandidates, 1);


	for (int i = 0; i < numCandidates; i++){
		arma::uvec index = arma::find(label == i);

		arma::uvec candi_rects_score(index);
		for (int j = 0; j < index.n_elem; j++){
			candi_rects_score[j] = candi_rects[j].score;
		}
		
		candi_rects(index).col(5);


		arma::uvec weight(candi_rects);
		Logistic(candi_rects_score, weight);
		rects_score[i] = arma::sum(weight);
		rects_neighbors[i] = index.n_elem;

		if (0 == rects_score[i]){
			int n_elem = weight.n_elem;
			weight.ones();
			weight = weight / n_elem;
		}
		else{
			weight = weight / arma::sum(weight);
		}

		rects_size[i] = floor([candi_rects(index).size] * weight);
		rects_col[i] = floor(([candi_rects(index).col] + [candi_rects(index).size] / 2) * weight - rects(i).size / 2);
		rects_row[i] = floor(([candi_rects(index).row] + [candi_rects(index).size] / 2) * weight - rects(i).size / 2);
	}













	return true;
}

