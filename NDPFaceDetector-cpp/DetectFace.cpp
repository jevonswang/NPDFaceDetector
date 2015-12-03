#include "DetectFace.h"


bool Logistic(arma::vec &candi_rects_score, arma::vec &weight){
	weight = arma::log(1 + arma::exp(candi_rects_score));
	/*
	for (int i = 0; i < candi_rects_score.n_elem; i++){
		weight[i] = log(1 + exp(candi_rects_score[i]));
	}
	*/

	return true;
}

bool DetectFace(vector<cv::Rect> &detected_rects, NPDModel &npdModel, cv::Mat &img){

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
	NPDScan(candi_rects, npdModel, armaImg, minFace, maxFace, numThreads);

	//candi_rects.print("candi_rects:");

	int numCandidates = candi_rects.n_rows;
	if (0 == numCandidates){
		detected_rects.clear();
		return true;
	}

	
	arma::mat predicate = arma::eye<arma::mat>(numCandidates,numCandidates);
	// i and j belong to the same group if predicate(i,j) = 1

	// mark nearby detections
	int h, w, s;
	//int count=0;
	for (int i = 0; i < numCandidates; i++){
		for (int j = i+1; j < numCandidates; j++){
			
			h = min(candi_rects(i,0) + candi_rects(i,2), candi_rects(j,0) + candi_rects(j,2))
				- max(candi_rects(i,0), candi_rects(j,0));
			w = min(candi_rects(i,1) + candi_rects(i,2), candi_rects(j,1) + candi_rects(j,2))
				- max(candi_rects(i,1), candi_rects(j,1));
			s = max(h, 0) * max(w, 0);

			//printf("%d: h=%d,w=%d,s=%d\n",count,h,w,s);
			//count++;

			if (s / (candi_rects(i,2) * candi_rects(i,2) + candi_rects(j,2) * candi_rects(j,2) - s) >= overlappingThreshold){
				predicate(i, j) = 1;
				predicate(j, i) = 1;
			}
		}
	}

	//predicate.print("predicate = ");
	
	// merge nearby detections
	arma::uvec label;

	Partition(predicate,label,numCandidates);
	

	arma::mat rects;
	rects.zeros(numCandidates, 7);
	// 0:row 1:col 2:size 3:score 4:neighbors 5:height 6:width


	for (int i = 0; i < numCandidates; i++){

		arma::uvec index = arma::find(label == i+1);

		// matlab: weight = Logistic([candi_rects(index).score]');
		arma::mat candi_rects_index = candi_rects.rows(index);
		arma::vec candi_rects_index_score = candi_rects_index.col(3);
		arma::vec weight;

		candi_rects_index_score.print();

		Logistic(candi_rects_index_score, weight);

		

		rects(i, 3) = arma::sum(weight); // scores
		rects(i, 4) = index.n_elem;  // neighbors

		if (0 == arma::sum(weight)){
			int n_elem = weight.n_elem;
			weight.ones(n_elem,1);
			weight = weight / n_elem;
		}
		else{
			weight = weight / arma::sum(weight);
		}

		
		//weight.print("weight:");
		//candi_rects_index.print("candi_rects_index");


		arma::mat ans = candi_rects_index.col(2).t() * weight;
		rects(i, 2) = floor(ans(0,0));
		
		ans.clear();
		ans = (candi_rects_index.col(1) + candi_rects_index.col(2) / 2).t() * weight - rects(i,2) / 2;
		rects(i, 1) = floor(ans(0,0));
		
		ans.clear();
		ans = (candi_rects_index.col(0) + candi_rects_index.col(2) / 2).t() * weight - rects(i,2) / 2;
		rects(i, 0) = floor(ans(0,0));
		
		cout << "i: " << i << endl;
		//rects.print("rects:");
	}

	candi_rects.clear();

	
	// find embeded rectangles
	predicate.clear();
	predicate.zeros(numCandidates,numCandidates);
	//predicate.print("predicate:");

	for (int i = 0; i < numCandidates; i++){
		for (int j = i + 1; j < numCandidates; j++){
			h = min(rects(i, 0) + rects(i, 2), rects(j, 0) + rects(j, 2))
				- max(rects(i, 0), rects(j, 0));
			w = min(rects(i, 1) + rects(i, 2), rects(j, 1) + rects(j, 2))
				- max(rects(i, 1), rects(j, 1));
			s = max(h, 0) * max(w, 0);

			if ((s / (rects(i, 2) * rects(i, 2)) >= overlappingThreshold) || s /(rects(j,2)*rects(j,2)) >= overlappingThreshold){
				predicate(i, j) = 1;
				predicate(j, i) = 1;
			}
		}
	}

	//predicate.print("predicate:");

	
	arma::uvec flag;
	flag.ones(numCandidates,1);

	// merge embeded rectangles
	for (int i = 0; i < numCandidates; i++){
		arma::uvec index = arma::find(predicate.col(i));

		if (index.is_empty()){
			continue;
		}

		arma::mat rects_rows_index = rects.rows(index);
		double s = max(rects_rows_index.col(3));
		if (s>rects(i, 3)){
			flag(i) = 0;
		}
	}
	
	rects = rects.rows(find(flag != 0));
	

	// check borders
	int height = img.rows;
	int width = img.cols;
	int numFaces = rects.n_rows;

	for (int i = 0; i < numFaces; i++){
		if (rects(i, 0) < 1){
			rects(i, 0) = 1;
		}
		if (rects(i, 1) < 1){
			rects(i, 1) = 1;
		}

		rects(i, 5) = rects(i, 2);
		rects(i, 6) = rects(i, 2);

		if (rects(i,0) + rects(i,5) - 1 > height){
			rects(i,5) = height - rects(i,0) + 1;
		}

		if (rects(i, 1) + rects(i,6) - 1 > width){
			rects(i,6) = width - rects(i,1) + 1;
		}
	}

	rects.print("rects:");
	for (int i = 0; i < rects.n_rows; i++){
		cv::Rect r(rects(i,1),rects(i,0),rects(i,6),rects(i,5));
		detected_rects.push_back(r);
	}
	
	return true;
}