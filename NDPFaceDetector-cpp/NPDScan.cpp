#include "NPDScan.h"

//void NDPScan(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
bool NPDScan(arma::mat &candi_rects, NPDModel &npdModel, arma::Mat<uchar> imgMat, int minFace=40, int maxFace=3000, int numThreads=4){

	/*
	candi_rects
		<< 345 << 291 << 29 << 8.72495 << arma::endr
		<< 343 << 289 << 35 << 7.31841 << arma::endr
		<< 249 << 265 << 86 << 12.6732 << arma::endr
		<< 253 << 265 << 86 << 7.31705 << arma::endr
		<< 257 << 265 << 86 << 7.48613 << arma::endr
		<< 253 << 269 << 86 << 10.2783 << arma::endr
		<< 257 << 269 << 86 << 11.3665 << arma::endr
		<< 261 << 269 << 86 << 8.71551 << arma::endr
		<< 257 << 273 << 86 << 11.4430 << arma::endr
		<< 251 << 251 << 103 << 8.96759 << arma::endr
		<< 241 << 256 << 103 << 9.88374 << arma::endr
		<< 246 << 256 << 103 << 14.7536 << arma::endr
		<< 251 << 256 << 103 << 9.85131 << arma::endr
		<< 241 << 261 << 103 << 10.9335 << arma::endr
		<< 246 << 261 << 103 << 9.62250 << arma::endr
		<< 251 << 261 << 103 << 11.6838 << arma::endr
		<< 256 << 261 << 103 << 8.80227 << arma::endr
		<< 246 << 266 << 103 << 10.9474 << arma::endr
		<< 235 << 241 << 124 << 11.4615 << arma::endr
		<< 229 << 247 << 124 << 7.55402 << arma::endr
		<< 235 << 247 << 124 << 16.6715 << arma::endr
		<< 247 << 247 << 124 << 12.8067 << arma::endr
		<< 232 << 225 << 149 << 7.88044 << arma::endr
		<< 239 << 225 << 149 << 8.81577 << arma::endr
		<< 246 << 225 << 149 << 8.73358 << arma::endr
		<< 232 << 232 << 149 << 7.79533 << arma::endr
		<< 239 << 232 << 149 << 11.9822 << arma::endr
		<< 225 << 239 << 149 << 11.4714 << arma::endr
		<< 232 << 239 << 149 << 15.4952 << arma::endr
		<< 239 << 239 << 149 << 8.59414 << arma::endr
		<< 233 << 225 << 178 << 8.12269 << arma::endr
		<< 225 << 233 << 178 << 9.48288 << arma::endr;
	*/

	
	// Set the number of threads
	//int numProcs = omp_get_num_procs();
	int numProcs = 4;
	if (numThreads > numProcs) numThreads = numProcs;



	//omp_set_num_threads(numThreads);
	//printf("minFace=%d, maxFace=%d, numThreads=%d\n", minFace, maxFace, numThreads);


	// get the NPD detector

	// get objSize
	int objSize = npdModel.objSize;
	//cout << "objSize: " << objSize << endl;
	
	// get numStages
	int numStages = npdModel.numStages;
	//cout << "numStages: " << numStages << endl;

	// get numLeafNodes
	int numLeafNodes = npdModel.numLeafNodes;
	//cout << "numLeafNodes: " << numLeafNodes << endl;
	
	// get numBranchNodes
	int numBranchNodes = npdModel.numBranchNodes;
	//cout << "numBranchNodes: " << numBranchNodes << endl;

	// get stageThreshold
	const double *pStageThreshold = npdModel.stageThreshold.memptr();
	//cout << "stageThreshold:" << endl;
	//for (int i = 0; i < numStages; i++){
	//	cout << *(pStageThreshold++) << endl;
	//}

	// get treeRoot
	const int *pTreeRoot = (int *)npdModel.treeRoot.memptr();
	//cout << "treeRoot:" << endl;
	//for (int i = 0; i < numStages; i++){
	//	cout << *(pTreeRoot++) << endl;
	//}

	// get numScales
	int numScales = npdModel.numScales;
	//cout << "numScales: " << numScales << endl;

	// get pixel1 and pixel2
	vector<int *> ppPoints1(numScales);
	vector<int *> ppPoints2(numScales);
	arma::umat pixel1 = npdModel.pixel1.t();
	arma::umat pixel2 = npdModel.pixel2.t();
	ppPoints1[0] = (int *)pixel1.memptr();
	ppPoints2[0] = (int *)pixel2.memptr();
	for (int i = 1; i < numScales; i++)
	{
		ppPoints1[i] = ppPoints1[i - 1] + numBranchNodes;
		ppPoints2[i] = ppPoints2[i - 1] + numBranchNodes;
	}

	// get cutpoint
	const int* ppCutpoint[2];
	arma::umat cutpoint = npdModel.cutpoint.t();
	ppCutpoint[0] = (int *)cutpoint.memptr();
	ppCutpoint[1] = ppCutpoint[0] + numBranchNodes;

	// get leftChild, rightChild and fit
	const int *pLeftChild = (int *)npdModel.leftChild.memptr();
	const int *pRightChild = (int *)npdModel.rightChild.memptr();
	const double *pFit = (double *)npdModel.fit.memptr();

	// get npdTable
	vector<int *> ppNpdTable(256);
	arma::umat npdTable = npdModel.npdTable.t();
	ppNpdTable[0] = (int *)npdTable.memptr();
	for (int i = 1; i < 256; i++) ppNpdTable[i] = ppNpdTable[i - 1] + 256;

	// get scaleFactor
	double scaleFactor = npdModel.scaleFactor;
		
	// get winSize
	const int *pWinSize = (int *)npdModel.winSize.memptr();
		
	int height = imgMat.n_rows;
	int width = imgMat.n_cols;

	//arma::Mat<uchar> transI = imgMat.t();
	const unsigned char *I = imgMat.memptr();
	

	minFace = max(minFace, objSize);
	maxFace = min(maxFace, min(height, width));

	if (min(height, width) < minFace){
		return true;
	}

	// containers for the detected faces
	vector<double> row, col, size, score;

	for (int k = 0; k < numScales; k++) // process each scale
	{
		if (pWinSize[k] < minFace) continue;
		else if (pWinSize[k] > maxFace) break;

		// determine the step of the sliding subwindow
		int winStep = (int)floor(pWinSize[k] * 0.1);
		if (pWinSize[k] > 40) winStep = (int)floor(pWinSize[k] * 0.05);

		// calculate the offset values of each pixel in a subwindow
		// pre-determined offset of pixels in a subwindow
		vector<int> offset(pWinSize[k] * pWinSize[k]);
		int p1 = 0, p2 = 0, gap = height - pWinSize[k];

		for (int j = 0; j < pWinSize[k]; j++) // column coordinate
		{
			for (int i = 0; i < pWinSize[k]; i++) // row coordinate
			{
				offset[p1++] = p2++;
			}

			p2 += gap;
		}

		int colMax = width - pWinSize[k] + 1;
		int rowMax = height - pWinSize[k] + 1;

	//#pragma omp parallel for //private(c, pPixel, r, treeIndex, _score, s, node, p1, p2, fea, _row, _col, _size)

		// process each subwindow
		for (int c = 0; c < colMax; c += winStep) // slide in column
		{
			const unsigned char *pPixel = I + c * height;

			for (int r = 0; r < rowMax; r += winStep, pPixel += winStep) // slide in row
			{
				int treeIndex = 0;
				float _score = 0;
				int s;

				// test each tree classifier
				for (s = 0; s < numStages; s++)
				{
					int node = pTreeRoot[treeIndex];

					// test the current tree classifier
					while (node > -1) // branch node
					{
						cout << ppPoints1[k][node] << endl;
						cout << ppPoints2[k][node] << endl;

						cout << offset[ppPoints1[k][node]] << endl;
						cout << offset[ppPoints2[k][node]] << endl;

						unsigned char p1 = pPixel[offset[ppPoints1[k][node]]];
						unsigned char p2 = pPixel[offset[ppPoints2[k][node]]];
						unsigned char fea = ppNpdTable[p1][p2];
						//printf("node = %d, fea = %d, cutpoint = (%d, %d)\n", node, int(fea), int(ppCutpoint[0][node]), int(ppCutpoint[1][node]));

						if (fea < ppCutpoint[0][node] || fea > ppCutpoint[1][node]) node = pLeftChild[node];
						else node = pRightChild[node];
					}

					// leaf node
					node = -node - 1;
					_score = _score + pFit[node];
					treeIndex++;

					//printf("stage = %d, score = %f\n", s, _score);
					if (_score < pStageThreshold[s]) break; // negative samples
				}

				if (s == numStages) // a face detected
				{
					double _row = r + 1;
					double _col = c + 1;
					double _size = pWinSize[k];

	//#pragma omp critical // modify the record by a single thread
					
					row.push_back(_row);
					col.push_back(_col);
					size.push_back(_size);
					score.push_back(_score);
					
				}
			}
		}
	}

	int numFaces = (int)row.size();

	//if (numFaces == 0) return true;
	candi_rects.set_size(numFaces, 4);
	for (int i = 0; i < numFaces; i++){
		candi_rects(i, 0) = row.at(i);
		candi_rects(i, 1) = col.at(i);
		candi_rects(i, 2) = size.at(i);
		candi_rects(i, 3) = score.at(i);
	}

	candi_rects.print("candi_rects:");
	
	return true;
}
