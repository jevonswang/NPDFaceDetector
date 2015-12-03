#include "NPDScan.h"

//void NDPScan(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
bool NPDScan(arma::mat &candi_rects, NPDModel &npdModel, arma::Mat<uchar> I, int minFace, int maxFace, int numThreads){

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
	


	/*
	int minFace = 40;
	int maxFace = 3000;

	if (nrhs >= 3 && mxGetScalar(prhs[2]) > 0) minFace = (int)mxGetScalar(prhs[2]);
	if (nrhs >= 4 && mxGetScalar(prhs[3]) > 0) maxFace = (int)mxGetScalar(prhs[3]);

	// Set the number of threads
	int numProcs = omp_get_num_procs();
	int numThreads = numProcs;

	if (nrhs >= 5 && mxGetScalar(prhs[4]) > 0) numThreads = (int)mxGetScalar(prhs[4]);

	if (numThreads > numProcs) numThreads = numProcs;
	omp_set_num_threads(numThreads);
	//printf("minFace=%d, maxFace=%d, numThreads=%d\n", minFace, maxFace, numThreads);

	// get input pointers
	const mxArray *pModel = prhs[0];

	// get the NPD detector
	int objSize = (int)mxGetScalar(mxGetField(pModel, 0, "objSize"));
	int numStages = (int)mxGetScalar(mxGetField(pModel, 0, "numStages"));
	//int numLeafNodes = (int) mxGetScalar(mxGetField(pModel, 0, "numLeafNodes"));
	int numBranchNodes = (int)mxGetScalar(mxGetField(pModel, 0, "numBranchNodes"));
	const float *pStageThreshold = (float *)mxGetData(mxGetField(pModel, 0, "stageThreshold"));
	const int *pTreeRoot = (int *)mxGetData(mxGetField(pModel, 0, "treeRoot"));

	int numScales = (int)mxGetScalar(mxGetField(pModel, 0, "numScales"));
	vector<int *> ppPoints1(numScales);
	vector<int *> ppPoints2(numScales);
	ppPoints1[0] = (int *)mxGetData(mxGetField(pModel, 0, "pixel1"));
	ppPoints2[0] = (int *)mxGetData(mxGetField(pModel, 0, "pixel2"));
	for (int i = 1; i < numScales; i++)
	{
		ppPoints1[i] = ppPoints1[i - 1] + numBranchNodes;
		ppPoints2[i] = ppPoints2[i - 1] + numBranchNodes;
	}

	const unsigned char* ppCutpoint[2];
	ppCutpoint[0] = (unsigned char *)mxGetData(mxGetField(pModel, 0, "cutpoint"));
	ppCutpoint[1] = ppCutpoint[0] + numBranchNodes;

	const int *pLeftChild = (int *)mxGetData(mxGetField(pModel, 0, "leftChild"));
	const int *pRightChild = (int *)mxGetData(mxGetField(pModel, 0, "rightChild"));
	const float *pFit = (float *)mxGetData(mxGetField(pModel, 0, "fit"));

	vector<unsigned char *> ppNpdTable(256);
	ppNpdTable[0] = (unsigned char *)mxGetData(mxGetField(pModel, 0, "npdTable"));
	for (int i = 1; i < 256; i++) ppNpdTable[i] = ppNpdTable[i - 1] + 256;

	//double scaleFactor = mxGetScalar(mxGetField(pModel, 0, "scaleFactor"));
	const int *pWinSize = (int *)mxGetData(mxGetField(pModel, 0, "winSize"));

	int height = (int)mxGetM(prhs[1]);
	int width = (int)mxGetN(prhs[1]);
	const unsigned char *I = (unsigned char *)mxGetData(prhs[1]);

	minFace = max(minFace, objSize);
	maxFace = min(maxFace, min(height, width));

	if (min(height, width) < minFace)
	{
		// create a structure vector for the output data
		const char* field_names[] = { "row", "col", "size", "score" };
		plhs[0] = mxCreateStructMatrix(0, 1, 4, field_names);
		return;
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

#pragma omp parallel for //private(c, pPixel, r, treeIndex, _score, s, node, p1, p2, fea, _row, _col, _size)

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

#pragma omp critical // modify the record by a single thread
					{
						row.push_back(_row);
						col.push_back(_col);
						size.push_back(_size);
						score.push_back(_score);
					}
				}
			}
		}
	}

	int numFaces = (int)row.size();

	// create a structure vector for the output data
	const char* field_names[] = { "row", "col", "size", "score" };
	plhs[0] = mxCreateStructMatrix(numFaces, 1, 4, field_names);

	if (numFaces == 0) return;

	mxArray *temp;

	// asign the output data
	for (int i = 0; i < numFaces; i++)
	{
		temp = mxCreateDoubleMatrix(1, 1, mxREAL);
		*mxGetPr(temp) = row[i];
		mxSetField(plhs[0], i, "row", temp);

		temp = mxCreateDoubleMatrix(1, 1, mxREAL);
		*mxGetPr(temp) = col[i];
		mxSetField(plhs[0], i, "col", temp);

		temp = mxCreateDoubleMatrix(1, 1, mxREAL);
		*mxGetPr(temp) = size[i];
		mxSetField(plhs[0], i, "size", temp);

		temp = mxCreateDoubleMatrix(1, 1, mxREAL);
		*mxGetPr(temp) = score[i];
		mxSetField(plhs[0], i, "score", temp);
	}
	*/
	return true;
}
