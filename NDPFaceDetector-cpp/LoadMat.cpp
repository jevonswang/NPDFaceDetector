#include "LoadMat.h"

bool loadMat(ifstream &in, arma::mat &M){

	string line;
	int row, col;
	istringstream ss;
	string tmpStr;

	// get stageThreshold
	getline(in, line);
	ss.clear();
	ss.str(line);
	ss >> row >> col;
	//cout << row << " " << col << endl;
	for (int i = 0; i < row; i++){
		getline(in, line);
		tmpStr += line + ";";
	}
	M = tmpStr;

	//cout << arma::size(M) << endl;
	return true;
}

bool loadModelFile(string &modelFile, NPDModel &npdModel){

	ifstream in(modelFile.c_str(), ios_base::in);
	if (!in.is_open())
	{
		cout << "Model not found." << endl;
		return false;
	}

	string line;
	istringstream ss;

	getline(in,line);
	ss.str(line);
	ss >> npdModel.objSize;
	ss >> npdModel.numStages;
	ss >> npdModel.numBranchNodes;
	ss >> npdModel.numLeafNodes;
	ss >> npdModel.scaleFactor;
	ss >> npdModel.numScales;

	loadMat(in,npdModel.stageThreshold);
	loadMat(in, npdModel.treeRoot);
	loadMat(in, npdModel.pixel1);
	loadMat(in, npdModel.pixel2);
	loadMat(in, npdModel.cutpoint);
	loadMat(in, npdModel.leftChild);
	loadMat(in, npdModel.rightChild);
	loadMat(in, npdModel.fit);
	loadMat(in, npdModel.npdTable);
	loadMat(in, npdModel.winSize);

	return true;
}