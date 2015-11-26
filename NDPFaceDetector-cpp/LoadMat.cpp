#include "LoadMat.h"

bool loadModelFile(string &modelFile, NPDModel &npdModel){

	ifstream in(modelFile.c_str(), ios_base::in);
	if (!in.is_open())
	{
		cout << "Model not found." << endl;
		return false;
	}

	string line;
	
	getline(in,line);

	istringstream ss;
	ss.str(line);
	ss >> npdModel.objSize;
	ss >> npdModel.numStages;
	ss >> npdModel.numBranchNodes;
	ss >> npdModel.numLeafNodes;
	ss >> npdModel.scaleFactor;
	ss >> npdModel.numScales;

	int row, col;
	getline(in, line);
	ss.str(line);
	ss >> row >> col;
	for (int i = 0; i < row; i++){
		getline(in, line);
		ss.str(line);
		for (int j = 0; j < col; j++){
			ss >> npdModel.stageThreshold(i,j);
		}
	}


	return true;
}