#include "LoadMat.h"

bool loadModelFile(string &modelFile, NPDModel &npdModel){

	ifstream in("model_frontal.txt");
	
	if (!in.is_open()){
		cout << "Model not found!." <<endl;
	}

	string line;
	char buffer[1000];
	int lineCount = 0;
	while (!in.eof()){
		in.getline(buffer, 1000);
		cout << buffer << endl;
		lineCount++;
		if (lineCount > 100)break;
	}
	




	return true;
}