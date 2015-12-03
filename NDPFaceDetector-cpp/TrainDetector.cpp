/*
@Filename: TrainDetector.cpp
@Author: Zhefeng Wang(jevons.wz@gmail.com)
@Created on: 2015/11/23
@Modified on: 2015/11/24
@Version: 1.0
@desc:
	Train a Nomalized Pixel Difference(NPD) based face detector.
@param
	<faceDBFile> : MAT file for the face images.It contains an array FaceDB
				   of size[objSize, objSize, numFaces].
	<nonfaceDBFile> : MAT file for the nonface images.It contains the following variables :
		numSamples : the number of cropped nonface images of size[objSize,objSize].
		numNonfaceImgs : the number of big nonface images for bootstrapping.
		NonfaceDB : an array of size[objSize, objSize, numSamples]
					containing the cropped nonface images.This is used in the
		            begining stages of the detector training.
		NonfaceImages : a cell of size[numNonfaceImgs, 1] containing the
						big nonface images for bootstrapping.
	<outFile> : the output file to store the training result.
	[optioins] : optional parameters.See the beginning codes of this function
				 for the parameter meanings and the default values.
@return
	<model> : output of the trained detector.
*/


#include "TrainDetector.h"

cv::Mat loadFaceDBFile(const string &faceDBFile){
	cv::Mat mat;
	return mat;
}


NPDModel TrainDetector(string faceDBFile,
	string nonfaceDBFile, string outFile, Options options){

	NPDModel npdModel;

	return npdModel;
}