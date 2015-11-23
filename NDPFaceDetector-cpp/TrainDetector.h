#pragma once
#include "structs.h"
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;

NPDModel TrainDetector(string faceDBFile, string nonfaceDBFile, string outFile, Options options);