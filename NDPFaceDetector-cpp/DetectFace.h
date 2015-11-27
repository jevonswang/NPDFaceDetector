#pragma once

#include <armadillo>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>

#include "structs.h"
#include "NPDScan.h"
#include "Partition.h"

using namespace std;

bool DetectFace(arma::mat &rects,NPDModel &npdModel, cv::Mat &img);
