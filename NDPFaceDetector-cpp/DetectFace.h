#pragma once

#include <armadillo>
#include <opencv2/opencv.hpp>
#include <vector>

#include "structs.h"
#include "NPDScan.h"

using namespace std;

bool DetectFace(vector<cv::Rect> &rects,NPDModel &npdModel, cv::Mat &img);
