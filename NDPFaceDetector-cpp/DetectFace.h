#pragma once

#include <armadillo>
#include <opencv2/opencv.hpp>
#include <vector>

#include "structs.h"

using namespace std;

bool DetectFace(vector<cv::Rect> &rects,const NPDModel &npdModel, const cv::Mat &img);
