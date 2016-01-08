#pragma once

#include <math.h>
#include <armadillo>
#include <opencv2/opencv.hpp>

#include "structs.h"
#include <omp.h>

using namespace std;

bool NPDScan(arma::mat &candi_rects, NPDModel &npdModel, arma::Mat<uchar> I, int minFace, int maxFace, int numThreads);
