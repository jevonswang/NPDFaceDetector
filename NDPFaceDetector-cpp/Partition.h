#pragma once

#include <armadillo>
#include <opencv2/opencv.hpp>

#include "structs.h"
using namespace std;

bool Partition(arma::mat &A, arma::uvec &label, int &nGroups);