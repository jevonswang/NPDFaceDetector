#pragma once

#include <armadillo>
#include <opencv2/opencv.hpp>

#include "structs.h"

bool Partition(arma::mat &predicate, arma::uvec &label, int &numCandidates);