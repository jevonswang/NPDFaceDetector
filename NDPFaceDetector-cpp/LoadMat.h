#pragma once
#include "structs.h"
#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include <armadillo>
using namespace std;

bool loadModelFile(string &modelFile, NPDModel &npdModel);
