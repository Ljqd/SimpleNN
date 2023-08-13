#pragma once

#include <cmath>
#include <matplot/matplot.h>

#include <Eigen/Dense>

#include "../src/model/Sequential.h"

#include "../src/layers/Dense.h"

#include "../src/activation/Relu.h"
#include "../src/activation/Tanh.h"
#include "../src/activation/Sigmoid.h"

#include "../src/cost/MeanSquareError.h"
#include "../src/cost/BinaryCrossEntropy.h"

#include "../src/optimization/GradientDescent.h"

void numeric_binary_classification();