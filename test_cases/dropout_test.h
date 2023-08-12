#pragma once

#include <cmath>
#include <matplot/matplot.h>

#include <Eigen/Dense>

#include "../src/model/Sequential.h"

#include "../src/layers/Dense.h"
#include "../src/layers/Dropout.h"

#include "../src/activation/Relu.h"
#include "../src/activation/Tanh.h"
#include "../src/activation/Sigmoid.h"

#include "../src/cost/MeanSquareError.h"
#include "../src/cost/BinaryCrossEntropy.h"

void dropout_test();