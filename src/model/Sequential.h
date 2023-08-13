#pragma once

#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <numeric>

#include "AbstractModel.h"
#include "../layers/AbstractLayer.h"
#include "../cost/AbstractCost.h"
#include "../optimization/AbstractOptimizer.h"

class Sequential : public AbstractModel
{

public:
    // Add a layer to the model
    void addLayer(std::shared_ptr<AbstractLayer> layer);

    void compile(std::shared_ptr<AbstractCost> loss, std::shared_ptr<AbstractOptimizer> optimizer);

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;

    void backward(const Eigen::MatrixXf& dOutput, double learningRate) override;

    void train(const Eigen::MatrixXf& data, const Eigen::MatrixXf& target, int epochs, double learningRate, int batchSize = 32) override;

    Eigen::MatrixXf predict(const Eigen::MatrixXf& input) override;

    void saveModel(const std::string& filepath) override;

    void loadModel(const std::string& filepath) override;

private:

    std::pair<int, int> getBatchSize(int batchSize, int nSamples);
    void setDropoutOnTrain();
    void setDropoutOnPredict();

    std::vector<std::shared_ptr<AbstractLayer>> layers;
    std::shared_ptr<AbstractCost> lossFunction;
    std::shared_ptr<AbstractOptimizer> optimizer;
};