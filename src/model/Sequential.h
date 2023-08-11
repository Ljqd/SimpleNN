#pragma once

#include "AbstractModel.h"
#include "../layers/AbstractLayer.h"
#include "../cost/AbstractCost.h"

class Sequential : public AbstractModel
{

public:
    // Add a layer to the model
    void addLayer(std::shared_ptr<AbstractLayer> layer);

    void compile(std::shared_ptr<AbstractCost> loss);

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;

    void backward(const Eigen::MatrixXf& dOutput, double learningRate) override;

    void train(const Eigen::MatrixXf& data, const Eigen::MatrixXf& target, int epochs, double learningRate, int batch_size = 32) override;

    Eigen::MatrixXf predict(const Eigen::MatrixXf& input) override;

    void saveModel(const std::string& filepath) override;

    void loadModel(const std::string& filepath) override;

private:
    std::vector<std::shared_ptr<AbstractLayer>> layers;
    std::shared_ptr<AbstractCost> lossFunction;
};