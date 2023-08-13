#pragma once

#include "AbstractLayer.h"

class Dropout : public AbstractLayer
{
public:
    Dropout(double rate);

    enum DropoutMode
    {
        training, prediction
    };

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& dOutput) override;

    void updateWeights() override {};
    void setOptimizer(std::shared_ptr<AbstractOptimizer> optimizer) override {};

    void setTrainingMode(DropoutMode mode);

private:
    double dropoutRate;
    Eigen::MatrixXf mask;
    bool mode;
};

