#pragma once

#include <Eigen/Dense>

class AbstractModel
{
public:
    virtual ~AbstractModel() {}

    // Forward pass through the model
    virtual Eigen::MatrixXf forward(const Eigen::MatrixXf& input) = 0;

    // Backward pass through the model for gradient computation
    virtual void backward(const Eigen::MatrixXf& dOutput, double learningRate) = 0;

    // Train the model given input data and corresponding target output
    virtual void train(const Eigen::MatrixXf& data, const Eigen::MatrixXf& target, int epochs, double learningRate, int batch_size=32) = 0;

    // Predict the output for a given input
    virtual Eigen::MatrixXf predict(const Eigen::MatrixXf& input) = 0;

    // Save the model's parameters to a file
    virtual void saveModel(const std::string& filepath) = 0;

    // Load the model's parameters from a file
    virtual void loadModel(const std::string& filepath) = 0;
};

