#include "numeric_binary_classification.h"

void numeric_binary_classification() {
    // Generate synthetic data
    int nSamples = 1000;
    Eigen::MatrixXf X(nSamples, 2);
    Eigen::MatrixXf Y(nSamples, 1);

    int n = 4; // For instance, this would produce a 4-petal rose if n is even, or 8 petals

    for (int i = 0; i < nSamples; i++)
    {
        double theta = (rand() / (double)RAND_MAX) * 2 * 3.147; // random angle between 0 and 2*pi
        double r = cos(n * theta); // rose curve radius

        X(i, 0) = r * cos(theta);
        X(i, 1) = r * sin(theta);

        int petal = floor(n * theta / (2 * 3.147)); // this gives us which petal the point is closest to
        Y(i, 0) = (petal % 2 == 0) ? 1 : 0; // even petals labeled as 1 and odd as 0
    }

    std::cout << "Dataset: 1-labels ~ " << Y.sum() << ";    0-labels ~ " << nSamples - Y.sum() << ";" << std::endl;

    // Convert Eigen matrices to std::vectors for Matplot++ compatibility
    if (true)
    {
        std::vector<double> x_0, y_0, x_1, y_1;

        for (int i = 0; i < nSamples; ++i) {
            if (Y(i, 0) == 0) {
                x_0.push_back(X(i, 0));
                y_0.push_back(X(i, 1));
            }
            else {
                x_1.push_back(X(i, 0));
                y_1.push_back(X(i, 1));
            }
        }

        // Plot using Matplot++
        matplot::hold(true);
        matplot::plot(x_0, y_0, "ro");
        matplot::plot(x_1, y_1, "bo");
        matplot::xlabel("X axis");
        matplot::ylabel("Y axis");
        matplot::title("2D Binary Classification Data");
        matplot::grid(true);
        matplot::show();
    }

    Sequential model;

    // Add a Dense layer
    model.addLayer(std::make_shared<Dense>(2, 4, std::make_shared<Tanh>(), "xavier"));
    model.addLayer(std::make_shared<Dense>(4, 1, std::make_shared<Sigmoid>()));

    // Compile the model
    model.compile(std::make_shared<BinaryCrossEntropy>());

    // Train the model
    model.train(X, Y, 3000, 0.1, -1);

    // Test
    Eigen::MatrixXf sample(1, 2);
    sample << 0.5, 0.5;
    Eigen::MatrixXf prediction = model.predict(sample);
    std::cout << "Prediction for (0.5, 0.5): " << (prediction.coeff(0, 0)) << std::endl;

    sample << -0.5, -0.5;
    prediction = model.predict(sample);
    std::cout << "Prediction for (-0.5, -0.5): " << (prediction.coeff(0, 0)) << std::endl;

    sample << 0.5, -0.5;
    prediction = model.predict(sample);
    std::cout << "Prediction for (0.5, -0.5): " << (prediction.coeff(0, 0)) << std::endl;

    sample << -0.5, 0.5;
    prediction = model.predict(sample);
    std::cout << "Prediction for (-0.5, 0.5): " << (prediction.coeff(0, 0)) << std::endl;
}