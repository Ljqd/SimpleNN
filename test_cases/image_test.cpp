#include "image_test.h"

void image_test() 
{
    /*
    ImageLoader loader("./resources/images/cats_dogs_light");
    loader.load();

    std::vector<Eigen::MatrixXf> trainImages = loader.getTrainImages();
    std::vector<Eigen::MatrixXf> testImages = loader.getTestImages();

    std::vector<int> trainLabels = loader.getTrainLabels();
    std::vector<int> testLabels = loader.getTestLabels();

    DataSet datasets[2];

    for (int dataSetIndex = 0; dataSetIndex < 2; dataSetIndex++) {
        std::vector<Eigen::MatrixXf>& images = (dataSetIndex == 0) ? trainImages : testImages;
        std::vector<int>& vec = (dataSetIndex == 0) ? trainLabels : testLabels;

        int rows = images.size();
        int cols = images[0].cols();

        datasets[dataSetIndex].batch.resize(rows, cols);
        datasets[dataSetIndex].labels.resize(rows, 1);

        for (int rowIndex = 0; rowIndex < rows; ++rowIndex) {
            datasets[dataSetIndex].batch.row(rowIndex) = images[rowIndex].row(0);
            datasets[dataSetIndex].labels(rowIndex, 0) = vec[rowIndex];
        }
    }

    Sequential model;

    // Add a Dense layer
    model.addLayer(std::make_shared<Flatten>());
    model.addLayer(std::make_shared<Dense>(2, 4, std::make_shared<Tanh>()));
    model.addLayer(std::make_shared<Dense>(4, 1, std::make_shared<Sigmoid>()));

    // Compile the model
    model.compile(std::make_shared<BinaryCrossEntropy>());

    // Train the model
    model.train(datasets[0].batch, datasets[0].labels, 3000, 0.1);

    // Test
    */
}