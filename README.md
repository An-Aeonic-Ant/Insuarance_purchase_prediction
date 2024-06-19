# Insurance_purchase_prediction
This repository contains a machine learning project to predict whether a customer will purchase insurance based on affordability and age using a single-layered neural network in Keras. The second part also contains the implementation of this model from scratch, including the gradient descent function, log loss, and sigmoid prediction for an under-the-hood understanding of how the model works.
The results and comparison of weights of the 2 models are displayed at the end

## Dependencies
- Pandas
- NumPy
- Scikit-Learn
- Tensorflow-Keras

## Implementation approach
A simple batch gradient descent function is implemented, which goes through the training data for the specified number of epochs and calculates the binary cross-entropy or the log loss. The derivative values of the loss are calculated and subtracted from the preset weights to reach closer to the minima and minimise the loss.


The weights are returned when the loss reaches below the loss threshold achieved by the Keras model.


Finally, the results are predicted using the returned weights and are scaled using the sigmoid function.


## Results and comparison
The implemented model was found to be very accurate and reached the loss threshold in less than 500 epochs as compared to the 5000 epochs required by the Keras model. However, this was only due to the much higher learning rate of our implementation(0.5 as compared to 0.01 of the Adam optimizer)


Given an infinite number of epochs, the Keras model would still be faster and more accurate than our implementation


