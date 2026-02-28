# DL-AI46-SV

I have implemented the binary classification pipeline using a Neural Network (Multi-Layer Perceptron) in PyTorch

Problem Description: Breast Cancer Classification
The objective is to classify whether a breast tumor is malignant (1) or benign (0)

Implementation Pipeline

Preprocess: The CSV is loaded using pandas. The id column is dropped, and the diagnosis target is mapped to binary values. The data is then split into Training, Validation, and Test sets and scaled using StandardScaler

Model Definition: A neural network with input, hidden, and output layers is defined, using ReLU activation for hidden layers and Sigmoid for the final output to produce a probability

Train & Validation: The model is trained using the Adam optimizer and Binary Cross-Entropy Loss; it is validated after every 10 epochs.

Test: The final model is evaluated on the unseen Test set
