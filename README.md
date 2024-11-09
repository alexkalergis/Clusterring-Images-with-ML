## Project Overview

The tasks in this project include:
1. Implementing a Bayes optimal test for a given classification problem.
2. Generating synthetic data for two classes and evaluating the Bayes classifier's error rate.
3. Training neural networks on generated data using cross-entropy, exponential, and hinge methods.
4. Comparing the performance of the neural networks with the Bayes optimal classifier.

### Structure of the Project

1. **Bayes Classifier** - The Bayes optimal classifier is implemented for synthetic data generated from normal distributions.
2. **Neural Network Training** - Fully connected neural networks are trained using different loss functions and compared against the Bayes classifier.
3. **Image Classification Task** - Neural networks are trained on MNIST-like data for digit classification.

## Files in this Repository

- `Data.py`: Contains data generation functions for creating synthetic datasets used in the Bayes classifier and neural network training.
- `Network.py`: Contains classes and functions for building and training neural networks, including fully connected layers, activation layers, and loss functions.
- `main.py`: The main script for implementing and comparing the Bayes classifier with neural networks using different methods.
- `main2.py`: A script for training neural networks on image data (e.g., MNIST).

## Usage Instructions

1. **Setup**
   - Ensure Python 3 is installed along with required libraries (NumPy, Matplotlib, Keras).
   - Load necessary libraries by running:
     ```bash
     pip install numpy matplotlib keras
     ```
   - Place the data files and scripts in the same directory.

2. **Running the Scripts**
   - Execute `main.py` to run the Bayes classification and neural network training with generated data.
   - Execute `main2.py` for image-based neural network classification.

3. **Visualization**
   - Plots showing the error rates and comparisons between different methods will be displayed using Matplotlib.

## Dependencies

- Python 3.x
- NumPy
- Matplotlib
- Keras (used for loading MNIST data)

## Project Details

### Problem 1: Bayes Classifier and Neural Networks
- **Bayes Optimal Test**: Classification using a Bayes optimal decision rule for synthetic data from normal distributions.
- **Neural Networks**: Fully connected networks with ReLU and sigmoid activations, trained using cross-entropy and exponential methods.

### Problem 2: Image Classification
- **Data**: MNIST-like image data of handwritten digits.
- **Neural Networks**: Fully connected networks with ReLU activations, trained using cross-entropy, exponential, and hinge loss functions.

## License

This project is for educational purposes. Licensing details can be added here if applicab
