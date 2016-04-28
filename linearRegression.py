import numpy as np
from scipy import linalg

class LinearRegression(object):
    def __init__(self, dataset):
        self.data = dataset['data']
        # Add column for bias term
        k = np.ones((len(self.data)))
        self.data = np.vstack((k, self.data))
        self.data = self.data.T
        self.target = dataset['target']

        self.weights_gd = np.random.randn(self.data.shape[1]) # Random initialization for gradient descent
        print('The random weights are', self.weights_gd)
        self.learning_rate = 0.001
        print(self.data.shape)

    @staticmethod
    def calculate_hypothesis(input, weights):
        return input.dot(weights.transpose())

    def calculate_weights_formula(self):
        # calculation of weight matrix using (data'data)^-1*target
        t_data = self.data.T
        print(t_data.shape)
        inv = linalg.inv(np.dot(t_data, self.data))
        target = self.target.reshape(self.target.shape[0],1)
        print(target.shape)
        self.weight = np.dot(inv, np.dot(t_data,target))
        print(self.weight)

    def predict(self, dataset):
        data = dataset['test']
        # Adding ones for the bias term
        k = np.ones((len(data)))
        processed_data = np.hstack((k, data))
        processed_data = processed_data.T

        # Predict for each item
        predictions = []
        for item in range(len(k)):
            pred = np.dot(self.weight, processed_data[item])
            predictions.append(pred)

        print(predictions)


def generate_dataset(length=100):
    dataset = {}
    dataset['data'] = np.random.randn(length)
    dataset['target'] = np.random.randn(length) + 0.5 # adding some noise
    return dataset

def generate_test_dataset(length=100):
    dataset = {}
    dataset['test'] = np.random.randn(length)
    return dataset

datasets = generate_dataset()
linear = LinearRegression(datasets)
#linear.calculate_weights_formula()
linear.calculate_weights_formula()
test_datasets = generate_test_dataset()
linear.predict(test_datasets)
