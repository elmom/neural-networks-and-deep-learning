using MNIST_loader
using NN

training_data, validation_data, test_data = MNIST_loader.load_data_wrapper()

net = Network([784, 30, 10])

SGD(net, training_data, 2, 10, 3.0, test_data=test_data)
