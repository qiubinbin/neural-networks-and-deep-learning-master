import netural_work_3
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = netural_work_3.Network([784, 100, 10])
net.SGD(training_data, 30, 10, 0.5, test_data=test_data)