import mnist_loader
import network
import pickle

training_data, validation_data , test_data = mnist_loader.load_data_wrapper()

training_data = list(training_data)
test_data = list(test_data)

net=network.Network([784,15,10])

net.SGD( training_data, 10, 5, 0.1, test_data=test_data) ## Epocas=15, Mini_batch_size=10, learning rate=0.1 

archivo = open("red_prueba2_pkl",'wb')
pickle.dump(net,archivo)
archivo.close()
exit()