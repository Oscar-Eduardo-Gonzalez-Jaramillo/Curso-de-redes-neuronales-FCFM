"""
network.py
~~~~~~~~~~

El propósito de este código de Python es generar una red neuronal implementada 
desde cero usando únicamente librerías básicas como NumPy, con el fin 
de entender perfectamente el funcionamiento interno básico de las redes neuronales.
El código usa backpropagation implementada desde cero y como función de costos 
por defecto utiliza MSE (por sus siglas en inglés de Mean Square Error), asumiendo 
etiquetas en formato one-hot para clasificación. Permite de manera sencilla 
construir diferentes tipos de redes neuronales.

"""
# Llamamos las librerías necesarias 
import random
import numpy as np

"""
Definimos una clase llamada Network para poder utilizar de una manera organizada 
y bien definida nuestra red.
"""
class Network(object):

    def __init__(self, sizes):
        """
        Aquí el código espera una lista de valores los cuales determinarán la 
        cantidad de capas y también la cantidad de neuronas que tendrán cada una 
        de estas. 
        Por ello se esperan valores escalares mayores que cero.
        """
        # Define el número de capas de la red 
        self.num_layers = len(sizes)  
        # Define la cantidad de neuronas en cada capa
        self.sizes = sizes

        """
        Inicializamos los biases y los pesos como matrices con cada uno de sus elementos
        aleatorios en una distribución gaussiana.
        """
        # Define una lista de vectores donde cada vector representa los sesgos de cada capa 
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # Define una lista que contiene las matrices de todos los pesos de la red neuronal.
        self.weights = [np.random.randn(y, x)  
                        for x, y in zip(sizes[:-1], sizes[1:])]
    """
    Definimos la función de feedforward, la cual nos permitirá entrenar y posteriormente
    utilizar la red en el momento que necesitemos. Esta función espera un vector de entrada
    (por ejemplo, la imagen aplanada de un número escrito a mano) y devuelve las activaciones 
    de la capa de salida, que se interpretan para obtener la predicción (como el número reconocido).
    """
    def feedforward(self, a):
        # Utilizamos el ciclo for para que los valores recibidos sean procesados por cada capa
        for b, w in zip(self.biases, self.weights):
            # Utilizamos la función sigmoide como función de activación en todas las capas
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        En esta parte se entrena la red neuronal al definir un conjunto de datos de entrenamiento, 
        los datos de prueba, las épocas, la tasa de aprendizaje y el tamaño del mini batch. 
        Es decir, aquí se definen gran parte de los hiperparámetros que determinarán fuertemente 
        el desempeño de la red.
        Los valores esperados son dos listas de datos para el caso de training_data y test_data, y 
        para el resto son valores mayores a 0.
        Cabe destacar que no es necesario definir siempre los datos de prueba, pero guiarnos 
        únicamente por el desempeño de la red en los datos de entrenamiento podría llevarnos a
        un sobreajuste, por lo cual es altamente recomendado reservar una cierta cantidad de datos 
        para probar la red.
        """
        # Si existen los datos de prueba, aquí los convertimos en una lista para su uso más adelante 
        if test_data:
            test_data = list(test_data)
            # Contamos la cantidad de pruebas para así al final mostrar la precisión de la red
            n_test = len(test_data)
        # Definimos la lista de valores de entrenamiento; cabe destacar que estos valores son una tupla 
        # donde el primer valor representa a la entrada y el segundo a la salida esperada 
        training_data = list(training_data)
        n = len(training_data)
        # En esta sección se empaquetan las imágenes en mini batches de manera aleatoria para evitar 
        # que la red aprenda patrones dados por el orden de las imágenes y permitir una mejor aproximación 
        # de la función de costos "real".
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            # Aquí evaluamos la red en los datos de prueba y mostramos su precisión
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            # Para el caso en el que no tengamos datos de prueba, simplemente imprime el valor de la época 
            # completada
            else:
                print("Epoch {0} complete".format(j))

    
    def update_mini_batch(self, mini_batch, eta):
        """
        En esta sección utilizamos el mini batch y la tasa de aprendizaje 
        para actualizar los valores de los pesos y de los sesgos. 
        Para ello calculamos con ayuda de la backpropagation y del mini batch
        una aproximación del gradiente de la función de costos y usamos este valor
        para "dirigir" o "guiar" a los valores de los pesos y sesgos de manera que 
        minimicen el valor de la función de costos.
        """
        # Para lograr lo anteriormente descrito, "clonamos" las dimensiones de los 
        # biases y pesos para poder actualizarlos correctamente 
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Calculamos y almacenamos los valores de las derivadas parciales de cada 
        # peso y bias; en otras palabras, calculamos la "culpa" de cada peso y bias. 
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        En esta sección del código calculamos los valores del gradiente de la función 
        de costos dada por el mini batch. Aprovechamos una de las propiedades de la 
        derivada de nuestra función de activación sigmoide para poder propagar hacia atrás
        las derivadas parciales de la función de costos y así obtener el gradiente. 
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] 
        zs = [] 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        """
        Para cambiar a la función de activación solamente debemos redefinir la derivada
        a su derivada parcial con respecto a las salidas de acuerdo a la función que 
        queramos usar.
        Para este caso queremos usar la BCE.
        """
        delta = self.cost_derivative(activations[-1], y) #* sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Notamos que la variable l en el bucle a continuación se usa de manera diferente 
        # a la notación en algunos libros. Aquí, l=1 significa la última capa de neuronas, 
        # l=2 la penúltima, y así sucesivamente. Esto aprovecha los índices negativos en listas de Python.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            #Cambiamos la definición de delta para ajustarla a la nueva función de activación 
            delta = np.dot(self.weights[-l+1].transpose(), delta)* sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Esta función devuelve el número de entradas de prueba para las cuales la red 
        neuronal produce el resultado correcto. La salida de la red se asume como el 
        índice de la neurona en la capa final con la activación más alta (usando argmax).
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Devuelve el vector de derivadas parciales para las activaciones de salida, 
        basado en la función de costo MSE.
        """
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """Definimos la función sigmoide"""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivamos la función sigmoide anidandola"""
    return sigmoid(z)*(1-sigmoid(z))