#Tarea_2 
Para la primera parte de la tarea se implemento la función de costos BCE en el código de network.py . 
El overflow comentado anteriormente desaparcio debido a que estaba implementando de manera incorrecta la 
propagación hacia atrás, ya que para la capa mas externa existe una cancelación que creí que se propagaba 
de igual manera en el resto de capas, el error provoco el overflow antes mencionado.
Se implemento de manera correcta la inicialización de pesos discutida en la clase, mejorando el 
rendimiento de la red. 
Finalmente se añadio el optimizador ADAM en lugar del SGD de manera que ahora la red parece aprender de una 
manera mucho mas rápida que con el antes mencionado SGD. 
