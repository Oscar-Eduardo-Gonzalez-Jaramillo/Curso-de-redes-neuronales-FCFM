#Tarea_2 
Para la primera parte de la tarea se implemento la función de costos BCE en el código de network.py . 
El overflow comentado anteriormente desaparcio debido a que estaba implementando de manera incorrecta la 
propagación hacia atrás, ya que para la capa mas externa existe una cancelación que creí que se propagaba 
de igual manera en el resto de capas, el error provoco el overflow antes mencionado.
Ahora implementare la inicialización de los pesos