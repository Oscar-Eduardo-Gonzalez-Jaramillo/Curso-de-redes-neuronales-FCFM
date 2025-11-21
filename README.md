## Tarea 6
 La idea original para esta tarea era primero encontrar una arquitectura que presentará los mejores resultados de entre todas, después con la arquitectura elegida utilizarla en una base de datos mas grande pero que mantenga una gran correlación con la base de datos del problema de la tarea con lo cuál el modelo entrenado en estos datos podría ser nuevamente ajustado en el conjunto de datos menor con ello se esperaba que el modelo generalizara de mejor manera en el conjunto de datos de la tarea. 
 
 Con esta idea en mente se comenzo la busqueda de arquitecturas obteniendo una arquitectura con un desempeño superior al resto en el conjunto de datos del concurso, posteriormente se elegió la base de datos de "plantnet" para el entrenamiento de mi modelo, la cuál es una base de datos que contiene 300K de imagenes de plantas etiquetadas por especies siendo un total de 1081 especies. Esta base de datos fue elegida para entrenar aquí al modelo y para después hacer transfer learning en el conjunto mas pequeño. 
 
 Lamentablemente el entrenamiento con una base de datos tan grande resulto ser un desfio que no logre superar debido a constantes crasheos de mi computadora y tiempos de entrenamiento absurdamente grandes. Por ello decidí al final reiniciar el experimento de optuna y emepzar de cero para unicamente realizar la busqueda de optuna de la mejor arquitectura en la base de datos de la tarea.
 
 El resultado de la busqueda de optuna así como el porcentaje de precisión final en los datos de test se pueden encontrar en el siguiente link de mlflow con el nombre de Tarea_6_FINAL, en la run llamada Fine_tunning:
 https://dagshub.com/Oscar-Eduardo-Gonzalez-Jaramillo/Curso-de-redes-neuronales-FCFM.mlflow/#/experiments/27?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D

 
 En el link anterior se pueden observar los multiples trials junto con el modelo final, cabe destacar que el modelo fue elegido por Optuna y re-entrenado desde cero lo que parece presentar ciertas incosistencias con los resultados en la precisión y la perdida en los datos  de entrenamiento y validación. Por otra parte en el repositorio se encuentra el código usado para la busqueda en optuna y tambien el resultado final de precisión en los datos de validación.

 

