# Containerized Q-Learning
Q-Learning es un algoritmo de aprendizaje por refuerzo que utiliza una matriz para guardar los valores-Q y así definir una política óptima. Para poder llegar a esa misión es que se entrena con numerosos episodios para que la matriz se acerque a los valores óptimos. Uno de los problemas de este algoritmo es que en problemas complejos se necesita una gran cantidad de episodios para poder entrenar y eso puede ser costoso por recursos computacionales o por tiempo. Utilizando un Kubernetes es que se plantea un sistema distribuido para ejecutar eficientemente episodios en distintos containers con la finalidad de entrenar de manera más adecuada a los recursos que se tiene sea localmente o en la nube.

# Implementeación

La arquitectura del deployment cuenta con 1 nodo maestro (master) y N nodos trabajadores (worker). El nodo maestro cuenta con el modelo de Q-Learning, incluyendo la matriz Q (Q-Table) y el entorno. Los nodos trabajdores cuentan una matriz Q auxiliar y una copia del entorno. Al inicializar el nodo maestro le pasará una copia del entorno a todos los trabajadores. Depués, al momento de entrenar el nodo maestro les dará a todos los trabajadores una misma copia de la matriz Q actual y un número de episodios a trabajar. Terminados los episodios cada trabajdor mandará su versión terminada de la matriz. Cuando el maestro tiene todas las matrices se calculará una nueva matriz media que será la nueva matriz principal. Este proceso se repetirá hasta que se cumplan todos los episodios pedidos al maestro.

# Uso
Para facilitar el uso es que se tiene un makefile con los comandos.

0. Tener docker y kubernetes en la máquina y corriendo.
1. Construir las imágenes (make images) en el caso no estén construidas.
2. Especificar los parámetros. Dentro de roles/configMap.yaml se encuentran variables de ejecución del algoritmo. Dentro de worker/worker.yaml se puede especificar el número de trabajadores necesarios en "replicas"
3. Construir los nodos (make build)
4. Dejar que ejecute
5. Al final de la ejecución exportar el modelo (make get-model id='id del master-node')