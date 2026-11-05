1. realizando la prueba de neurito sin entrenamiento durante 3 minutos, se obtuvieron los siguientes saltos y objetos esquivados:
- 0 obstaculos esquivados.
- 28 choques.

2. al realizar la prueba de subir de 100 en 100 epocas en un total de 300, se obtuvieron lso siguientes datos.
- cuando estaba en 0 su probabilidad de salto era de 43.7%, la precisión del 44.9% y su comportamiento era tener choques constantes con los obstacilos y no contaba aun con la estadistica de perdida.

- para las 100 epocas ya obtenia una PERDIDA: 0.2601, PRECISIÓN: 91.2%, PROBABILIDAD DE SALTO: esta tiene un comportamiento bastante extraño y que varia durante el salto y la distancia a la que este del salto, donde aumenta mientras este mas cerca del obstaculo y al saltarlo llega a su pico maximo y al pasar el obstaculo o estrellarse baja por completo a 0. !!!ESTO PASA CON TODAS LAS DEMAS PRUEBAS!!!.

- en las 200 epocas neurito tiene una presicion mas alta por lo que la probabilidad de salto con la aproximacion del obstaculo suele ser mas alta, aqui estan los datos, PERDIDA: 0.2306, PRECISIÓN: 91.9%

- En las 300 epocas neurito los datos obtenidos fueron, PERDIDA: 0.2066, PRECISIÓN: 92.3%.

Con estos datos se puede hacer la comparacion que se pide.

donde comparanto los 4 datos de 0 a 300 se ve un aumento bastante de la precision del salto por lo cual genera mas confianza en cuanto a que tan probable es que se realice el salto.

con la perdida pasa lo mismo donde la perdida en vez de subir suele bajar.

su comportamiento de por si es aumentar mas la probabilidad de los saltos precisos.

3. 
- de pasar de 12 a 3, neurito de por si no se mueve, sin importar si se aumentan las epocas o no, aunque su precision aumenta y su perdida sigue bajando y donde llega a esquivar un objeto luego de muchas perdidad al rededor de unas 40.

- con 24 neurito empezaria a saltar repetidamente, cada que salta  probabilidad de salto aumenta parabaja y asi susesivamente su precision aumenta al 58,2% y su velocidad es muy baja.

4. al realizar la prueba de modificar la tasa de aprendizaje de 0.8 a 0.01 y a 1.2 para ver como convergía, se obtuvieron estos datos:

- con 0.01 la cosa va extremadamente lenta, la perdida se quedo estancada en 0.6449 con una curva casi plana y neurito no reaccionaba a tiempo, sacó 90.7% de precisión pero solo esquivo 2 obstáculos.

- con 0.80 que era la base, la cosa era mas equilibrada entre velocidad y estabilidad, logrando bajar la perdida a 0.2601 y manteniendo un control mas preciso.

- con 1.2 la perdida cayó rapidísimo a 0.2738 y esquivó 24 obstáculos con un 92.7% de precisión, pero la curva de perdida se puso rara con oscilaciones y ruido porque la red daba pasos muy grandes y rebotaba en el mínimo.

con estos datos se concluye que si le pones una tasa muy baja el aprendizaje se paraliza en la practica, y si se la pones muy alta la red se vuelve inestable aunque aprenda rápido.

5. PREGUNTA FINAL.

- la precisión da engañosamente alta porque si neurito esta lejos del obstáculo y no salta, el sistema le suma puntos como si lo estuviera haciendo bien. entonces acumula un 90% o 92% de precisión solo por quedarse sin hacer nada en el camino recto, pero basta con que se equivoque en un solo cuadro cuando tiene el obstáculo al frente para estrellarse.

- también pasa que la red se confía o no sabe el momento exacto donde reaccionar, como en la prueba de 0.01 donde sacó 90.7% de precisión pero solo esquivó 2 objetos porque no le daba el tiempo para saltar justo cuando tocaba.

en resumen, la precisión solo mide que tanto acierta en promedio a lo largo de toda la pista, pero al juego solo le importa si saltó a tiempo en el momento exacto.