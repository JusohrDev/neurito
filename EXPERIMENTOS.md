# Neurito – Experimentos y resultados (Clase 07: Redes neuronales)

**Autor:** Juan Esteban Herrera Rojas (JusohrDev)
**Curso:** Inteligencia Artificial – Universidad Santo Tomás, Seccional Tunja

Neurito es una red neuronal **3-6-3** (3 entradas → 6 neuronas ocultas con ReLU → 3 salidas con Softmax) entrenada con descenso de gradiente sobre 3000 ejemplos sintéticos. Las entradas son **hambre**, **energía** y **aburrimiento**, y las salidas son las acciones **EAT (comer)**, **SLEEP (dormir)** y **PLAY (jugar)**. La política que la red debe aprender es:

```
acción = argmax( hambre , 1 - energía , aburrimiento )
```

---

## 0. Línea base

Con la configuración original (6 neuronas ocultas, 1200 épocas, tasa de aprendizaje 0.1) la pérdida baja de **1.0742** a **0.1225** y la precisión en 1000 ejemplos de prueba es **99.40 %**.

---

## Experimentos

### Experimento 1 – Mantener dos necesidades constantes y variar la tercera. ¿Cuándo cambia la acción prevista?

Se fijaron dos necesidades y la tercera se movió de 0 a 1 en pasos de 0.001:

| Variable que cambia | Valores fijos | Cambio de acción | Punto de cambio (red) | Punto esperado (política) |
|---|---|---|---|---|
| Hambre | energía 0.5, aburrimiento 0.3 | SLEEP → EAT | **0.50** | 0.50 (cuando hambre supera el cansancio 1 − 0.5) |
| Energía | hambre 0.5, aburrimiento 0.3 | SLEEP → EAT | **0.50** | 0.50 (cuando el cansancio baja de la hambre) |
| Aburrimiento | hambre 0.5, energía 0.6 | EAT → PLAY | **0.51** | 0.50 (cuando aburrimiento supera la hambre) |

**Resultado:** la acción cambia justo cuando la necesidad que se mueve **supera a la mayor de las otras dos**. La red aprendió la frontera casi exacta (solo se corre 0.006 en el caso del aburrimiento). Además, el cambio no es brusco: cerca de la frontera las probabilidades pasan suavemente de una acción a otra (curva en forma de "S").

### Experimento 2 – Comparar prioridades casi idénticas. ¿Cómo cambian los resultados de Softmax?

| Caso | Hambre | Cansancio (1 − energía) | Aburrimiento | EAT | SLEEP | PLAY |
|---|---|---|---|---|---|---|
| Prioridad clara | 0.90 | 0.20 | 0.10 | **100.0 %** | 0.0 % | 0.0 % |
| Empate exacto | 0.50 | 0.50 | 0.50 | **35.2 %** | 33.4 % | 31.3 % |
| Casi empate, gana hambre | 0.52 | 0.50 | 0.50 | **44.3 %** | 28.9 % | 26.8 % |
| Casi empate, gana cansancio | 0.50 | 0.52 | 0.50 | 30.6 % | **42.3 %** | 27.1 % |
| Casi empate, gana aburrimiento | 0.50 | 0.50 | 0.52 | 30.7 % | 29.0 % | **40.3 %** |
| Casi empate alto | 0.80 | 0.79 | 0.78 | **42.3 %** | 32.7 % | 24.9 % |

**Resultado:** cuando una necesidad domina, Softmax da casi el 100 % a una acción (la red está "segura"). Cuando las prioridades son casi iguales, las probabilidades se **aplanan** hacia ~33 % cada una: la red sigue eligiendo la acción correcta, pero con poca confianza (40–44 %). Con una diferencia de apenas 0.02 la acción ganadora cambia, así que en esa zona un pequeño ruido en la entrada hace que Neurito "dude" o cambie de animación.

### Experimento 3 – Cambiar las neuronas ocultas de 6 a 3, 10 o 20

| Red | Pérdida final | Precisión de prueba |
|---|---|---|
| 3-3-3 | 0.1618 | 96.30 % |
| **3-6-3 (original)** | **0.1225** | **99.40 %** |
| 3-10-3 | 0.1262 | 99.30 % |
| 3-20-3 | 0.1220 | 98.80 % |

**Resultado:** con **3 neuronas** la red se queda corta (más pérdida y 3 puntos menos de precisión): tiene muy pocas "piezas" para dibujar las fronteras entre las tres acciones. De **6 en adelante** la mejora se estanca: 10 o 20 neuronas dan prácticamente la misma pérdida y precisión, con más parámetros y más cálculo. Para este problema tan simple, 6 neuronas son suficientes; agregar más no ayuda (la diferencia entre 98.8 % y 99.4 % es de solo 6 ejemplos de 1000 y depende de la inicialización aleatoria).

### Experimento 4 – Cambiar las épocas o la tasa de aprendizaje. ¿Mejora, se estanca o se vuelve inestable?

**Épocas (lr = 0.1):**

| Épocas | Pérdida final | Precisión |
|---|---|---|
| 100 | 0.8088 | 77.30 % |
| 300 | 0.3397 | 95.90 % |
| 1200 | 0.1225 | 99.40 % |
| 3000 | 0.0711 | 99.50 % |
| 6000 | 0.0484 | 99.60 % |

**Tasa de aprendizaje (1200 épocas):**

| lr | Pérdida final | Precisión | Comportamiento |
|---|---|---|---|
| 0.01 | 0.7376 | 82.90 % | Aprende muy lento (le faltan épocas) |
| 0.1 | 0.1225 | 99.40 % | Estable (original) |
| 0.5 | 0.0484 | 99.60 % | Estable y más rápido: el mejor resultado |
| 1.0 | 0.0447 | 98.70 % | **Inestable**: picos de pérdida que suben y bajan |
| 5.0 | 1.0979 | 33.80 % | **Se rompe**: la pérdida queda en ≈ ln(3) = 1.0986 y responde siempre EAT |
| 20.0 | 6.1621 | 33.80 % | **Diverge**: la pérdida oscila sin control |

**Resultado:**
- **Más épocas mejoran**, pero con rendimientos decrecientes: de 100 a 1200 épocas la precisión sube 22 puntos; de 1200 a 6000 solo 0.2 puntos. El entrenamiento **se estanca** porque ya casi no queda error que corregir.
- **Tasa muy baja** (0.01): los pasos son tan pequeños que en 1200 épocas no llega al mínimo.
- **Tasa moderada** (0.1 – 0.5): aprende bien; 0.5 llega en 1200 épocas a lo que 0.1 logra en 6000.
- **Tasa alta** (≥ 1.0): los pasos son tan grandes que "se pasan" del mínimo. Con 1.0 aparecen picos; con 5.0, 5 de las 6 neuronas ReLU quedan "muertas" (siempre devuelven 0) y la red responde EAT para todo (por eso 33.8 %, la proporción de EAT); con 20.0 mueren las 6 y la pérdida oscila sin control. Es como bajar una montaña dando saltos gigantes: en vez de llegar al valle, saltas de un lado al otro.

### Experimento 5 – ¿Qué datos y etiquetas requeriría una cuarta acción?

Para agregar, por ejemplo, la acción **BATHE (bañarse)** se necesita:

1. **Una nueva entrada (dato):** una necesidad que la justifique, por ejemplo **suciedad** (0 a 1). Cada ejemplo pasa de `[hambre, energía, aburrimiento]` a `[hambre, energía, aburrimiento, suciedad]`, así que `X` pasa de forma `(3000, 3)` a `(3000, 4)` y `W1` de `(3, 6)` a `(4, 6)`.
2. **Una nueva etiqueta:** la clase 3 = BATHE. La política sintética pasa a `argmax(hambre, 1 − energía, aburrimiento, suciedad)`, `W2` pasa de `(6, 3)` a `(6, 4)` y `b2` de `(1, 3)` a `(1, 4)`.
3. **Ejemplos suficientes y balanceados de la nueva clase:** en la prueba salieron 728 ejemplos de BATHE de 3000 (≈ 25 %, igual que las demás).
4. **En el juego:** sprites `banar_1.png`, `banar_2.png`, el texto `"bathe"` en `TEXT`, la tupla `actions` con 4 elementos, una barra de "Suciedad" y teclas para subirla/bajarla.

Se probó un prototipo 4-6-4 con estos cambios y alcanzó **97.80 %** de precisión, un poco menos que la red de 3 acciones porque ahora hay más fronteras que aprender con las mismas 6 neuronas ocultas.

---

## Explica tus resultados

### 1. ¿En qué se diferencian los pesos de los sesgos?

- Los **pesos** (`W1`, `W2`) **multiplican** a las entradas: dicen *cuánto importa* cada entrada para cada neurona y en qué dirección (positivo = empuja a favor, negativo = empuja en contra). Hay uno por cada conexión: 3×6 + 6×3 = **36 pesos**.
- Los **sesgos** (`b1`, `b2`) **se suman** al final y no dependen de ninguna entrada: mueven el umbral a partir del cual la neurona se activa. Hay uno por neurona: 6 + 3 = **9 sesgos**.
- Analogía: en una receta, los pesos son *cuánto* de cada ingrediente pones; el sesgo es la sal base que agregas siempre, haya lo que haya.
- En el código también se ve la diferencia: los pesos se inician aleatorios (`np.random.randn * 0.5`) para que cada neurona aprenda algo distinto, mientras que los sesgos empiezan en **cero** (`np.zeros`). Sin sesgo, una neurona solo podría trazar fronteras que pasen por el origen (0, 0, 0).

### 2. ¿Por qué el cansancio es `1 - energy`?

Porque la red decide comparando **necesidades**: cuanto más alta la necesidad, más urgente la acción. Hambre alta → comer; aburrimiento alto → jugar. Pero la energía funciona al revés: **energía alta significa que NO necesita dormir**. Al usar `1 - energía` se convierte la energía en "cansancio", que sí crece cuando aumenta la necesidad de dormir. Así las tres columnas quedan en la misma escala (0 a 1) y con el mismo sentido ("más alto = más urgente"), y `argmax` puede compararlas de forma justa. Por eso en el Experimento 1 Neurito duerme cuando la energía **baja** de 0.5 y no cuando sube.

### 3. ¿Qué función cumplen las funciones ReLU y Softmax?

- **ReLU** (`max(0, x)`), en la capa oculta: deja pasar los valores positivos y convierte los negativos en 0. Aporta la **no linealidad**: sin ella, apilar capas sería lo mismo que una sola multiplicación de matrices y la red no podría formar las fronteras "en esquina" que necesita `argmax`. Es como un interruptor: cada neurona se "enciende" solo en cierta zona de necesidades. (El Experimento 4 muestra su riesgo: con lr = 5 casi todas las neuronas quedan siempre en 0, "muertas", y la red deja de aprender.)
- **Softmax**, en la salida: convierte los 3 puntajes de la red en **probabilidades** positivas que suman 100 %. Así podemos leer la confianza de Neurito (Experimento 2: 100 % cuando está seguro, ~33 % cuando duda) y calcular la pérdida de entropía cruzada para entrenar. La acción final se elige con la probabilidad más alta.

### 4. ¿Por qué una alta precisión puede resultar engañosa?

El 99.40 % es real, pero no significa que Neurito "entienda" a una mascota:

1. **Solo mide qué tan bien copia una regla que nosotros inventamos.** Los datos y las etiquetas salen de `argmax(hambre, 1 − energía, aburrimiento)`; si esa política es mala o poco realista, la red la copia igual con 99 %. La precisión es "frente a la política sintética", no frente al comportamiento real.
2. **Los errores se concentran donde la decisión es difícil.** Separando los 1000 casos de prueba por qué tan reñidas están las dos necesidades más altas: con margen menor a 0.02 la precisión cae a **90 %**, y con margen mayor a 0.02 es **100 %**. La mayoría de los ejemplos son fáciles y "inflan" el promedio.
3. **El estado CALM no lo aprende la red.** Es un `if` en el bucle de dibujo. En el estado de calma (0.2, 0.9, 0.2) la red en realidad duda entre EAT 48.2 % y PLAY 44.7 %, pero la pantalla muestra "CALM".
4. **Datos de prueba iguales a los de entrenamiento.** Entrenamiento y prueba vienen de la misma distribución uniforme entre 0 y 1; con datos reales (sensores con ruido, necesidades que no son uniformes) la precisión podría bajar.
5. **Una métrica, un número.** La precisión no muestra la confianza ni el tipo de error; hay que mirar también la matriz de confusión y la pérdida (los 6 errores fueron 4 SLEEP→PLAY y 2 PLAY→EAT, todos en casos casi empatados).

Analogía: es como sacar 99 en un examen donde casi todas las preguntas son fáciles y fueron escritas por el mismo profesor que hizo la guía; no prueba que sepas resolver problemas nuevos.

---

## Conclusiones

- Neurito aprende casi a la perfección la política sintética (99.4 %) y sus fronteras de decisión coinciden con las esperadas (Exp. 1).
- Softmax refleja la confianza: alta cuando una necesidad domina, baja cuando hay casi empate (Exp. 2).
- 6 neuronas ocultas son suficientes; menos limita la red y más no aporta (Exp. 3).
- Más épocas ayudan hasta estancarse; la tasa de aprendizaje debe ser moderada: muy baja es lenta y muy alta es inestable o rompe la red (Exp. 4).
- Una cuarta acción requiere una nueva entrada, una nueva etiqueta, más ejemplos y ajustar las dimensiones de la red (Exp. 5).
- Una precisión alta debe leerse con cuidado: depende de cómo se generan los datos y oculta los casos difíciles.
