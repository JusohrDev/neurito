# Respuestas: experimentos de Neurito 3.0 (laberinto)

Los datos se obtuvieron con `experimentos/experimentos_laberinto.py`. El script corre el juego sin Pygame, con las mismas reglas: un máximo de 45 intentos por reto, y un choque contra la pared también cuenta como intento. Cada configuración se evaluó en 300 retos (100 por mapa), con la semilla 42 para la red. Los resultados completos están en `experimentos/resultados_laberinto.json`.

Datos: 2860 ejemplos de entrenamiento y 716 de prueba.

Significado de las columnas:
- **Comidas %**: retos completados.
- **Pasos extra**: pasos de más respecto a la ruta más corta, contando solo los retos completados.
- **Choques/reto**: choques promedio contra las paredes.

---

## 1. Neurito sin entrenar. ¿Por qué se atasca?

| Modo | Exactitud prueba | Comidas % | Choques/reto |
|---|---|---|---|
| Red pura | 27.7 % | 2.3 % | 37.9 |
| Asistido | 27.7 % | 85.7 % | 0 |

Con pesos aleatorios, la red elige una dirección casi al azar. Su exactitud ronda el 25 %, que es lo esperado al adivinar entre 4 opciones. El problema de fondo es que la red es **determinista**. Si la dirección elegida apunta a una pared, Neurito no se mueve. Entonces la entrada (mapa, posición, meta) sigue siendo la misma, y la red vuelve a elegir la misma dirección. Así se queda chocando contra la pared hasta agotar los 45 intentos (≈38 choques por reto). Cuando sí logra moverse, suele caer en un ciclo entre dos casillas (A→B→A), porque la red no sabe dónde ya estuvo. Las únicas comidas (2.3 %) salen de casos con la comida al lado y una dirección aleatoria que justo apuntaba hacia ella.

## 2. Entrenar 100, 200 y 400 épocas

| Épocas | Pérdida | Exactitud entreno | Exactitud prueba | Comidas % (pura) | Choques/reto (pura) | Comidas % (asistido) | Pasos extra (asistido) |
|---|---|---|---|---|---|---|---|
| 0 | — | 25.3 % | 27.7 % | 2.3 % | 37.9 | 85.7 % | 12.5 |
| 100 | 0.1352 | 94.1 % | 84.9 % | 48.7 % | 0.87 | 100 % | 1.97 |
| 200 | 0.0376 | 99.9 % | 90.1 % | 82.7 % | 0 | 100 % | 0.51 |
| 400 | 0.0107 | 100 % | 91.5 % | 85.7 % | 0 | 100 % | 0.35 |

- La **pérdida** baja siempre. De 200 a 400 épocas cae casi 4 veces, pero la exactitud de prueba apenas sube (90.1 → 91.5 %). La red ya memorizó el entrenamiento (100 %) y lo que gana de ahí en adelante es sobre todo seguridad en lo que ya sabía.
- La **exactitud de prueba** se estanca alrededor del 91 %. Hay una brecha de 8–9 puntos con el entrenamiento: eso es el límite de generalización a pares nuevos.
- **Movimientos:** cuando la red pura llega a la comida, lo hace casi por la ruta óptima (0.03 pasos extra). El problema son los retos que no termina: un solo error suele bastar para dejarla en un ciclo. Por eso, con 91.5 % de exactitud solo completa 85.7 % de los retos.

## 3. Red pura frente a navegación asistida. ¿Qué es del modelo y qué de las reglas?

| Épocas | Comidas % pura | Comidas % asistido | Pasos extra asistido |
|---|---|---|---|
| 0 | 2.3 % | 85.7 % | 12.5 |
| 400 | 85.7 % | 100 % | 0.35 |

- **Lo que pone el modelo:** el sentido de dirección, es decir, saber hacia dónde queda la comida. Se ve en la red entrenada sin reglas: completa 85.7 % de los retos y casi siempre por la ruta más corta.
- **Lo que ponen las reglas:** no chocar nunca (se descartan los movimientos contra paredes) y salir de los ciclos (se penalizan las casillas ya visitadas). Las reglas por sí solas ya resuelven mucho. Con la red **sin entrenar**, el modo asistido completa 85.7 % de los retos, porque se convierte en una exploración aleatoria que evita repetir casillas, y en un laberinto de 9×9 eso suele bastar. Pero lo hace con 12.5 pasos de más.
- **Conclusión:** el 100 % del modo asistido no mide lo que aprendió la red. Para medir el aprendizaje hay que mirar el modo **red pura**, o los **pasos extra** del modo asistido (12.5 sin entrenar → 0.35 entrenada). Las reglas garantizan que llegue; el modelo hace que llegue por el camino corto.

## 4. Cambiar las 96 neuronas ocultas por 16, 32 o 128 (200 épocas)

| Ocultas | Pérdida | Exactitud entreno | Exactitud prueba | Comidas % (pura) | Choques/reto | Pasos extra (asistido) |
|---|---|---|---|---|---|---|
| 16 | 0.0992 | 96.9 % | 88.7 % | 70.7 % | 0.58 | 1.21 |
| 32 | 0.0630 | 99.2 % | 87.4 % | 77.7 % | 0.15 | 0.83 |
| 96 | 0.0376 | 99.9 % | 90.1 % | 82.7 % | 0 | 0.51 |
| 128 | 0.0366 | 99.9 % | 89.2 % | 86.0 % | 0.44 | 0.57 |

- Con **16 neuronas**, la red no alcanza a memorizar todo (96.9 % en entrenamiento), y en el juego completa bastante menos (70.7 %).
- Con más neuronas, la pérdida baja más rápido y en el juego le va mejor. A partir de 96 la mejora es pequeña: 96 y 128 quedan casi empatadas y la exactitud de prueba no sube. Más capacidad ayuda a memorizar, pero no garantiza generalizar mejor.
- 96 neuronas es un buen equilibrio. Fue la única configuración con 0 choques.

## 5. Cambiar la tasa de aprendizaje 0.45. ¿Converge? (200 épocas)

| Tasa | Pérdida época 10 | Pérdida final | Exactitud prueba | Comidas % (pura) | ¿Converge? |
|---|---|---|---|---|---|
| 0.01 | 1.379 | 1.0367 | 66.5 % | 15.7 % | Muy lento, casi plano |
| 0.1 | 1.258 | 0.2841 | 82.7 % | 48.7 % | Sí, pero lento |
| **0.45** | 0.682 | 0.0376 | 90.1 % | 82.7 % | Sí (valor base) |
| 1.0 | 0.551 | 0.0079 | 93.2 % | 87.3 % | Sí, más rápido |
| 2.0 | 0.482 | 0.0026 | 93.4 % | 88.7 % | Sí |
| 3.0 | 0.630 | 0.0017 | 93.7 % | — | Sí |
| 5.0 | 1.446 | 1.1491 | 31.0 % | — | **No**: se estanca |
| 8.0 | 1.702 | 1.4350 | 23.6 % | — | **No** |
| 15.0 | 3.305 | 2.9338 | 24.4 % | — | **No**: la pérdida sube y salta |

Las filas 3.0, 5.0, 8.0 y 15.0 salen de una corrida aparte, hecha solo para ver dónde diverge la red, sin simular el juego.

- Con 0.45 la red **sí converge**, aunque no es el valor más rápido. Este problema tolera tasas altas porque las entradas son one-hot (dispersas): cada ejemplo solo mueve unas pocas filas de W1. Hasta 3.0, la red converge más rápido y generaliza un poco mejor.
- Con 0.01 el aprendizaje está prácticamente paralizado. Tras 200 épocas, la pérdida sigue por encima de 1.
- A partir de ~5.0, los pasos son tan grandes que la red se "rompe". Muchas neuronas ReLU quedan muertas (siempre en 0), la pérdida se queda alta o rebota, y la exactitud cae a nivel de azar.

## 6. Cambiar de mapa: pares inicio-meta de prueba (red de 400 épocas)

Solo se usaron combinaciones **(mapa, inicio, meta) que nunca aparecieron en el entrenamiento**:

| Mapa | Ejemplos de prueba | Exactitud prueba | Comidas % (pura, desde un par de prueba) |
|---|---|---|---|
| 1 | 210 | 92.4 % | 84.7 % |
| 2 | 251 | 90.0 % | 82.0 % |
| 3 | 255 | 92.2 % | 86.7 % |

- El rendimiento es parecido en los tres mapas. El mapa 2 resulta un poco más difícil: tiene más casillas libres, más bifurcaciones y caminos alternativos de igual longitud.
- La red generaliza a pares que no vio porque aprendió por separado cómo se relacionan la posición y la meta dentro de cada mapa. Además, en un recorrido desde un par de prueba, las casillas intermedias con esa misma meta sí suelen estar en el entrenamiento. Basta con acertar el primer paso para "entrar" en terreno conocido.
- Como dice la guía, esto **no** es generalizar a laberintos nuevos. La red no ve las paredes, solo un identificador del mapa. Con un cuarto mapa desconocido no sabría qué hacer.

## 7. Cambiar el experto: DFS, BFS y A* (400 épocas cada uno)

En el script implementé dos expertos nuevos, DFS y A*, y entrené una red nueva con las etiquetas de cada uno.

| Experto | Etiquetas óptimas | Experto solo llega | Exactitud prueba | Comidas % (pura) | Comidas % (asistido) | Pasos extra (asistido) |
|---|---|---|---|---|---|---|
| BFS | 100 % | 100 % | 91.5 % | 85.7 % | 100 % | 0.35 |
| **DFS** | 54.9 % | **16.6 %** | **94.8 %** | **13.7 %** | 59.3 % | 12.2 |
| A* (Manhattan) | 100 % | 100 % | 92.6 % | 88.7 % | 100 % | 0.45 |

- **DFS empeora mucho a Neurito.** DFS encuentra *un* camino, no el más corto. Y como se recalcula desde cada casilla, sus consejos no son coherentes entre sí: desde A dice "ve a B", y desde B puede decir "regresa a A". Incluso siguiendo al experto DFS al pie de la letra, sin red, solo llega a la comida en el 16.6 % de los casos. La red copia muy bien a un mal profesor, así que hereda sus ciclos: 13.7 % en modo puro.
- **Ejemplo claro de que la exactitud no equivale a éxito:** la red DFS tiene la **mayor exactitud de prueba (94.8 %)** y la **peor tasa de comidas (13.7 %)**. La exactitud mide qué tan bien imita al profesor, no si el profesor es bueno.
- **A* es equivalente a BFS** en calidad: ambos dan siempre rutas óptimas (100 % de etiquetas óptimas). Solo cambian en cómo desempatan entre caminos de igual longitud. La pequeña ventaja de A* (88.7 % frente a 85.7 %) cabe dentro del ruido de una sola semilla. La ventaja real de A* es que el *profesor* explora menos casillas para encontrar la ruta. Eso importa en mapas grandes, pero a Neurito no le cambia nada.
- **Conclusión:** en aprendizaje por imitación, el techo del alumno es el profesor. BFS y A* son buenos profesores para este problema; DFS no.

---

## Conclusión general

La exactitud de clasificación mide cuántas veces la red elige el mismo movimiento que el experto, casilla por casilla. Completar el laberinto exige acertar **todos** los pasos seguidos y no caer en ciclos. Por eso, una red con 91 % de exactitud completa solo ~86 % de los retos, y una red que imita a DFS con 95 % de exactitud casi nunca llega. Las reglas del modo asistido tapan estos errores, pero son conocimiento escrito a mano, no aprendido.
