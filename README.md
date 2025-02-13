# Ajuste de Formas Geométricas

Este proyecto tiene como objetivo implementar un programa en Python que realice ajustes geométricos y algebraicos a conjuntos de datos. El enfoque principal del proyecto es ajustar una forma geométrica a un conjunto de puntos en el plano.

## Descripción del problema

El problema consiste en ajustar una circunferencia a un conjunto de puntos obtenidos por un robot que mide puntos sobre la circunferencia. Se desea calcular el radio y la posición del centro de la circunferencia a partir de los datos medidos.

## Ajuste algebraico

El programa implementa una solución basada en el ajuste algebraico utilizando el método de los mínimos cuadrados. Dado un conjunto de N puntos $(x_i, y_i)$, el programa encuentra los parámetros $(x_0, y_0, r)$ que minimizan la suma $\sum^N_{i=0} (Ax_i + By_i + C − (x_i^2 + y_i^2 ))^2$

El programa proporciona los parámetros de la circunferencia ajustada.

## Ajuste geométrico

Además del ajuste algebraico, el programa también ofrece un ajuste geométrico utilizando el método de Newton para minimizar la distancia entre los puntos y la circunferencia ajustada.

Para esto, se implementa la función de distancia $\varepsilon_g (x0, y0, r)$ que representa la distancia cuadrática entre un punto $(x_i, y_i)$ y la circunferencia de parámetros $x_0, y_0, r$. Luego, se aplica el método de Newton para minimizar esta distancia y obtener una mejor aproximación de los parámetros $x0, y0, r$.

## Cálculo del gradiente y el hessiano

El programa incluye funciones para calcular el gradiente y el hessiano de una función f en un punto z utilizando diferencias forward.

## Método de Newton

El programa implementa el método de Newton para ajuste geométrico. Utiliza las funciones previamente mencionadas para calcular el gradiente y el hessiano, y el método de Cholesky para resolver el sistema resultante.

## Ejemplos y resultados

Se proporcionan diferentes conjuntos de datos generados artificialmente para probar el programa. Estos conjuntos incluyen:

- a) Datos sobre un círculo completo sin ruido.
- b) Datos sobre un círculo completo con ruido aleatorio.
- c) Datos sobre un arco de circunferencia de amplitud α, con ruido. Se investiga el comportamiento del ajuste para valores pequeños de α.

El programa se encarga de ajustar la circunferencia a cada conjunto de datos y graficarlos junto con la circunferencia obtenida a partir de ellos.

## Requisitos del sistema

El programa requiere tener instalado Python y las siguientes bibliotecas adicionales:
- NumPy
- Matplotlib

¡Disfrute del ajuste geométrico y algebraico de formas con este proyecto! Si tiene alguna pregunta o sugerencia, no dude en contactarme.

