# Respaldo de Avances

## Objetivo 1: Calcular el Laplaciano de $\mu_h$


La solución de elementos finitos $\mu_h$ se escribe como:

$$
\mu_h(x) = \sum_i \mu_i \, \phi_i(x)
$$

donde $\mu_i$ son los valores nodales y $\phi_i$ las funciones base.

Para calcular el laplaciano de la solución, se aplica:

$$
\Delta \mu_h(x) = \sum_i \mu_i \, \Delta \phi_i(x)
$$

Por linealidad, el operador se distribuye sobre la suma, de modo que el laplaciano de la solución es la combinación lineal de los laplacianos de las funciones base.

Si se utilizan funciones cuadráticas $P2$, las segundas derivadas son constantes dentro de cada elemento. No obstante, se debe considerar el mapeo afín entre el elemento de referencia y el dominio físico.

Sea $F$ la función afín que transforma el elemento de referencia $\hat{x}$ al dominio real $x$:

$$
x = F(\hat{x}) = A \hat{x} + b
$$

Si se desea calcular derivadas de las funciones base respecto al dominio físico, se tiene que:

$$
\phi(x) = \hat{\phi}(F^{-1}(x)) = \hat{\phi}(A^{-1}(x - b))
$$

Aplicando la regla de la cadena, la primera derivada respecto a la coordenada $x_i$ resulta:

$$
\frac{\partial \phi}{\partial x_i} 
= \sum_{a=1}^{2} \frac{\partial \hat{\phi}}{\partial \hat{x}_a} 
\frac{\partial \hat{x}_a}{\partial x_i}
$$

donde 
$ \frac{\partial \hat{x}_a}{\partial x_i} = (A^{-1})_{a i}$.

Por lo tanto,

$$
\frac{\partial \phi}{\partial x_i} 
= (A^{-1})_{a i} \frac{\partial \hat{\phi}}{\partial \hat{x}_a}
$$

y en forma matricial:

$$
\nabla_x \phi(x) = A^{-T} \, \nabla_{\hat{x}} \hat{\phi}(\hat{x})
$$


Para las segundas derivadas, se deriva nuevamente respecto a $x_j$:

$$
\frac{\partial^2 \phi}{\partial x_j \partial x_i}
= \frac{\partial}{\partial x_j}
\left[
(A^{-1})_{a i} \frac{\partial \hat{\phi}}{\partial \hat{x}_a}
\right]
$$

Dado que $A$ es constante dentro de un elemento afín, 
$ \frac{\partial A^{-1}}{\partial x_j} = 0$, 
entonces:

$$
\frac{\partial^2 \phi}{\partial x_j \partial x_i}
= (A^{-1})_{a i} (A^{-1})_{b j} 
\frac{\partial^2 \hat{\phi}}{\partial \hat{x}_b \partial \hat{x}_a}
$$


En notación matricial, esto se escribe como:

$$
\nabla_x^2 \phi(x) = A^{-T}  \hat{H}(\hat{\phi}) A^{-1} = A^{-T}A^{-T}  \hat{H}(\hat{\phi}) = G\hat{H}(\hat{\phi}) 
$$

donde $\hat{H}(\hat{\phi})$ es el hessiano de la función base en el dominio de referencia.

Finalmente, el laplaciano se obtiene como la traza de dicha matriz:

$$
\Delta_x \phi = \mathrm{tr}(\,G\,\hat{H}(\hat{\phi})\, )
$$

Entonces el laplaciano de la función de elementos finitos es:

$$
\Delta_x \mu_h = \mathrm{tr}(\,G\,\hat{H}(\hat{\phi})\, )
$$