# Proyecto: Predicción de Temperatura Promedio con Regresión Lineal

## Descripción

Este proyecto tiene como objetivo predecir la temperatura promedio utilizando la temperatura máxima durante la Segunda Guerra Mundial a través de un modelo de regresión lineal. Se entrenó el modelo con un conjunto de datos que contiene 119,040 instancias y 31 características, de las cuales solo se utilizaron `MeanTemp` y `MaxTemp` debido a su completitud (sin valores nulos).

## Estructura del Proyecto

### 1. Introducción
- **Objetivo**: Predecir la temperatura promedio a partir de la temperatura máxima.
- **Dataset**: Datos sobre temperaturas durante la Segunda Guerra Mundial.
- **Selección de Datos**: De las 119,040 instancias disponibles, se utilizaron las primeras 2,000 para entrenar el modelo. El dataset fue dividido en 70% para entrenamiento y 30% para prueba.

### 2. Hiperparámetros y Modificaciones
- **Learning Rate (α)**: Se probó con valores de `[0.1, 0.01, 0.001, 0.0001]`, determinando que `0.001` era el valor óptimo para evitar pesos altos y lograr un costo decreciente.
- **Epochs**: Se experimentó con `[100, 500, 1000, 5000, 10000, 15000, 20000]`. Se observó que después de 8,000 epochs, las mejoras en el costo eran mínimas, pero se decidió continuar hasta los 15,000 epochs para un ajuste más fino.

### 3. Valores Óptimos, Predicciones y Gráfico
- **Función de Costo (MSE)**: Comenzó con un valor de `463.04` y se redujo a `0.77` después de 15,000 epochs.
- **Valores Óptimos**: Pendiente `w` de `0.82` y bias `b` de `1.56`.
- **Predicciones**: El modelo muestra predicciones cercanas a los valores reales, con una diferencia promedio de `1 grado Celsius`.

![Gráfico de la función hipótesis](https://github.com/Axel3246/TC3006C-Portafolio-Implementacion-1/blob/main/FuncionHipotesis.png?raw=true)

### 4. Conclusión
- **Rendimiento**: El modelo logra predicciones razonablemente precisas, pero aún hay margen para mejoras.
- **Sugerencias para Mejoras**:
  1. **Incluir más datos**: Ampliar el conjunto de entrenamiento podría mejorar la generalización del modelo.
  2. **Ajustar el número de epochs**: Aumentar los epochs podría disminuir aún más el error.
  3. **Modificar el split de Train/Test**: Cambiar la proporción de datos entre entrenamiento y prueba para refinar los valores óptimos.
 
### 5. Referencias

Smith, S. (2017). Weather Conditions in World War II. Kaggle. 
https://www.kaggle.com/datasets/smid80/weatherww2/data?select=Summary+of+Weather.csv 
