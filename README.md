# Modelo de Regresión Lineal - Datos Climáticos de la Segunda Guerra Mundial

## Descripción

Este proyecto utiliza un modelo de regresión lineal para predecir las temperaturas mínimas (MinTemp) a partir de las temperaturas máximas (MaxTemp) utilizando un conjunto de datos climáticos de la Segunda Guerra Mundial.

## Dataset

- **Fuente**: [Condiciones Climáticas en la Segunda Guerra Mundial](https://www.kaggle.com/datasets/smid80/weatherww2/data?select=Summary+of+Weather.csv)
- **Total de Instancias**: 119,040
- **Features**: 31
- **Datos Utilizados**: Las primeras 5,000 instancias con las características MinTemp y MaxTemp, divididas en un 70% para entrenamiento y un 30% para prueba.

## Ajuste de Hiperparámetros

- **Learning Rates Probados**: [0.1, 0.01, 0.001, 0.0001]
  - **Óptimo**: 0.001
- **Epochs Probados**: [100, 500, 1000, 5000, 10000, 20000]
  - **Óptimo**: 10,000

## Desempeño del Modelo

- **Pesos Finales**: w = 0.67, b = 2.55
- **MSE Final**: 3.96
- **Precisión de Predicción**: El modelo muestra una diferencia de 1 a 3 grados entre los valores predichos y reales, lo cual es esperado debido a la dispersión de los datos.

## Conclusiones y Futuro

El modelo tiene un buen desempeño, pero hay margen para mejoras:
1. **Más Datos**: Incluir más instancias podría mejorar el entrenamiento del modelo.
2. **Ajuste de Épocas**: Aumentar el número de epochs podría reducir aún más el MSE.
3. **Ajuste de la División de Datos**: Modificar la proporción de datos entre train y test set podría optimizar más los parámetros.
4. **Grados Polinomiales**: Agregar grados polinomiales podría capturar relaciones más complejas en los datos.
