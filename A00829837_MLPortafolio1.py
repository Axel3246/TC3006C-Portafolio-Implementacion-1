"""
A00829837_MLPortafolio1.py
Script de Python para el Portafolio 1

Creado by Axel Amós Hernández Cárdenas el 21/08/24.

Referencia para el dataset usado: 
Smith, S. (2017). Weather Conditions in World War II. Kaggle. 
https://www.kaggle.com/datasets/smid80/weatherww2/data?select=Summary+of+Weather.csv 
"""

# ======= IMPORT DE LIBRERIAS =======
"""
Librerias para gráficar el final y para la manipulación del dataset.
"""
import matplotlib.pyplot as plt
import pandas as pd

# ======= I. CREACIÓN DEL MODELO DE REGRESIÓN LINEAL =======

# === 1.1 Actualización de parámetros w y b ===
"""
Función: Actualiza los parámetros de w [pendiente] y b [intercept] para la función hipótesis de la regresión lineal.
"""
def update_params_w_and_b (X, y, w, b, alpha):

    # 1. Valores iniciales para el cálculo
    dl_dw = 0.0 
    dl_db = 0.0 
    N = len(X) 
    
    # 2. For loop para realizar el cálculo N veces de las derivadas parciales (gradient descent) de [w] y [b] 
    for i in range(N):
        dl_dw += -2 * X[i] * (y[i] - (w * X[i] + b)) # Calculará la sumatoria para la derivada parcial de w
        dl_db += -2 * (y[i] - (w * X[i] + b)) # Lo de arriba pero para la [b]
    
    # 3. Calcular los nuevos valores para [w] y [b]
    w = w - (1 / float(N)) * dl_dw * alpha #  Actualizar los parámetros por la sumatoria de gradient descent para [w]
    b = b - (1 / float(N)) * dl_db * alpha #  Lo de arriba pero para la [b]

    return w, b


"""
Función: Esta función calcula el Mean Squared Error (Función de Costo)
"""
def avg_loss(X, y, w, b):

    # 1. Valores iniciales
    N = len(X)
    total_error = 0.0 # Error Acumulado Inicial
   
   # 2. For para calcular el error total con la fórmula de MSE
    for i in range(N):
       total_error += (y[i] - (w * X[i] + b)) ** 2

    return total_error / float (N) 


"""
Función: Esta función realiza N (epoch) iteraciones para encontrar los valores óptimos
para los parámetros w [pendiente] y b [intercept], además de calcular la función de costo
"""
def train(X, y, w, b, alpha, epochs):
    print("=== Progreso del Entrenamiento ===")
    # 1. For loop para actualizar los valores de los parámetros [w] y [b] llamando a la función que realiza
    # # todo el calculo de gradient descent
    for e in range(epochs):
        # print(f' epoch {e}')
        w, b = update_params_w_and_b(X, y, w, b, alpha)
        if e % 100 == 0:
            avg_loss_train = avg_loss(X, y, w, b)
            print(f'En el epoch {e}, el loss fue de {avg_loss_train:0.2f}, [w] es {w:0.2f} y [b] es {b:0.2f}')

    return w, b

    
"""
Función: Es la función hipótesis y = wx + b. Utilizando los valores obtenidos por
el modelo para los parámetros [w] y [b], calcularemos 'y' a partir de 'x'.
"""    
def predict(x, w, b):
    return w * x + b


# ======= II. LOADING DATASET =======

pathToFile = 'Portafolio/weather_summary.csv'

df = pd.read_csv(pathToFile)
print(df.head(), end='\n')
print(f'\nEl dataframe cuenta con un total de {len(df)} instancias y {len(df.columns)} features')
print(f'\nLos features del dataset son: \n {df.columns}')
print(f'\nEl data set tiene datos nulos en: \n{df.isnull().sum()}')

# ======= III. SETUP DE PARÁMETROS Y HIPERPARÁMETROS =======

w = 0.0  # Valor inicial de la pendiente
b = 0.0  # Valor inicial del intercept

alpha = 0.001 # Valor del learning rate
epochs = 10001 # Valor de los epochs

# ======= IV. SEPARACIÓN DE DATASET EN TRAIN (70%) Y TEST (30%) =======

df = df[:5000]    # Limitando el dataset
X = df["MaxTemp"] # Independiente
y = df["MinTemp"] # Dependiente

#print(f' X.head es \n: {X.head()}\n')
#print(f' y.head es \n: {y.head()}')

trainSize = int(len(df) * 0.7) # Obtención del numero de instancias necesarias para el 70% de split Train - Test
print(f'\nEl train set abarcará desde la instancia 0 hasta la {trainSize}')
print(f'\nEl test set abarcará desde la instancia {trainSize+1} hasta {len(X)}\n')

# Instancia 0 -> Instancia 3500 (Train Set)
# Instancia 3501 -> Instancia 5000 (Test Set)

X_train = X[:trainSize] 
x_test = X[trainSize:] 
y_train = y[:trainSize]
y_test = y[trainSize:]  

#print(f'\nX_train tiene {len(X_train)} instancias')
#print(f'\nx_test tiene {len(x_test)} instancias')
#print(f'\ny_train tiene {len(y_train)} instancias')
#print(f'\ny_test tiene {len(y_test)} instancias')

# ======= V. EMPEZAR EL ENTRENAMIENTO =======

w, b = train(X_train, y_train, w, b, alpha, epochs)
print(f'\nEl valor de w es {w} y el de b es {b}')

# ======= VI. PREDECIR CON EL TEST SET =======

y_pred = predict(x_test, w, b)

# ======= VII. COMPARACIÓN =======
dfComparacion = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})
print(dfComparacion.head())  # Mostrar las primeras filas de la comparación

# ======= VIII. GRÁFICO FUNCIÓN HIPÓTESIS FINAL =======
plt.scatter(X_train, y_train, color='blue', label='Train Set')
plt.scatter(x_test, y_test, color='purple', label='Test Set')
plt.plot(x_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.show()


"""
======= REPORTE FINAL =======

I. Introducción

El dataset utilizado recopila información sobre las temperaturas durante la Segunda Guerra Mundial. El objetivo de este script es determinar si es posible predecir 
la temperatura mínima a partir de la temperatura máxima utilizando un modelo de regresión lineal.

Mediante el uso de funciones en Python y la librería pandas se identificó que el dataset cuenta con un total de 119,040 instancias y 31 features. Cabe destacar que, 
aunque varias de estas features presentan valores nulos, las dos que se utilizarán en este análisis (MinTemp y MaxTemp) no contienen datos faltantes en ninguna de las 
instancias. Por lo tanto, no será necesario realizar limpieza o imputación de datos para estas variables. Sin embargo, es necesario limitar la cantidad de datos para 
optimizar el uso de recursos computacionales, por lo que solo se utilizarán las primeras 5,000 instancias para entrenar el modelo, haciendo una división del 70% para 
Train y 30% para Test.

II. Hiperparámetros y Modificaciones

Se realizaron pruebas utilizando diferentes valores para el learning rate: [0.1, 0.01, 0.001, 0.0001]. Durante estas pruebas, se observó que, al utilizar un learning rate mayor, 
la función de costo comenzaba a arrojar valores NaN, indicando posibles pesos altos. Por otro lado, cuando se utilizó un learning rate demasiado bajo, el costo permanecía elevado, 
lo que sugiere la presencia de un alto bias o underfitting, donde el modelo no logra capturar los patrones de los datos, resultando en un modelo demasiado simple. Al final, 
se determinó que el valor óptimo para el learning rate sería de 0.001, ya que permitió que el cálculo fuera sin errores y que el costo se redujera progresivamente.

También se probaron diferentes valores para los epochs: [100, 500, 1000, 5000, 10000, 20000]. Se observó que después de aproximadamente 5,000 epochs, los ajustes en los valores 
calculados se volvieron mínimos. Aunque el error continuaba disminuyendo con un mayor número de epochs, las mejoras adicionales en la precisión eran cada vez menos significativas.

En conclusión, los valores óptimos determinados para los hiperparámetros fueron: un learning rate (alpha) de 0.001 y un número de epochs de 10,000.

III. Valores Óptimos, Predicciones y Gráfico

Inicialmente, la función de costo MSE arrojaba un valor elevado de aproximadamente 329. Sin embargo, a medida que el modelo se entrenaba a lo largo de los epochs, el valor de 
la función de costo se fue estabilizando, alcanzando 3.96 después de 10,000 epochs. Como se mencionó anteriormente, aunque el error podría continuar disminuyendo si se incrementa 
el número de epochs, las mejoras adicionales en el cálculo del costo serían pequeñas.

Teniendo en cuenta lo anterior, el entrenamiento del modelo de regresión lineal resultó en valores óptimos para la pendiente \( w = 0.67 \), para el bias \( b = 2.55 \), y un MSE de 3.96. 
Al graficar la función hipótesis junto con los datos, se observa que la línea de ajuste es adecuada. Las predicciones muestran que el modelo se aproxima a los valores reales de los datos, 
aunque no los calcula con exactitud, ya que en la mayoría de los casos se encuentra una diferencia de 1 a 3 grados. Esto, sin embargo, es de esperarse, dado que muchos datos se 
encuentran tanto alejados como cercanos a la función hipótesis en ambos conjuntos de datos (test y train).

IV. Conclusión

Se concluye que el modelo realiza predicciones muy aproximadas a los valores reales de MinTemp. Aunque el error, en comparación con el calculado en los primeros epochs, es bajo, 
el modelo aún tiene un gran margen de mejora para lograr predicciones más precisas, aunque no exactas. Finalmente, los siguientes puntos son buenos aspectos a considerar 
en caso de que se desee mejorar el modelo:

1. Incluir más datos: Dado que se redujo significativamente el número de instancias utilizadas para el entrenamiento de este modelo, incorporar más datos podría mejorar la 
capacidad del modelo para generalizar y hacer predicciones más precisas.
2. Ajustar el número de epochs: Aunque las mejoras en el MSE no fueron tan significativas con un mayor número de epochs, esto no implica que un aumento en los epochs no pueda 
lograr un MSE mucho más bajo que 3.96.
3. Modificar el split de instancias del Train y Test Set: Ajustar la proporción de instancias asignadas a los conjuntos de entrenamiento y prueba podría permitir obtener valores
óptimos más precisos con los hiperparámetros actuales.
4. Agregar grados polinomiales: Aunque el gráfico final no sugiere una tendencia que requiera grados polinomiales, experimentar con esta opción podría mejorar la capacidad del
modelo para capturar relaciones no lineales en los datos.
"""