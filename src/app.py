# # Linda Vasquez proyecto de árbol de decisión

# ## 1. Carga de datos

import pandas as pd

ruta_datos = "https://breathecode.herokuapp.com/asset/internal-link?id=930&path=diabetes.csv"

datos = pd.read_csv(ruta_datos)

print(datos.head())

# ## 2. EDA y división de datos

from sklearn.model_selection import train_test_split

print("Información del dataset:")
print(datos.info())
print("\nResumen estadístico:")
print(datos.describe())

caracteristicas = datos.drop(columns=["Outcome"])
objetivo = datos["Outcome"]

x_entrena, x_prueba, y_entrena, y_prueba = train_test_split(caracteristicas, objetivo, test_size=0.2, random_state=42)

# ## 3. Árbol de decisión

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree

# Creamos y entrenamos el modelo con criterio Gini
modelo_gini = DecisionTreeClassifier(criterion="gini", random_state=42)
modelo_gini.fit(x_entrena, y_entrena)

# ///// Criterio Entropía
modelo_entropia = DecisionTreeClassifier(criterion="entropy", random_state=42)
modelo_entropia.fit(x_entrena, y_entrena)

# Predecimos con ambos modelos
predicciones_gini = modelo_gini.predict(x_prueba)
predicciones_entropia = modelo_entropia.predict(x_prueba)

# Comparamos la precisión
precision_gini = accuracy_score(y_prueba, predicciones_gini)
precision_entropia = accuracy_score(y_prueba, predicciones_entropia)

print(f"Precisión del árbol con Gini: {precision_gini}")
print(f"Precisión del árbol con Entropía: {precision_entropia}")

plt.figure(figsize=(10,8))
tree.plot_tree(modelo_gini, max_depth=2, feature_names=list(caracteristicas.columns), class_names=["No Diabetes", "Diabetes"], filled=True)
plt.show()

# ## 4.Optimización del modelo

from sklearn.model_selection import GridSearchCV

parametros_a_probar = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

modelo_base = DecisionTreeClassifier(random_state=42)

#Búsqueda en cuadrícula (GridSearch)
busqueda_cuadricula = GridSearchCV(modelo_base, parametros_a_probar, cv=5, scoring='accuracy')

busqueda_cuadricula.fit(x_entrena, y_entrena)

# Se extrae el mejor modelo
mejor_modelo = busqueda_cuadricula.best_estimator_

# Prueba con los datos de prueba
predicciones_mejoradas = mejor_modelo.predict(x_prueba)
precision_mejorada = accuracy_score(y_prueba, predicciones_mejoradas)

print(f"Mejor combinación de hiperparámetros: {busqueda_cuadricula.best_params_}")
print(f"Nueva precisión optimizada: {precision_mejorada}")

# ## 5. Almacenamiento

from pickle import dump

nombre_archivo = "arbol_decision_optimizado_42.sav"
dump(mejor_modelo, open(nombre_archivo, "wb"))

print(f"Modelo guardado como {nombre_archivo}")