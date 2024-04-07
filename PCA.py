import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Importacion librerias KNN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Importacion librerias SVM
from sklearn.svm import SVC

# Importacion librerias Matriz de cofusión
from sklearn.metrics import confusion_matrix

# Importacion librerias para GridSearch
from sklearn.model_selection import GridSearchCV

# Main ---------------------------------------------------------------

# leer datos y guardarlos
data_high = np.loadtxt('sources/high_emotions.txt')
data_low = np.loadtxt('sources/low_emotions.txt')

# Unir matrices
data = np.vstack((data_high, data_low))

print(len(data_high))
print(len(data_low))
print(len(data_low.T))
print(len(data_high.T))

# Estandarizar los datos, Normalizacion de caracteristicas
data_normalized = StandardScaler().fit_transform(data)

# Reducción de dimensionalidad
pca = PCA(n_components=2)
# ext = variable caracteristicas reducidas
ext_data = pca.fit_transform(data_normalized)

Y = np.zeros((len(data),))
Y[0:len(data_high)-1] = 1

# Se dividen los datos 70% para entrenamiento y 30% para test
data_train, data_test, Y_train, Y_test = train_test_split(
    ext_data, Y, test_size=0.3, random_state=42)

# Normalización de los datos de entrenamiento y prueba

scaler = StandardScaler()
data_train_normalized = scaler.fit_transform(data_train)
data_test_normalized = scaler.transform(data_test)

# KNN ---------------------------------------------------------------
# Se crea un clasificador KNN
knn = KNeighborsClassifier()

# Diccionario de parametros
param_grid_knn = {'n_neighbors': [1, 3, 5, 7, 11, 13, 15]}

# Configurar el objeto GridSearchCV con cv=1 para evitar crossvalidacion
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='accuracy')

# Entrenar el clasificador
grid_search_knn.fit(data_train_normalized, Y_train)

# Obtener el mejor modelo encontrado
best_knn = grid_search_knn.best_estimator_

# Realizar predicciones en los datos de prueba usando el mejor modelo
Y_pred_knn = best_knn.predict(data_test_normalized)

# Calcular la precisión del modelo KNN
accuracy_knn = (Y_pred_knn == Y_test).mean()

best_param = grid_search_knn.best_params_
print('Mejor parámetro: ', best_param)

# SVM ---------------------------------------------------------------
# Se crea un clasificador SVM
svm_classifier = SVC()

# Definir parámetros
param_grid_svm = {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 1000, 10000],
                  'kernel': ['linear', 'rbf'],
                  'gamma': [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 50, 100, 1000]}

grid_search_svm = GridSearchCV(
    svm_classifier, param_grid_svm, cv=5, scoring='accuracy')

# Entrenar el clasificador SVM
grid_search_svm.fit(data_train_normalized, Y_train)

# Obtener el mejor modelo
best_svm = grid_search_svm.best_estimator_

# Realizar predicciones en los datos de prueba
Y_pred_svm = best_svm.predict(data_test_normalized)

# Calcular la precisión del modelo SVM
accuracy_svm = (Y_pred_svm == Y_test).mean()

best_params_svm = grid_search_svm.best_params_
print('Mejores parámetros: ', best_params_svm)

# Matriz de cofusión ------------------------------------------------
# Calcular la matriz de confusión
confusion_knn = confusion_matrix(Y_test, Y_pred_knn)
confusion_svm = confusion_matrix(Y_test, Y_pred_svm)

# Calcular la precisión utilizando la matriz de confusión
accuracy_cofusion_knn = (
    confusion_knn[0, 0] + confusion_knn[1, 1]) / confusion_knn.sum()
print(f'Precisión del modelo KNN: {accuracy_cofusion_knn * 100: .2f}%')

accuracy_cofusion_svm = (
    confusion_svm[0, 0] + confusion_svm[1, 1]) / confusion_svm.sum()
print(f'Precisión del modelo SVM: {accuracy_cofusion_svm * 100: .2f}%')

print()
print('Sensibilidad y Especificidad')

# Calcular la sensibilidad de KNN
sensibilidad_knn = confusion_knn[1, 1] / \
    (confusion_knn[1, 0] + confusion_knn[1, 1])
# Calcular la especificidad de KNN
especificidad_knn = confusion_knn[0, 0] / \
    (confusion_knn[0, 0] + confusion_knn[0, 1])

print(f'Sensibilidad del modelo KNN: {sensibilidad_knn * 100: .2f}%')
print(f'Especificidad del modelo KNN: {especificidad_knn * 100: .2f}%')

# Calcular la sensibilidad de SVM
sensibilidad_svm = confusion_svm[1, 1] / \
    (confusion_svm[1, 0] + confusion_svm[1, 1])
# Calcular la especificidad de SVM
especificidad_svm = confusion_svm[0, 0] / \
    (confusion_svm[0, 0] + confusion_svm[0, 1])

print(f'Sensibilidad del modelo SVM: {sensibilidad_svm * 100: .2f}%')
print(f'Especificidad del modelo SVM: {especificidad_svm * 100: .2f}%')


# Plotear ----------------------------------------------------------

# plt.scatter es para gráficos de dispersión
plt.scatter(ext_data[0:len(data_high)-1, 0], ext_data[0:len(data_high)-1, 1],
            color='Red', facecolor='none', marker='o', label='High emotions')
plt.scatter(ext_data[len(data_high):, 0], ext_data[len(data_high):, 1],
            color='Blue', facecolor='none', marker='o', label='Low emotions')

plt.title('PCA para base de datos de emociones')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()

# Crear un gráfico de dispersión para visualizar el resultado
plt.figure(figsize=(10, 6))

# # Separar los puntos de datos de prueba en dos grupos basados en las predicciones
# plt.scatter(data_test[Y_pred == 0][:, 0], data_test[Y_pred == 0]
#             [:, 1], c='blue', label='Low Emotions', marker='o')
# plt.scatter(data_test[Y_pred == 1][:, 0], data_test[Y_pred == 1]
#             [:, 1], c='red', label='High Emotions', marker='s')

# plt.xlabel('PCA 1')
# plt.ylabel('PCA 2')
# plt.title('Resultados del KNN después de PCA')
# plt.legend()
# plt.show()
