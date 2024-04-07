import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Importacion librerias SVM
from sklearn.svm import SVC

# Importacion librerias para KFold
from sklearn.model_selection import KFold

# Importacion librerias para GridSearch
from sklearn.model_selection import GridSearchCV

# Importacion librerias Matriz de cofusión
from sklearn.metrics import confusion_matrix

# Main ---------------------------------------------------------------

# leer datos y guardarlos
data_high = np.loadtxt('sources/high_emotions.txt')
data_low = np.loadtxt('sources/low_emotions.txt')

# Unir matrices
data = np.vstack((data_high, data_low))

# Estandarizar los datos, Normalizacion de caracteristicas
data_normalized = StandardScaler().fit_transform(data)

accuracy_media = []

param_grid = {'C': [0.1, 0.5, 1, 5, 10],
              'gamma': [1e-2, 0.1, 1, 10]}

# Reducción de dimensionalidad
pca = PCA(n_components=2)
# ext = variable caracteristicas reducidas
X = pca.fit_transform(data_normalized)

Y = np.zeros((len(data),))
Y[0:len(data_high)-1] = 1

print(len(data_normalized))
temp = 0

# Dividir datos de train test a partir de crossvalidación
kf = KFold(n_splits=20, shuffle=True, random_state=42)

for train_index, test_index in kf.split(X):

    if temp == 0:
        print(f'Train: {len(train_index)/len(data_normalized)*100: .1f}%')
        print(f'Test: {len(test_index)/len(data_normalized)*100: .1f}%')
        print('')
    temp = 1

    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # Normalizar los datos Train, Test
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    # Clasificador
    svm = SVC(kernel='rbf')
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')

    # Entrenar clasificador
    grid_search. fit(X_train_normalized, Y_train)

    # Obtener Mejor modelo
    best_param = grid_search.best_estimator_

    # Predicciones
    Y_pred = best_param.predict(X_test_normalized)

    confusion = confusion_matrix(Y_test, Y_pred)

    accuracy = (confusion[0, 0] + confusion[1, 1]) / confusion.sum()
    accuracy_media.append(accuracy)

average_accuracy = np.mean(accuracy_media)
desviaton = np.std(accuracy_media)

print(f'Accuracy: {average_accuracy *
      100:.2f}, Desviation: {np.std(accuracy_media)*100:.2f}')
