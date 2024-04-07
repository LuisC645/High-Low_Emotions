import numpy as np
import matplotlib.pyplot as plt
from statistics import mode
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Importacion librerias SVM
from sklearn.svm import SVC

# Importacion librerias knn
from sklearn.neighbors import KNeighborsClassifier

# Importacion librerias Matriz de cofusión
from sklearn.metrics import confusion_matrix

# Importacion librerias para GridSearch
from sklearn.model_selection import GridSearchCV

# Importacion librerias para KFold
from sklearn.model_selection import KFold

# Importacion librerias ROC
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Importacion Librerias sns
import seaborn as sns

# Main ---------------------------------------------------------------

# Variables
# param_grid = {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 1000, 10000, 100000],
#              'gamma': [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 50, 100, 1000]}
param_grid = {'C': [0.1, 0.5, 1, 5, 10],
              'gamma': [1e-2, 0.1, 1, 10]}

param_grid_knn = {'n_neighbors': [1, 3, 5, 7, 9, 13, 15, 17, 21]}

# Arrays KNN
accuracy_media_knn = []
sensitivity_knn = []
specificity_knn = []
Y_probabilities_knn = []

# Arrays SVM
accuracy_media = []
sensitivity = []
specificity = []
Y_probabilities = []
decision_scores = []
true_labels = []
moda_c = []
moda_gamma = []
moda_k = []

# leer datos y guardarlos
data_high = np.loadtxt('sources/high_emotions.txt')
data_low = np.loadtxt('sources/low_emotions.txt')

# Unir matrices
data = np.vstack((data_high, data_low))

# Estandarizar los datos, Normalizacion de caracteristicas
data_normalized = StandardScaler().fit_transform(data)

# Reducción de dimensionalidad
pca = PCA(n_components=2)
# ext = variable caracteristicas reducidas
X = pca.fit_transform(data_normalized)

Y = np.zeros((len(data),))
Y[0:len(data_high)-1] = 1

# Dividir datos
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Realizar cross-validación de los datos
for train_index, test_index in kf.split(X):

    # Conjuntos de Train y test
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # Normalizar los datos Train, Test
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    # Clasificadores
    svm = SVC(kernel='rbf', probability=True)
    knn = KNeighborsClassifier()

    # Grid-Search
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
    grid_search_knn = GridSearchCV(
        knn, param_grid_knn, cv=5, scoring='accuracy')

    # Entrenar clasificador
    grid_search. fit(X_train_normalized, Y_train)
    grid_search_knn.fit(X_train_normalized, Y_train)

    # Obtener Mejor modelo
    best_param = grid_search.best_estimator_
    best_param_knn = grid_search_knn.best_estimator_

    # Predicciones
    Y_pred = best_param.predict(X_test_normalized)
    params = grid_search.best_params_
    moda_gamma.append(params['gamma'])
    moda_c.append(params['C'])

    Y_pred_knn = best_param_knn.predict(X_test_normalized)
    params_knn = grid_search_knn.best_params_
    moda_k.append(params_knn['n_neighbors'])

    # Curva ROC
    Y_prob = best_param.predict_proba(X_test_normalized)[:, 1]
    decision_score = best_param.decision_function(X_test_normalized)

    Y_prob_knn = best_param.predict_proba(X_test_normalized)[:, 1]

    # Agregar probabilidades de clase
    Y_probabilities.extend(Y_prob)
    decision_scores.extend(decision_score)
    true_labels.extend(Y_test)

    Y_probabilities_knn.extend(Y_prob_knn)

    # Matriz de confusión
    confusion = confusion_matrix(Y_test, Y_pred)
    confusion_knn = confusion_matrix(Y_test, Y_pred_knn)

    accuracy = (confusion[0, 0] + confusion[1, 1]) / confusion.sum()
    accuracy_media.append(accuracy)

    sens = confusion[1, 1] / \
        (confusion[1, 0] + confusion[1, 1])
    sensitivity.append(sens)

    spec = confusion[0, 0] / \
        (confusion[0, 0] + confusion[0, 1])
    specificity.append(spec)

    accuracy_knn = (confusion_knn[0, 0] +
                    confusion_knn[1, 1]) / confusion_knn.sum()
    accuracy_media_knn.append(accuracy_knn)

    sens_knn = confusion_knn[1, 1] / \
        (confusion_knn[1, 0] + confusion_knn[1, 1])
    sensitivity_knn.append(sens_knn)

    spec_knn = confusion_knn[0, 0] / \
        (confusion_knn[0, 0] + confusion_knn[0, 1])
    specificity_knn.append(spec_knn)

average_accuracy = np.mean(accuracy_media)
average_sens = np.mean(sensitivity)
average_spec = np.mean(specificity)

average_accuracy_knn = np.mean(accuracy_media_knn)
average_sens_knn = np.mean(sensitivity_knn)
average_spec_knn = np.mean(specificity_knn)

# Curva ROC predict.prob
fprP, tprP, thresholdsP = roc_curve(true_labels, Y_probabilities)
fprP_k, tprP_k, thresholdsP_k = roc_curve(true_labels, Y_probabilities_knn)

# curva ROC decision_function
fprD, tprD, thresholdsD = roc_curve(true_labels, decision_scores)

# Area bajo la curva ROC
roc_aucP = auc(fprP, tprP)
roc_aucD = auc(fprD, tprD)
roc_aucP_k = auc(fprP_k, tprP_k)

print(f'Accuracy SVM: {average_accuracy *
      100: .2f}, Desviation: {np.std(accuracy_media)*100: .2f}')
print(f'Sensitivity SVM: {average_sens *
      100: .2f}, Desviation: {np.std(sensitivity)*100: .2f}')
print(f'Specificity SVM: {average_spec *
      100: .2f}, Desviation: {np.std(specificity)*100: .2f}')
print(f'AUC predict.proba SVM: {roc_aucP: .3f}')
print(f'AUC decision_function SVM: {roc_aucD: .3f}')
print('')
print(f'Accuracy KNN: {average_accuracy_knn *
      100: .2f}, Desviation: {np.std(accuracy_media_knn)*100: .2f}')
print(f'Sensitivity KNN: {average_sens *
      100: .2f}, Desviation: {np.std(sensitivity_knn)*100: .2f}')
print(f'Specificity KNN: {average_spec_knn *
      100: .2f}, Desviation: {np.std(specificity_knn)*100: .2f}')
print(f'AUC predict.proba KNN: {roc_aucP_k: .3f}')
print('')
print(f'Moda parámetro K: {mode(moda_k)}')
print(f'Moda parámetro C: {mode(moda_c)}')
print(f'Moda parámetro Gamma: {mode(moda_gamma)}')


Y_probabilities = np.array(Y_probabilities)
Y_probabilities_knn = np.array(Y_probabilities_knn)
true_labels = np.array(true_labels)
decision_scores = np.array(decision_scores)

# Plotar Curvas ROC
plt.figure(figsize=(7, 7))
# plt.subplot(2, 2, 1)
plt.plot(fprD, tprD, color='red', linewidth=2,
         linestyle='-', label='AUC SVM desicion_function')
plt.xlabel('Specificity')
plt.ylabel('Sensitivity')

# plt.subplot(2, 2, 2)
plt.plot(fprP, tprP, color='blue', linewidth=3,
         linestyle='-', label='AUC SVM predict.proba')

# plt.subplot(2, 2, 3)
plt.plot(fprP_k, tprP_k, color='cyan', linewidth=1,
         linestyle='-', label='AUC KNN')
plt.title(f'Curvas ROC')
plt.legend(loc='lower right')

plt.figure()
# plt.subplot(3, 1, 1)
sns.kdeplot(decision_scores[true_labels == 0],
            color='blue', label='Negative Class')
sns.kdeplot(decision_scores[true_labels == 1],
            color='orange', label='Positive Class')
plt.xlabel('Decision Scores')
plt.ylabel('Frecuency')
plt.title('SVM decision_function')

# plt.subplot(3, 1, 2)
plt.figure()
sns.kdeplot(Y_probabilities[true_labels == 0],
            color='blue', label='Negative Class')
sns.kdeplot(Y_probabilities[true_labels == 1],
            color='orange', label='Positive Class')
plt.xlabel('Predicted Probabilities')
plt.ylabel('Frecuency')
plt.title('SVM predict.proba')

# plt.subplot(3, 1, 3)
plt.figure()
sns.kdeplot(Y_probabilities_knn[true_labels == 0],
            color='blue', label='Negative Class')
sns.kdeplot(Y_probabilities_knn[true_labels == 1],
            color='orange', label='Positive Class')
plt.xlabel('Predicted Probabilities')
plt.ylabel('Frecuency')
plt.title('KNN predict.proba')

plt.tight_layout()
plt.show()
