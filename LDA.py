import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

data_high = np.loadtxt('sources/high_emotions.txt')
data_low = np.loadtxt('sources/low_emotions.txt')

data = np.vstack((data_high, data_low))

# Matriz de etiquetas
Y = np.zeros((len(data),))
Y[0:len(data_high)-1] = 1

# Estandarizar los datos, Normalizacion de caracteristicas
data_normalized = StandardScaler().fit_transform(data)

# LDA
lda = LDA(n_components=1)
# ext = variable caracteristicas reducidas
ext_data = lda.fit_transform(data_normalized, Y)

# Plot
plt.plot(ext_data, color='blue', label='Low Emotions')
plt.plot(ext_data[:len(data_high-1), :],
         label='High Emotions', markersize=5, color='red')

plt.title('Proyección lineal considerando dos discriminadores lineales')
plt.xlabel('LDA 1')
plt.ylabel('LDA 2')
plt.legend()
plt.show()
