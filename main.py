# Se importan las librerías necesarias
# pydub: para cargar el archivo de audio
# numpy: para convertir el audio a un arreglo de numpy (NumericalPython)
# matplotlib: para graficar el audio

from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt

# Adquisición de datos
# Carga del audio
audio_wav = 'C:/Users/luisc/Escritorio/Tutorial_GITA/sources/voz1.wav'
audio_dc = AudioSegment.from_wav(audio_wav)
sample_orignal = np.array(audio_dc.get_array_of_samples())

# Frecuencia de muestreo
sample_rate = audio_dc.frame_rate

time_seconds = np.arange(0, len(sample_orignal)) / sample_rate


# Procesado
# Eliminar señal DC
# .mean calcula la media de la señal y se la resta a cada punto del audio
dc_offset = np.mean(audio_dc.get_array_of_samples())
audio = audio_dc - dc_offset

# Audio sin DC en formato numpy array
sample = np.array(audio.get_array_of_samples())

# Normalizarmos el audio
max_amp = max(abs(sample))  # Calculamos el valor absoluto máximo de la señal
sample_normalized = sample / max_amp  # Normalizamos la señal

# Gráfica del audio
plt.figure(figsize=(9, 4))  # Ancho, alto en pulgadas
plt.plot(time_seconds, sample_normalized)
plt.title('Voz procesada')
plt.xlabel('Segundos')
plt.ylabel('Amplitud')
plt.grid(True)
plt.show()

plt.figure(figsize=(9, 4))  # Ancho, alto en pulgadas
plt.plot(time_seconds, sample_orignal)
plt.title('Voz Original')
plt.xlabel('Segundos')
plt.ylabel('Amplitud')
plt.grid(True)
plt.show()
