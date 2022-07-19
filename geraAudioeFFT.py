
'''
---------------------------------------------UNIFESSPA----------------------------------------------------
-------------------------------VISUALIZADOR DE FFT EM TEMPO REAL------------------------------------------

#Classe para gerar arquivo de audio pela as amostrar e Cálculo da FFT

Trabalho de Processamento Digital de Sinais         Engenharia da Computção 
Professora: Leslye Estefania Castro Eras

Discentes:
1. Alaim de Jesus Leão Costa - 201940601001
2. Klauber Araujo Sousa - 201940601010
3. Manoel Malon Costa de Moura - 201940601025

'''

import soundfile as sf
import numpy as np
import wave
import matplotlib.pyplot as plt
from src.fft import getFFT

from numpy import loadtxt


#importar arquivo de texto para o vetor NumPy
data = loadtxt('dadosGerados.txt')

#Quantidade padrão de taxa de amostragem
rate = 44100

#Gerar o arquivo de audio com os dados coletados 
sf.write('dados.wav', data, rate)

sf.write('dadosAmplificados.wav', data*10, rate)

time = np.arange(0, len(data) * 1/rate, 1/rate)

#Plot do audio original
plt.figure(1)
plt.title('Sinal Original Capturado')
plt.plot(time, data)
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.show()

n = len(data)
tx = 200

freq = np.fft.fftfreq(n)
mascara = freq > 0

fft_calculo = np.fft.fft(data)
fft_abs = 2.0*np.abs(fft_calculo/n)

#Plot da FFT
plt.figure(2)
plt.title('FFT do audio')
plt.plot(freq[mascara], fft_abs[mascara])
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude')
plt.show()


