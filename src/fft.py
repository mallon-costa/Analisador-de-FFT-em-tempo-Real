'''
---------------------------------------------UNIFESSPA----------------------------------------------------
-------------------------------VISUALIZADOR DE FFT EM TEMPO REAL------------------------------------------

#Classe de Cálculo da FFT em tempo Real

Trabalho de Processamento Digital de Sinais         Engenharia da Computção 
Professora: Leslye Estefania Castro Eras

Discentes:
1. Alaim de Jesus Leão Costa - 201940601001
2. Klauber Araujo Sousa - 201940601010
3. Manoel Malon Costa de Moura - 201940601025

'''

import numpy as np

def getFFT(data, rate, chunk_size, log_scale=False):
    #abre um arquivo com o nome dadosGerados - se não tiver, é criado
    arquivo = open('dadosGerados.txt', 'a')

    #escreve cada valor no documento 
    for i in range(0, len(data)):
        arquivo.write(str(data[i])+" ")
    
    data = data * np.hamming(len(data))
    try:
        FFT = np.abs(np.fft.rfft(data)[1:])
    except:
        FFT = np.fft.fft(data)
        left, right = np.split(np.abs(FFT), 2)
        FFT = np.add(left, right[::-1])

    # fftx = np.fft.fftfreq(chunk_size, d=1.0/rate)
    # fftx = np.split(np.abs(fftx), 2)[0]

    if log_scale:
        try:
            FFT = np.multiply(20, np.log10(FFT))
        except Exception as e:
            print('Log(FFT) failed: %s' % str(e))
    
    return FFT


