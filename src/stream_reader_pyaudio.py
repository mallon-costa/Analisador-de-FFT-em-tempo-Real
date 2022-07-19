'''
---------------------------------------------UNIFESSPA----------------------------------------------------
-------------------------------VISUALIZADOR DE FFT EM TEMPO REAL------------------------------------------

#Classe que lê continuamente os dados de uma fonte de som selecionada usando o PyAudio

Trabalho de Processamento Digital de Sinais         Engenharia da Computção 
Professora: Leslye Estefania Castro Eras

Discentes:
1. Alaim de Jesus Leão Costa - 201940601001
2. Klauber Araujo Sousa - 201940601010
3. Manoel Malon Costa de Moura - 201940601025

'''

import numpy as np
import pyaudio
import time, sys, math
from collections import deque

from src.utils import *

class Stream_Reader:
    """
    Argumentos:

        device: int or None:    Selecione qual fluxo de áudio ler.
        rate: float or None:    Taxa de amostragem a ser usada. O padrão é algo suportado.
        updatesPerSecond: int:  Com que frequência gravar novos dados.
    """

    def __init__(self,
        device = None,
        rate = None,
        updates_per_second  = 1000,
        FFT_window_size = None,
        verbose = False):

        self.rate = rate
        self.verbose = verbose
        self.pa = pyaudio.PyAudio()

        #Variáveis temporarias!
        self.update_window_n_frames = 1024 #Necessário para testar o dispositivo
        self.data_buffer = None

        self.device = device
        if self.device is None:
            self.device = self.input_device()
        if self.rate is None:
            self.rate = self.valid_low_rate(self.device)

        self.update_window_n_frames = round_up_to_even(self.rate / updates_per_second)
        self.updates_per_second = self.rate / self.update_window_n_frames
        self.info = self.pa.get_device_info_by_index(self.device)
        self.data_capture_delays = deque(maxlen=20)
        self.new_data = False
        if self.verbose:
            self.data_capture_delays = deque(maxlen=20)
            self.num_data_captures = 0

        self.stream = self.pa.open(
            format = pyaudio.paInt16,
            channels = 1,
            rate = self.rate,
            input=True,
            frames_per_buffer = self.update_window_n_frames,
            stream_callback=self.non_blocking_stream_read)

        print("\n##################################################################################################")
        print("\nPadrão para usar o primeiro microfone de trabalho, executando em:")
        self.print_mic_info(self.device)
        print("\n##################################################################################################")
        print('Gravação de %s a %d Hz\nUsando janelas de dados (não sobrepostas) de %d amostras (atualizando a %.2ffps)'
            %(self.info["nome"],self.rate, self.update_window_n_frames, self.updates_per_second))

    def non_blocking_stream_read(self, in_data, frame_count, time_info, status):
        if self.verbose:
            start = time.time()

        if self.data_buffer is not None:
            self.data_buffer.append_data(np.frombuffer(in_data, dtype=np.int16))
            self.new_data = True

        if self.verbose:
            self.num_data_captures += 1
            self.data_capture_delays.append(time.time() - start)

        return in_data, pyaudio.paContinue

    def stream_start(self, data_windows_to_buffer = None):
        self.data_windows_to_buffer = data_windows_to_buffer

        if data_windows_to_buffer is None:
            self.data_windows_to_buffer = int(self.updates_per_second / 2) #Por padrão, buffer de 0,5 segundo de áudio
        else:
            self.data_windows_to_buffer = data_windows_to_buffer

        self.data_buffer = numpy_data_buffer(self.data_windows_to_buffer, self.update_window_n_frames)

        print("\n--🎙  -- Iniciando transmissão de áudio ao vivo...\n")
        self.stream.start_stream()
        self.stream_start_time = time.time()

    def terminate(self):
        print("👋  Enviando comando de término de stream...")
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    def valid_low_rate(self, device, test_rates = [44100, 22050]):
        """Defina a taxa para a faixa de áudio mais baixa suportada."""
        for testrate in test_rates:
            if self.test_device(device, rate=testrate):
                return testrate

        #If none of the test_rates worked, try the default rate:
        self.info = self.pa.get_device_info_by_index(device)
        default_rate = int(self.info["TaxaDeAmostraPadrao"])

        if self.test_device(device, rate=default_rate):
            return default_rate

        print("ALGO ESTÁ ERRADO! Não consigo descobrir uma boa taxa de amostragem para DEVICE =>", device)
        return default_rate

    def test_device(self, device, rate=None):
        """dado um ID de dispositivo e uma taxa, retorne True/False se for válido."""
        try:
            self.info = self.pa.get_device_info_by_index(device)
            if not self.info["CanaisDeEntradaMaximos"] > 0:
                return False

            if rate is None:
                rate = int(self.info["TaxaDeAmostraPadrao"])

            stream = self.pa.open(
                format = pyaudio.paInt16,
                channels = 1,
                input_device_index=device,
                frames_per_buffer=self.update_window_n_frames,
                rate = rate,
                input = True)
            stream.close()
            return True
        except Exception as e:
            #print(e)
            return False

    def input_device(self):
        """
        Veja quais dispositivos podem ser abertos para entrada de microfone.
         Devolva o primeiro dispositivo válido
        """
        mics=[]
        for device in range(self.pa.get_device_count()):
            if self.test_device(device):
                mics.append(device)

        if len(mics) == 0:
            print("Nenhum dispositivo de microfone em funcionamento encontrado!")
            sys.exit()

        print("Encontrado(s) %d dispositivo(s) de microfone funcionando:" % len(mics))
        for mic in mics:
            self.print_mic_info(mic)

        return mics[0]

    def print_mic_info(self, mic):
        mic_info = self.pa.get_device_info_by_index(mic)
        print('\nMIC %s:' %(str(mic)))
        for k, v in sorted(mic_info.items()):
            print("%s: %s" %(k, v))