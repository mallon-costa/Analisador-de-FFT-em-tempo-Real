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
import time, sys, math
from collections import deque
import sounddevice as sd

from src.utils import *

class Stream_Reader:
    """

    Arguments:

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

        print("Dispositivos de áudio disponíveis:")
        device_dict = sd.query_devices()
        print(device_dict)

        try:
            sd.check_input_settings(device=device, channels=1, dtype=np.float32, extra_settings=None, samplerate=rate)
        except:
            print("As configurações de som de entrada para o dispositivo %s e a taxa de amostragem %s Hz não são suportadas, usando os padrões..." %(str(device), str(rate)))
            rate = None
            device = None

        self.rate = rate
        if rate is not None:
            sd.default.samplerate = rate

        self.device = device
        if device is not None:
            sd.default.device = device

        self.verbose = verbose
        self.data_buffer = None

        # This part is a bit hacky, need better solution for this:
        # Determine qual é a forma de buffer ideal transmitindo algum áudio de teste
        self.optimal_data_lengths = []
        with sd.InputStream(samplerate=self.rate,
                            blocksize=0,
                            device=self.device,
                            channels=1,
                            dtype=np.float32,
                            latency='low',
                            callback=self.test_stream_read):
            time.sleep(0.2)

        self.update_window_n_frames = max(self.optimal_data_lengths)
        del self.optimal_data_lengths

        #Alternativas:
        #self.update_window_n_frames = round_up_to_even(44100 / updates_per_second)

        self.stream = sd.InputStream(
                                    samplerate=self.rate,
                                    blocksize=self.update_window_n_frames,
                                    device=None,
                                    channels=1,
                                    dtype=np.float32,
                                    latency='low',
                                    extra_settings=None,
                                    callback=self.non_blocking_stream_read)

        self.rate = self.stream.samplerate
        self.device = self.stream.device

        self.updates_per_second = self.rate / self.update_window_n_frames
        self.info = ''
        self.data_capture_delays = deque(maxlen=20)
        self.new_data = False
        if self.verbose:
            self.data_capture_delays = deque(maxlen=20)
            self.num_data_captures = 0

        self.device_latency = device_dict[self.device]['default_low_input_latency']

        print("\n##################################################################################################")
        print("\nPadrão para usar o primeiro microfone de trabalho, Executando no microfone %s com propriedades:" %str(self.device))
        print(device_dict[self.device])
        print('Que tem uma latência de %.2f ms' %(1000*self.device_latency))
        print("\n##################################################################################################")
        print('Gravando áudio em %d Hz\nUsando janelas de dados (sem sobreposição) de %d amostras (atualizando em %.2ffps)'
            %(self.rate, self.update_window_n_frames, self.updates_per_second))

    def non_blocking_stream_read(self, indata, frames, time_info, status):
        if self.verbose:
            start = time.time()
            if status:
                print(status)

        if self.data_buffer is not None:
            self.data_buffer.append_data(indata[:,0])
            self.new_data = True

        if self.verbose:
            self.num_data_captures += 1
            self.data_capture_delays.append(time.time() - start)

        return

    def test_stream_read(self, indata, frames, time_info, status):
        '''
        Função fictícia para determinar qual tamanho de bloco o fluxo está usando
        '''
        self.optimal_data_lengths.append(len(indata[:,0]))
        return

    def stream_start(self, data_windows_to_buffer = None):
        self.data_windows_to_buffer = data_windows_to_buffer

        if data_windows_to_buffer is None:
            self.data_windows_to_buffer = int(self.updates_per_second / 2) #By default, buffer 0.5 second of audio
        else:
            self.data_windows_to_buffer = data_windows_to_buffer

        self.data_buffer = numpy_data_buffer(self.data_windows_to_buffer, self.update_window_n_frames)

        print("\n--🎙  -- Iniciando transmissão de áudio ao vivo...\n")
        self.stream.start()
        self.stream_start_time = time.time()

    def terminate(self):
        print("👋  Enviando comando de término de stream...")
        self.stream.stop()