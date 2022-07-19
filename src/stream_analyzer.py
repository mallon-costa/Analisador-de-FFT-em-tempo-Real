'''
---------------------------------------------UNIFESSPA----------------------------------------------------
-------------------------------VISUALIZADOR DE FFT EM TEMPO REAL------------------------------------------

#Classe de acesso a gravações contiuamente e processamento de dados de aúdio

Trabalho de Processamento Digital de Sinais         Engenharia da Computção 
Professora: Leslye Estefania Castro Eras

Discentes:
1. Alaim de Jesus Leão Costa - 201940601001
2. Klauber Araujo Sousa - 201940601010
3. Manoel Malon Costa de Moura - 201940601025

'''

import numpy as np
import time, math, scipy
from collections import deque
from scipy.signal import savgol_filter

from src.fft import getFFT
from src.utils import *

class Stream_Analyzer:
    """
    Argumentos:

        device: int or None:      Selecione qual dispositivo de áudio ler.
        rate: float or None:      Taxa de amostragem a ser usada. O padrão é algo suportado.
        FFT_window_size_ms: int:  Tamanho da janela de tempo (em ms) a ser usado para a transformação FFT
        updatesPerSecond: int:    Com que frequência gravar novos dados.

    """

    def __init__(self,
        device = None,
        rate   = None,
        tamanhoJanela_ms_FFT  = 50,
        atualizacaoPorSegundo  = 100,
        tamanhoSuavizacao_ms = 50,
        n_compartimentoFrequencia    = 51,
        visualize = True,
        verbose   = False,
        altura    = 450,
        proporcaoJanela = 24/9):

        self.n_frequency_bins = n_compartimentoFrequencia
        self.rate = rate
        self.verbose = verbose
        self.visualize = visualize
        self.height = altura
        self.window_ratio = proporcaoJanela

        try:
            from src.stream_reader_pyaudio import Stream_Reader
            self.stream_reader = Stream_Reader(
                device  = device,
                rate    = rate,
                updates_per_second  = atualizacaoPorSegundo,
                verbose = verbose)
        except:
            from src.stream_reader_sounddevice import Stream_Reader
            self.stream_reader = Stream_Reader(
                device  = device,
                rate    = rate,
                updates_per_second  = atualizacaoPorSegundo,
                verbose = verbose)

        self.rate = self.stream_reader.rate

        #Configurações personalizadas:
        self.rolling_stats_window_s    = 20     # A faixa de eixos dos recursos FFT se adaptará dinamicamente usando uma janela de N segundos
        self.equalizer_strength        = 0.20   # [0-1] --> gradualmente redimensiona todos os recursos FFT para ter a mesma média
        self.apply_frequency_smoothing = True   # Aplique um filtro de suavização de pós-processamento nas saídas FFT

        if self.apply_frequency_smoothing:
            self.filter_width = round_up_to_even(0.03*self.n_frequency_bins) - 1
        if self.visualize:
            from src.visualizer import Spectrum_Visualizer

        self.FFT_window_size = round_up_to_even(self.rate * tamanhoJanela_ms_FFT / 1000)
        self.FFT_window_size_ms = 1000 * self.FFT_window_size / self.rate
        self.fft  = np.ones(int(self.FFT_window_size/2), dtype=float)
        self.fftx = np.arange(int(self.FFT_window_size/2), dtype=float) * self.rate / self.FFT_window_size

        self.data_windows_to_buffer = math.ceil(self.FFT_window_size / self.stream_reader.update_window_n_frames)
        self.data_windows_to_buffer = max(1,self.data_windows_to_buffer)

        # Suavização temporal:
        # Atualmente o buffer atua nos FFT_features (que são computados apenas ocasionalmente, por exemplo, 30 fps)
        # Isso é ruim, pois a suavização depende da frequência com que o método .get_audio_features() é chamado...
        self.smoothing_length_ms = tamanhoSuavizacao_ms
        if self.smoothing_length_ms > 0:
            self.smoothing_kernel = get_smoothing_filter(self.FFT_window_size_ms, self.smoothing_length_ms, verbose=1)
            self.feature_buffer = numpy_data_buffer(len(self.smoothing_kernel), len(self.fft), dtype = np.float32, data_dimensions = 2)

        self.fftx_bin_indices = np.logspace(np.log2(len(self.fftx)), 0, len(self.fftx), endpoint=True, base=2, dtype=None) - 1
        self.fftx_bin_indices = np.round(((self.fftx_bin_indices - np.max(self.fftx_bin_indices))*-1) / (len(self.fftx) / self.n_frequency_bins),0).astype(int)
        self.fftx_bin_indices = np.minimum(np.arange(len(self.fftx_bin_indices)), self.fftx_bin_indices - np.min(self.fftx_bin_indices))

        self.frequency_bin_energies = np.zeros(self.n_frequency_bins)
        self.frequency_bin_centres  = np.zeros(self.n_frequency_bins)
        self.fftx_indices_per_bin   = []
        for bin_index in range(self.n_frequency_bins):
            bin_frequency_indices = np.where(self.fftx_bin_indices == bin_index)
            self.fftx_indices_per_bin.append(bin_frequency_indices)
            fftx_frequencies_this_bin = self.fftx[bin_frequency_indices]
            self.frequency_bin_centres[bin_index] = np.mean(fftx_frequencies_this_bin)

        #Parâmetros codificados:
        self.fft_fps = 30
        self.log_features = False   # Log de plotagem (recursos FFT) em vez de recursos FFT -> geralmente muito ruim
        self.delays = deque(maxlen=20)
        self.num_ffts = 0
        self.strongest_frequency = 0

        #Suponha que o som de entrada segue um espectro de ruído rosa:
        self.power_normalization_coefficients = np.logspace(np.log2(1), np.log2(np.log2(self.rate/2)), len(self.fftx), endpoint=True, base=2, dtype=None)
        self.rolling_stats_window_n = self.rolling_stats_window_s * self.fft_fps #Assumes ~30 FFT recursos por segundos
        self.rolling_bin_values = numpy_data_buffer(self.rolling_stats_window_n, self.n_frequency_bins, start_value = 25000)
        self.bin_mean_values = np.ones(self.n_frequency_bins)

        print("Usando FFT_window_size com tamanho de %d para FFT ---> window_size = %dms" %(self.FFT_window_size, self.FFT_window_size_ms))
        print("##################################################################################################")

        #Vamos começar:
        self.stream_reader.stream_start(self.data_windows_to_buffer)
        if self.visualize:
            self.visualizer = Spectrum_Visualizer(self)
            self.visualizer.start()

    def update_rolling_stats(self):
        self.rolling_bin_values.append_data(self.frequency_bin_energies)
        self.bin_mean_values  = np.mean(self.rolling_bin_values.get_buffer_data(), axis=0)
        self.bin_mean_values  = np.maximum((1-self.equalizer_strength)*np.mean(self.bin_mean_values), self.bin_mean_values)

    def update_features(self, n_bins = 3):

        latest_data_window = self.stream_reader.data_buffer.get_most_recent(self.FFT_window_size)
       
        self.fft = getFFT(latest_data_window, self.rate, self.FFT_window_size, log_scale = self.log_features)
        #Equalizar a queda do espectro de ruído rosa:
        self.fft = self.fft * self.power_normalization_coefficients
        self.num_ffts += 1
        self.fft_fps  = self.num_ffts / (time.time() - self.stream_reader.stream_start_time)

        if self.smoothing_length_ms > 0:
            self.feature_buffer.append_data(self.fft)
            buffered_features = self.feature_buffer.get_most_recent(len(self.smoothing_kernel))
            if len(buffered_features) == len(self.smoothing_kernel):
                buffered_features = self.smoothing_kernel * buffered_features
                self.fft = np.mean(buffered_features, axis=0)

        self.strongest_frequency = self.fftx[np.argmax(self.fft)]

        #ToDo: substitua este loop for por código numpy puro
        for bin_index in range(self.n_frequency_bins):
            self.frequency_bin_energies[bin_index] = np.mean(self.fft[self.fftx_indices_per_bin[bin_index]])

        '''
        Detecção de batida ToDo:
        https://www.parallelcube.com/2018/03/30/beat-detection-algorithm/
        https://github.com/shunfu/python-beat-detector
        https://pypi.org/project/vamp/
        '''
        return

    def get_audio_features(self):

        if self.stream_reader.new_data:  #Verifica se o stream_reader tem novos dados de áudio que precisamos processar
            if self.verbose:
                start = time.time()

            self.update_features()
            self.update_rolling_stats()
            self.stream_reader.new_data = False

            self.frequency_bin_energies = np.nan_to_num(self.frequency_bin_energies, copy=True)
            if self.apply_frequency_smoothing:
                if self.filter_width > 3:
                    self.frequency_bin_energies = savgol_filter(self.frequency_bin_energies, self.filter_width, 3)
            self.frequency_bin_energies[self.frequency_bin_energies < 0] = 0

            if self.verbose:
                self.delays.append(time.time() - start)
                avg_fft_delay = 1000.*np.mean(np.array(self.delays))
                avg_data_capture_delay = 1000.*np.mean(np.array(self.stream_reader.data_capture_delays))
                data_fps = self.stream_reader.num_data_captures / (time.time() - self.stream_reader.stream_start_time)
                print("\nAvg delay da fft: %.2fms  -- avg delay dos dados: %.2fms" %(avg_fft_delay, avg_data_capture_delay))
                print("Número de capturas de dados: %d (%.2ffps)-- cálculos de número fft: %d (%.2ffps)"
                    %(self.stream_reader.num_data_captures, data_fps, self.num_ffts, self.fft_fps))

            if self.visualize and self.visualizer._is_running:
                self.visualizer.update()

        return self.fftx, self.fft, self.frequency_bin_centres, self.frequency_bin_energies