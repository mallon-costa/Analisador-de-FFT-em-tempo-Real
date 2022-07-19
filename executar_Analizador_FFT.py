'''
---------------------------------------------UNIFESSPA----------------------------------------------------
-------------------------------VISUALIZADOR DE FFT EM TEMPO REAL------------------------------------------

#Classe principal de execução

Trabalho de Processamento Digital de Sinais         Engenharia da Computção 
Professora: Leslye Estefania Castro Eras

Discentes:
1. Alaim de Jesus Leão Costa - 201940601001
2. Klauber Araujo Sousa - 201940601010
3. Manoel Malon Costa de Moura - 201940601025

'''

import argparse
from traceback import print_tb #módulo em Python ajuda a criar um programa em um ambiente de linha de comando 
#de uma forma que parece não apenas fácil de codificar, mas também melhora a interação.
from src.stream_analyzer import Stream_Analyzer #importação da classe Stream_Analyzer
import time #importação da biblioteca Time para ajudar definir um tempo de processamento

import os

try:
    os.remove('dadosGerados.txt')
except OSError as e:
    print(f"Error:{ e.strerror}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=None, dest='device',
                        help='índice de dispositivo pyaudio (porta de audio)')
    parser.add_argument('--altura', type=int, default=450, dest='altura',
                        help='altura, em pixels, da janela do visualizador')
    parser.add_argument('--compartimentoFrequencia', type=int, default=400, dest='compartimentoFrequencia',
                        help='Os recursos FFT são agrupados em compartimentos')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--proporcaoJanela', default='24/9', dest='proporcaoJanela',
                        help='razão de flutuação da janela do visualizador. por exemplo. 24/9')
    parser.add_argument('--quadrosDormindo', dest='quadrosDormindo', action='store_true',
                        help='quando o processo dorme entre os fps para reduzir o uso da CPU (recomendado para baixas taxas de atualização)')
    return parser.parse_args()

def converter_ProporcaoDeJanela(proporcaoJanela):
    if '/' in proporcaoJanela:
        dividendo, divisor = proporcaoJanela.split('/')
        try:
            float_razao = float(dividendo) / float(divisor)
        except:
            raise ValueError('window_ratio deve estar no formato: float/float')
        return float_razao
    raise ValueError('window_ratio deve estar no formato: float/float')

def executar_AnalizadorDeFFT():
    args = parse_args()
    proporcaoJanela = converter_ProporcaoDeJanela(args.proporcaoJanela)

    ear = Stream_Analyzer(
                    device = args.device,        # Índice de dispositivo Pyaudio (portaudio), padrão para a primeira entrada de microfone
                    rate   = None,               # Taxa de amostragem de áudio, None usa as configurações de origem padrão
                    tamanhoJanela_ms_FFT  = 60,    # Tamanho da janela usado para a transformação FFT
                    atualizacaoPorSegundo  = 1000,  # Com que frequência ler o fluxo de áudio para novos dados
                    tamanhoSuavizacao_ms = 50,    # Aplique alguma suavização temporal para reduzir recursos ruidosos
                    n_compartimentoFrequencia = args.compartimentoFrequencia, # Os recursos FFT são agrupados em compartimentos
                    visualize = 1,               # Visualize os recursos FFT com PyGame
                    verbose   = args.verbose,    # Imprima estatísticas de execução (latência, fps, ...)
                    altura    = args.altura,     # Altura, em pixels, da janela do visualizador,
                    proporcaoJanela = proporcaoJanela  # Taxa de flutuação da janela do visualizador. por exemplo. 24/9
                    )

    fps = 60  #Com que frequência atualizar os quadros de FFT & tela
    ultima_Atualizacao = time.time()
    while True:
        if (time.time() - ultima_Atualizacao) > (1./fps):
            ultima_Atualizacao = time.time()
            raw_fftx, raw_fft, binned_fftx, binned_fft = ear.get_audio_features()
        elif args.quadrosDormindo:
            time.sleep(((1./fps)-(time.time()-ultima_Atualizacao)) * 0.99)

if __name__ == '__main__':
    executar_AnalizadorDeFFT()
