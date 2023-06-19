import math
import random
import argparse
import numpy as np
from scipy.signal import chirp
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift

parser = argparse.ArgumentParser()
parser.add_argument("--snr", default=-20, type=float, help='The Signal to Noise Ratio.')
parser.add_argument('--sf', type=int, default=8, help='The spreading factor.')
parser.add_argument('--bw', type=int, default=250000, help='The bandwidth.')
parser.add_argument('--fs', type=int, default=1000000, help='The sampling rate.')
opts = parser.parse_args()

opts.n_classes = 2 ** opts.sf # total number of symbols
opts.nsamp = int(opts.fs * opts.n_classes / opts.bw) # number of samples in a symbol

# generate standard upchirp and downchirp
t = np.linspace(0, opts.nsamp / opts.fs, opts.nsamp)
chirpI1 = chirp(t, f0=-opts.bw / 2, f1=opts.bw / 2, t1=2 ** opts.sf / opts.bw, method='linear', phi=90)
chirpQ1 = chirp(t, f0=-opts.bw / 2, f1=opts.bw / 2, t1=2 ** opts.sf / opts.bw, method='linear', phi=0)
upchirp = chirpI1 + 1j * chirpQ1

chirpI1 = chirp(t, f0=opts.bw / 2, f1=-opts.bw / 2, t1=2 ** opts.sf / opts.bw, method='linear', phi=90)
chirpQ1 = chirp(t, f0=opts.bw / 2, f1=-opts.bw / 2, t1=2 ** opts.sf / opts.bw, method='linear', phi=0)
downchirp = chirpI1 + 1j * chirpQ1


def gen_symbol(symbol_index, phase1, phase2):
    # symbol_index: the symbol of chirp to be generated
    # phase1 & phase2: initial phase of chirp (often random in COTS hardware)
    time_shift = round(symbol_index / opts.n_classes * opts.nsamp)
    data1 = upchirp[time_shift:] * (np.cos(phase1) + 1j * np.sin(phase1))
    data2 = upchirp[:time_shift] * (np.cos(phase2) + 1j * np.sin(phase2))
    return np.concatenate((data1, data2))

truth_idx = 64 # the real symbol of chirp

# generate raw chirp without noise
chirp_raw = gen_symbol(truth_idx, random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi))

# normalize
dataY = chirp_raw / np.mean(np.abs(chirp_raw))

# amplitude of additional noise
amp = math.pow(0.1, opts.snr / 20)
noise = amp / math.sqrt(2) * np.random.randn(opts.nsamp) + 1j * amp / math.sqrt(2) * np.random.randn(opts.nsamp)

# add noise
dataX = dataY + noise
dataX = dataX / np.mean(np.abs(dataX))

# multiply downchirp
chirp_data = dataX * downchirp

# perform fft
fft_raw = fft(chirp_data, len(chirp_data))

# up-down adding 
target_nfft = opts.n_classes
cut1 = np.array(fft_raw[:target_nfft])
cut2 = np.array(fft_raw[-target_nfft:])
fft_data = abs(cut1)+abs(cut2)

# plot
plt.plot(fft_data)
plt.plot(truth_idx, fft_data[truth_idx], 'o')
plt.savefig('001.jpg')
plt.waitforbuttonpress()
