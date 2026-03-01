import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord, based at frequency f, with sampling frequency Fs.
    '''
    N = int(Fs / 2)
    n = np.arange(N)

    f1 = f
    f2 = f * np.power(2, 4/12)
    f3 = f * np.power(2, 7/12)

    x = (np.cos(2*np.pi*f1*n/Fs)
       + np.sin(2*np.pi*f2*n/Fs)
       + np.cos(2*np.pi*f3*n/Fs))

    return x


def dft_matrix(N):
    '''
    Create a DFT transform matrix, W, of size N.
    '''
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(-1j * 2 * np.pi * k * n / N)
    return W


def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.
    '''
    X = np.abs(np.fft.fft(x))
    N = len(X)

    freqs = np.fft.fftfreq(N, d=1/Fs)

    mask = freqs > 0
    freqs = freqs[mask]
    X = X[mask]

    idx = np.argsort(X)[-3:]
    freqs_found = np.sort(freqs[idx])

    return freqs_found[0], freqs_found[1], freqs_found[2]
