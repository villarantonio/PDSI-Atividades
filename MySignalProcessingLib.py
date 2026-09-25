import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display

def SIGNALgenerate(N=100, signal='delta', params=None):
    """
    Gera sinais digitais básicos.
    Retorna:
        x: vetor com os valores do sinal
        n: vetor de amostras (índices de 0 a N-1)
    """
    n = np.arange(N)

    if signal == 'delta':
        x = np.zeros(N)
        center = N // 2
        x[center] = 1

    elif signal == 'step':
        x = np.concatenate((np.zeros(N // 2), np.ones(N - N // 2)))

    elif signal == 'exp':
        a = params[0] if params is not None else 0.85
        n_shifted = n - (N // 2)
        # Usa o degrau gerado internamente
        step_signal, _ = SIGNALgenerate(N, signal='step')
        x = (np.abs(a) ** n_shifted) * step_signal

    elif signal == 'osc':
        period = params[0] if params is not None else 40
        phase = params[1] if params is not None else 0
        w = (2 * np.pi) / period
        x = np.sin(w * n + phase)

    elif signal == 'random':
        if params == 'uniform':
            x = np.random.uniform(low=-1.5, high=1.5, size=N)
        else:
            x = np.random.normal(loc=0.0, scale=1.0, size=N)
    else:
        raise ValueError(f"Sinal '{signal}' desconhecido.")

    return x, n

generate = SIGNALgenerate # nome antigo

def SIGNALplot(x, n=None, xlim=None):
    """
    Plota x[n]. Se n for omitido, usa os indices 0..N-1.
    n pode ser amostras, tempo (s) ou frequencia (Hz).
    """
    if n is None:
        n = np.arange(len(x))

    fig, ax = plt.subplots()
    if len(x) <= 200:
        ax.stem(n, x, linefmt='b', markerfmt='bo')
    else:
        ax.plot(n, x, 'b') # stem fica lento para sinais longos
    if xlim is not None:
        ax.set_xlim(xlim)
    plt.show()

def play(x, fs):
    """
    Toca o sinal x com frequencia de amostragem fs.
    """
    display(Audio(x, rate=fs))
