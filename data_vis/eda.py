import os
import gc
import time
import math
#LOOK AT NUMBA FOR PARALLELIZATION
#from numba import jit
from math import log, floor
from sklearn.neighbors import KDTree

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle

import seaborn as sns
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.io as pio
from plotly.offline import plot

import pywt 
from statsmodels.robust import mad

import scipy
from scipy import signal
from scipy.signal import butter, deconvolve

SAMPLE_RATE = 25
SIGNAL_LEN = 1000
pio.renderers.default = "iframe"

TEST_PATH = "../data/test.csv"
TRAIN_PATH = "../data/train.csv"
SUBMISSION_PATH = "../data/sample_submission.csv"

test_data = pd.read_csv(TEST_PATH)
train_data = pd.read_csv(TRAIN_PATH)
test_data.drop(columns=['time'], inplace=True)
print(train_data.head())

#signal data vs time
plt.figure(figsize=(20, 10))
plt.plot(train_data["time"], train_data["signal"], color="r")
plt.title("Signal data", fontsize=20)
plt.xlabel("Time", fontsize=18)
plt.ylabel("Signal", fontsize=18)
plt.show()

def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise_signal(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    return pywt.waverec(coeff, wavelet, mode='per')

#not implemented here, but just change y_w1 etc

def average_smoothing(signal, kernel_size=3, stride=1):
    sample = []
    start = 0
    end = kernel_size
    while end <= len(signal):
        start = start + stride
        end = end + stride
        sample.extend(np.ones(end - start)*np.mean(signal[start:end]))
    return np.array(sample)

x = train_data.loc[:100]["time"]
y1 = train_data.loc[:100]["signal"]
y_w1 = denoise_signal(train_data.loc[:100]["signal"])
y2 = train_data.loc[100:200]["signal"]
y_w2 = denoise_signal(train_data.loc[100:200]["signal"])
y3 = train_data.loc[200:300]["signal"]
y_w3 = denoise_signal(train_data.loc[200:300]["signal"])

fig = make_subplots(rows=3, cols=1)

fig.add_trace(
    go.Scatter(x=x, mode='lines+markers', y=y1, marker=dict(color="lightskyblue"), showlegend=False,
               name="Original signal"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=x, y=y_w1, mode='lines', marker=dict(color="navy"), showlegend=False,
               name="Denoised signal"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=x, mode='lines+markers', y=y2, marker=dict(color="mediumaquamarine"), showlegend=False),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=x, y=y_w2, mode='lines', marker=dict(color="darkgreen"), showlegend=False),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=x, mode='lines+markers', y=y3, marker=dict(color="thistle"), showlegend=False),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=x, y=y_w3, mode='lines', marker=dict(color="indigo"), showlegend=False),
    row=3, col=1
)

fig.update_layout(height=1200, width=800, title_text="Original (pale) vs. Denoised (dark) signals")
fig.show()

#open channel distributions
fig = go.Figure(data=[
    go.Bar(x=list(range(11)), y=train_data['open_channels'].value_counts(sort=False).values)
])

fig.update_layout(title='Target (open_channels) distribution')
plot(fig, filename='open_dist.html')

signals = []
targets = []

train = train_data # shuffle(train_data).reset_index(drop=True)
for i in range(4000):
    min_lim = SIGNAL_LEN * i
    max_lim = SIGNAL_LEN * (i + 1)
    
    signals.append(list(train["signal"][min_lim : max_lim]))
    targets.append(train["open_channels"][max_lim])
    
signals = np.array(signals)
targets = np.array(targets)

#mean signal vs channels
df = pd.DataFrame(np.transpose([np.mean(np.abs(signals), axis=1), targets]))
df.columns = ["signal_mean", "open_channels"]
fig1 = go.Figure()

channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for channel in channels:
    fig1.add_trace(go.Box(x=df['open_channels'][df['open_channels'] == channel],
                         y=df['signal_mean'][df['open_channels'] == channel],
                         name=channel,
                         marker=dict(color='seagreen'), showlegend=False)
                         )
    
fig1.add_trace(go.Scatter(x=channels,
                         y=[df['signal_mean'][df['open_channels'] == channel].median() for channel in channels],
                         mode="lines+markers",
                         name=channel,
                         marker=dict(color='seagreen'), showlegend=False)
                         )

fig1.update_layout(title="Signal mean vs. Open channels", xaxis_title="Open channels", yaxis_title="Signal mean")
plot(fig1, filename='signal_mean.html')

def _embed(x, order=3, delay=1):
    N = len(x)
    if order * delay > N:
        raise ValueError("Error: order * delay should be lower than x.size")
    if delay < 1:
        raise ValueError("Delay has to be at least 1.")
    if order < 2:
        raise ValueError("Order has to be at least 2.")
    Y = np.zeros((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T

all = ['perm_entropy', 'spectral_entropy', 'svd_entropy', 'app_entropy',
       'sample_entropy']

def perm_entropy(x, order=3, delay=1, normalize=False):
    x = np.array(x)
    ran_order = range(order)
    hashmult = np.power(order, ran_order)
    # Embed x and sort the order of permutations
    sorted_idx = _embed(x, order=order, delay=delay).argsort(kind='quicksort')
    # Associate unique integer to each permutations
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    # Return the counts
    _, c = np.unique(hashval, return_counts=True)
    # Use np.true_divide for Python 2 compatibility
    p = np.true_divide(c, c.sum())
    pe = -np.multiply(p, np.log2(p)).sum()
    if normalize:
        pe /= np.log2(factorial(order))
    return pe

#permutation entropy vs open channels
df = pd.DataFrame(np.transpose([[perm_entropy(row) for row in signals], targets]))
df.columns = ["perm_entropy", "open_channels"]
fig2 = go.Figure()

channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for channel in channels:
    fig2.add_trace(go.Box(x=df['open_channels'][df['open_channels'] == channel],
                         y=df['perm_entropy'][df['open_channels'] == channel],
                         name=channel,
                         marker=dict(color='blueviolet'), showlegend=False)
                         )
    
fig2.add_trace(go.Scatter(x=channels,
                         y=[df['perm_entropy'][df['open_channels'] == channel].median() for channel in channels],
                         mode="lines+markers",
                         name=channel,
                         marker=dict(color='blueviolet'), showlegend=False)
                         )

fig2.update_layout(title="Permutation entropy vs. Open channels", xaxis_title="Open channels", yaxis_title="Permutation entropy")
plot(fig2, filename='permutation.html')

df = pd.DataFrame(np.transpose([[perm_entropy(row) for row in signals], targets]))
df.columns = ["perm_entropy", "open_channels"]
fig3 = go.Figure()

channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

fig3.add_trace(go.Scatter(x=channels,
                         y=[df['perm_entropy'][df['open_channels'] == channel].median() for channel in channels],
                         mode="lines+markers",
                         name=channel,
                         marker=dict(color='blueviolet'), showlegend=False)
                         )

fig3.update_layout(title="Median permutation entropy vs. Open channels", xaxis_title="Open channels", yaxis_title="Median permutation entropy")
plot(fig3, filename='mean_permutation.html')

def _log_n(min_n, max_n, factor):
    max_i = int(floor(log(1.0 * max_n / min_n) / log(factor)))
    ns = [min_n]
    for i in range(max_i + 1):
        n = int(floor(min_n * (factor ** i)))
        if n > ns[-1]:
            ns.append(n)
    return np.array(ns, dtype=np.int64)

def _higuchi_fd(x, kmax):
    n_times = x.size
    lk = np.empty(kmax)
    x_reg = np.empty(kmax)
    y_reg = np.empty(kmax)
    for k in range(1, kmax + 1):
        lm = np.empty((k,))
        for m in range(k):
            ll = 0
            n_max = floor((n_times - m - 1) / k)
            n_max = int(n_max)
            for j in range(1, n_max):
                ll += abs(x[m + j * k] - x[m + (j - 1) * k])
            ll /= k
            ll *= (n_times - 1) / (k * n_max)
            lm[m] = ll
        # Mean of lm
        m_lm = 0
        for m in range(k):
            m_lm += lm[m]
        m_lm /= k
        lk[k - 1] = m_lm
        x_reg[k - 1] = log(1. / k)
        y_reg[k - 1] = log(m_lm)
    higuchi, _ = _linear_regression(x_reg, y_reg)
    return higuchi

def higuchi_fd(x, kmax=10):
    x = np.asarray(x, dtype=np.float64)
    kmax = int(kmax)
    return _higuchi_fd(x, kmax)

def _linear_regression(x, y):
    n_times = x.size
    sx2 = 0
    sx = 0
    sy = 0
    sxy = 0
    for j in range(n_times):
        sx2 += x[j] ** 2
        sx += x[j]
        sxy += x[j] * y[j]
        sy += y[j]
    den = n_times * sx2 - (sx ** 2)
    num = n_times * sxy - sx * sy
    slope = num / den
    intercept = np.mean(y) - slope * np.mean(x)
    return slope, intercept

#higuchi fractals vs open channels
df = pd.DataFrame(np.transpose([[higuchi_fd(row) for row in signals], targets]))
df.columns = ["higuchi_fd", "open_channels"]
fig4 = go.Figure()

channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for channel in channels:
    fig4.add_trace(go.Box(x=df['open_channels'][df['open_channels'] == channel],
                         y=df['higuchi_fd'][df['open_channels'] == channel],
                         name=channel,
                         marker=dict(color='orange'), showlegend=False)
                         )
    
fig4.add_trace(go.Scatter(x=channels,
                         y=[df['higuchi_fd'][df['open_channels'] == channel].median() for channel in channels],
                         mode="lines+markers",
                         name=channel,
                         marker=dict(color='orange'), showlegend=False)
                         )

fig4.update_layout(title="Higuchi fractal dimension vs. Open channels", xaxis_title="Open channels", yaxis_title="Higuchi fractal dimension")
plot(fig4, filename='higuchi.html')