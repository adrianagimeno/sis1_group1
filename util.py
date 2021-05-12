import os
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import librosa
import IPython.display as ipd
from plotly.offline import iplot
import plotly.graph_objs as go
from scipy import signal

def load_audio(filepath):
  data, sr = sf.read(filepath)

  # Convert to mono
  if len(data.shape) > 1:
    data = data[:, 0]

  # Remove DC component
  data = data - np.mean(data)

  return data, sr
 
def plot_audio(y, sr, t_start=0, t_end=-1, name='audio signal'):
    if type(y) is not list:
        y = [y]

    if type(name) is not list:
        names = [name + ' ' + str(j) for j in range(len(y))]
    else:
        assert len(name) == len(y)
        names = name

    Ts = 1/sr

    t = np.linspace(0, len(y[0])*Ts, len(y[0]))

    if t_end == -1:
        t_end = len(y[0])*Ts

    samples_start = int(t_start*sr)
    samples_end = int(t_end*sr)

    data_plot = []
    for j in range(len(y)):
        data_plot.append(
            go.Scatter(
                x=t[samples_start:samples_end], y=y[j][samples_start:samples_end], name=names[j]
            )    
        )
    iplot(data_plot)
