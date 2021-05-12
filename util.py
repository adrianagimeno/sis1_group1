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
 
def plot_audio(t, y, name='audio signal'):
    if y is not list:
        data_plot = [
            go.Scatter(
                x=t, y=y, name=name
            )    
        ]
    else:
        data_plot = []
        for j in range(y):
            data_plot.append(
                go.Scatter(
                    x=t, y=y[j], name="{} {:d}".format(name, j)
                )    
            )
    iplot(data_plot)
