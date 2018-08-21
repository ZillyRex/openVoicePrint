import librosa
import matplotlib.pyplot as plt
from matplotlib import cm 
from matplotlib import axes
import numpy as np


def draw_heatmap(data, xlabels, ylabels):

    cmap = cm.Blues

    figure = plt.figure(facecolor='w')

    # ax = figure.add_subplot(2, 1, 1, position=[0.1, 0.15, 0.8, 0.8])
    ax = figure.add_subplot(1, 1, 1)

    ax.set_yticks(range(len(ylabels)))

    ax.set_yticklabels(ylabels)

    ax.set_xticks(range(len(xlabels)))

    ax.set_xticklabels(xlabels)

    vmax = data[0][0]

    vmin = data[0][0]

    for i in data:

        for j in i:

            if j > vmax:

                vmax = j

            if j < vmin:

                vmin = j

    map = ax.imshow(data, interpolation='nearest', cmap=cmap,
                    aspect='auto', vmin=vmin, vmax=vmax)

    cb = plt.colorbar(mappable=map, cax=None, ax=None, shrink=0.5)

    plt.show()


path = './test.wav'
y, sr = librosa.load(path, sr=44100)
X=[i for i in range(424)]
Y=[i for i in range(442)]
y=y.reshape(424,442)
draw_heatmap(y,X,Y)
