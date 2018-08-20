import librosa
import matplotlib.pyplot as plt

path = './test.wav'
y, sr = librosa.load(path, sr=44100)
plt.plot(y)
plt.show()