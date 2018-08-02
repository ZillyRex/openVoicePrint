import librosa

path = './test.wav'
y, sr = librosa.load(path, sr=44100)
mfccs = librosa.feature.mfcc(y, n_mfcc=39)
print('MFCC:')
print(mfccs)
print('MFCC shape:')
print(mfccs.shape)
print('wave data:')
print(y)
print('data shape:')
print(y.shape)
print('sample rate:')
print(sr)
