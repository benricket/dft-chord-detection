import sys
import os
import time
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.colors

def main(path="chords.mp3"):
    data,sample_rate = librosa.load(path)
    print(data)
    print(f"Audio data shape: {data.shape}")

    processed = librosa.stft(data,n_fft=1024)
    print(f"Processed: {processed}")
    print(f"Processed shape: {processed.shape}")

    real = np.real(processed)
    print(f"shape: {real.shape}")
    print(f"Min real: {np.min(real)}, max real: {np.max(real)}, std dev: {np.std(real)}")
    fig,ax = plt.subplots()
    im = ax.imshow(real,norm=matplotlib.colors.LogNorm(),cmap="plasma")
    plt.colorbar(im)
    plt.show()

if __name__ == "__main__":
    main()