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

    processed = librosa.stft(data,n_fft=2048)
    freq = librosa.fft_frequencies(sr=sample_rate,n_fft=2048)
    print(f"Processed: {processed}")
    print(f"Processed shape: {processed.shape}")

    real = np.abs(processed)
    print(f"shape: {real.shape}")
    #print(f"real: {real}")

    # Octave 4
    notes_4 = {
        261.63: "C",
        277.18: "C#",
        293.66: "D",
        311.13: "D#",
        329.63: "E",
        349.23: "F",
        369.99: "F#",
        392.00: "G",
        415.30: "G#",
        440.00: "A",
        466.16: "A#",
        493.88: "B"
    }
    notes = []
    for i in range(9):
        notes.append({k / (2**(4-i)): v for k,v in notes_4.items()})
    
    print (notes)

    low_hz = 10.0
    high_hz = 1000.0
    low_idx = np.where(freq >= low_hz)[0][0]
    high_idx = np.where(freq >= high_hz)[0][0]
    print(f"low idx, hi idx: {low_idx}, {high_idx}")

    thresh = np.quantile(real.flat,0.90)
    real_masked = (real > thresh) * real

    # Filtered amplitudes and frequencies
    real = real[low_idx:high_idx,:]
    freq = freq[low_idx:high_idx]

    print(f"shape: {real.shape}")
    print(f"Min real: {np.min(real)}, max real: {np.max(real)}, std dev: {np.std(real)}")
    print(f"Freqs: {freq}")
    fig,ax = plt.subplots(2,1)
    print(len(data)/sample_rate)
    im0 = ax[0].imshow(real,norm=matplotlib.colors.LogNorm(),cmap="plasma",extent=[0,len(data)/sample_rate,freq[-1],0.0],aspect=0.001)
    im_mask = ax[1].imshow(real_masked,norm=matplotlib.colors.LogNorm(),cmap="plasma",extent=[0,len(data)/sample_rate,freq[-1],0.0],aspect=0.001)
    plt.colorbar(im0)
    plt.show()

if __name__ == "__main__":
    main()