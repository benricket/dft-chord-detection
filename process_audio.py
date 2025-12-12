import sys
import os
import time
import numpy as np
import scipy
import librosa
import matplotlib.pyplot as plt
import matplotlib.colors

SAMPLE_INDEX_OFFSET = 0 # For every peak difference, we assume our note is prominent this many samples later
ARPEGGIO_THRESH = 0.25 # seconds 
ARPEGGIO_THRESH_SAMPLES = 20

def main(path="chord_test.mp3"):
    data,sample_rate = librosa.load(path)
    print(data)
    print(f"Audio data shape: {data.shape}")

    processed = librosa.stft(data,n_fft=4096)
    freq = librosa.fft_frequencies(sr=sample_rate,n_fft=4096)
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
    
    #print (notes)

    low_hz = 10.0
    high_hz = 1000.0
    low_idx = np.where(freq >= low_hz)[0][0]
    high_idx = np.where(freq >= high_hz)[0][0]
    print(f"low idx, hi idx: {low_idx}, {high_idx}")

    thresh = np.quantile(real.flat,0.90)
    real_masked = (real > thresh) * real

    # Filtered amplitudes and frequencies
    real = real[low_idx:high_idx,:]
    freq = np.array(freq[low_idx:high_idx])

    real_diff = np.diff(real,axis=1)
    real_diff_sum = np.sum(np.abs(real_diff),axis=0)
    runtime_seconds = len(data)/sample_rate

    quantile = 0.95
    diff_thresh = np.quantile(real_diff_sum,quantile)

    peaks = scipy.signal.find_peaks(real_diff_sum,prominence=5)

    chord_indices = peaks[0] + SAMPLE_INDEX_OFFSET
    chord_indices = [idx for idx in chord_indices if idx < real.shape[1] and real_diff_sum[idx] > diff_thresh]

    for i in range(len(chord_indices)):
        chord_index = chord_indices[i]
        if chord_index < len(chord_indices) - 1: # if we're not the last note
            if chord_indices[i+1] - chord_index <= ARPEGGIO_THRESH_SAMPLES: # notes are in an arpeggio
                continue

        # We've detected a chord at this sample index
        chord_freqs = real[:,chord_index]
        freq_peaks_idx = scipy.signal.find_peaks(chord_freqs,prominence=2)
        freq_peaks = freq[freq_peaks_idx[0]]
        #print(f"Frequencies of notes: {freq_peaks}")

        progress_seconds = (chord_index / len(real_diff_sum)) * runtime_seconds
        print(f"{progress_seconds} seconds: {librosa.hz_to_note(freq_peaks)}")

    #print(f"Peaks: {peaks}")

    #print(f"Shape of diff: {real_diff.shape}")
    #print(f"Real diff sum: {real_diff_sum}")

    #print(f"shape: {real.shape}")
    #print(f"Min real: {np.min(real)}, max real: {np.max(real)}, std dev: {np.std(real)}")
    #print(f"Freqs: {freq}")
    fig,ax = plt.subplots(3,1)
    #print(len(data)/sample_rate)
    im0 = ax[0].imshow(real,norm=matplotlib.colors.LogNorm(),cmap="plasma",extent=[0,len(data)/sample_rate,freq[-1],0.0],aspect=0.001)
    im_mask = ax[1].imshow(real_diff,norm=matplotlib.colors.LogNorm(),cmap="plasma",extent=[0,len(data)/sample_rate,freq[-1],0.0],aspect=0.001)
    diffs = ax[2].plot(real_diff_sum)
    ax[2].scatter(chord_indices,real_diff_sum[chord_indices],s=10,c='red')
    plt.colorbar(im0)
    plt.show()

if __name__ == "__main__":
    main()