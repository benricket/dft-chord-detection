import sys
import numpy as np
import scipy
import librosa
import matplotlib.pyplot as plt
import matplotlib.colors

SAMPLE_INDEX_OFFSET = 0 # For every peak diff, assume our note is prominent this many samples later
DO_LOGGING = False
ARPEGGIO_THRESH = 0.25 # seconds
ARPEGGIO_THRESH_STEPS = 20

def main(path: str):
    """
    Runs the chord detection algorithm, which is contained within one function
    for simplicity's sake

    Args:
        path (str): File path to the audio file to process
    """
    data,sample_rate = librosa.load(path)
    print(data)
    print(f"Audio data shape: {data.shape}")


    processed = librosa.stft(data,n_fft=4096,win_length=4096,hop_length=512)
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

    low_hz = 10.0
    high_hz = 1000.0
    low_idx = np.where(freq >= low_hz)[0][0]
    high_idx = np.where(freq >= high_hz)[0][0]

    if DO_LOGGING:
        print(f"low idx, hi idx: {low_idx}, {high_idx}")

    # Filtered amplitudes and frequencies
    real = real[low_idx:high_idx,:]
    freq = np.array(freq[low_idx:high_idx])

    real_diff = np.diff(real,axis=1)
    real_diff_sum = np.sum(np.abs(real_diff),axis=0)
    runtime_seconds = len(data)/sample_rate

    quantile = 0.8
    diff_thresh = np.quantile(real_diff_sum,quantile)

    peaks = scipy.signal.find_peaks(real_diff_sum,prominence=5)

    chord_indices = peaks[0] + SAMPLE_INDEX_OFFSET
    chord_indices = [idx for idx in chord_indices if idx < real.shape[1] and real_diff_sum[idx] > diff_thresh]
    chord_indices_filtered = []
    chord_times = []
    chords = {}

    print([(i,val) for i,val in enumerate(chord_indices)])
    for i,val in enumerate(chord_indices):
        if i < len(chord_indices) - 1: # if we're not the last note
            if chord_indices[i+1] - val <= ARPEGGIO_THRESH_STEPS: # notes are in an arpeggio
                continue

        # We've detected a chord at this sample index
        progress_seconds = (val / len(real_diff_sum)) * runtime_seconds
        print(f"seconds: {progress_seconds}")
        seconds_rounded = round(progress_seconds * 1000) / 1000
        chord_indices_filtered.append(val)
        chord_times.append(progress_seconds)
        chord_freqs = real[:,val]

        # Identify most prominent frequencies
        freq_peaks_idx = scipy.signal.find_peaks(chord_freqs,prominence=2)
        freq_peaks = freq[freq_peaks_idx[0]]
        if len(freq_peaks) == 0:
            # it's hard to find peak frequencies with the prominence we want, be more permissive
            freq_peaks_idx = scipy.signal.find_peaks(chord_freqs)
            freq_peaks = freq[freq_peaks_idx[0]]

        # Get note names for identified frequencies
        notes = librosa.hz_to_note(freq_peaks)
        chords[seconds_rounded] = notes

    if DO_LOGGING:
        print(f"Peaks: {peaks}")
        print(f"Shape of diff: {real_diff.shape}")
        print(f"Real diff sum: {real_diff_sum}")
        print(f"shape: {real.shape}")
        print(f"Min real: {np.min(real)}, max real: {np.max(real)}, std dev: {np.std(real)}")
        print(f"Freqs: {freq}")
    print(f"{progress_seconds} seconds: {librosa.hz_to_note(freq_peaks)}")

    for s,n in chords.items():
        print(f"{s} seconds: {n}")

    _,ax = plt.subplots(2,1)
    print(freq)
    im0 = ax[0].imshow(real,norm=matplotlib.colors.LogNorm(),cmap="plasma",extent=[0,len(data)/sample_rate,freq[-1],0.0],aspect=0.002)
    ax[0].set_xlabel("Time (seconds)")
    ax[0].set_ylabel("Frequency component (Hz)")
    ax[0].set_title("Spectrogram of audio track, displaying log frequency magnitudes for each time window")
    ax[1].plot(real_diff_sum)
    ax[1].set_xlabel("Time (seconds)")
    ax[1].set_ylabel("Sum of absolute frequency \nmagnitude differences (unitless)")
    ax[1].set_title("Sums across all frequencies of absolute differences in frequency magnitude between time windows")
    ax[1].scatter(chord_indices_filtered,real_diff_sum[chord_indices_filtered],s=10,c='red')

    ticks = ax[1].get_xticks()
    labels_source = np.linspace(0, len(data)/sample_rate, len(ticks))
    step = 1
    labels = [
        f"{v:.0f}" if i % step == 0 else ""
        for i, v in enumerate(labels_source)
    ]
    ax[1].set_xticks(ticks)
    ax[1].set_xticklabels(labels)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Must pass a path to an audio file as input!")
        sys.exit()
    else:
        main(path=sys.argv[1])