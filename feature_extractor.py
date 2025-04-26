import numpy as np
def format_time(ms):
    """Formats milliseconds to HH:MM:SS.MMM format."""
    s = ms // 1000 
    m, s = divmod(s, 60) 
    h, m = divmod(m, 60)  
    ms_remaining = ms % 1000
    return f"{h}:{m:02}:{s:02}.{ms_remaining:03}"

def custom_correlate(x, y):
    """
    Compute the cross-correlation of two 1D arrays.
    Assumes both input arrays are of the same length.
    """
    N = len(x)
    result = [0] * N
    
    for lag in range(N):
        sum_val = 0
        for i in range(N - lag):
            sum_val += x[i] * y[i + lag]
        result[lag] = sum_val
    
    return result

def extract_frequency_features(audio: np.ndarray, sr: int, frame_length: int, window_type: str = 'hanning'):
    """Extracts frame-level frequency domain features from the audio signal."""
    step = frame_length
    frames = [audio[i:i + frame_length] for i in range(0, len(audio) - frame_length + 1, step)]
    window = get_window(window_type, frame_length)
    freqs = np.fft.rfftfreq(frame_length, d=1/sr)

    features = []

    # Frequency bands for ERSB (in Hz)
    bands = [(0, 630), (630, 1720), (1720, 4400), (4400, 11025)]

    for frame in frames:
        if len(window) == frame_length:
            frame = frame * window
        elif len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)), 'constant')
            frame = frame * window[:len(frame)]
        spectrum = np.fft.rfft(frame)
        power_spectrum = np.abs(spectrum) ** 2

        # Volume
        volume = np.mean(power_spectrum)

        # Frequency Centroid
        fc = np.sum(freqs * power_spectrum) / (np.sum(power_spectrum) + 1e-10)

        # Effective Bandwidth
        bw = np.sqrt(np.sum(((freqs - fc) ** 2) * power_spectrum) / (np.sum(power_spectrum) + 1e-10))

        # ERSBs
        total_energy = np.sum(power_spectrum)
        ersb = []
        for f_low, f_high in bands[:3]:
            indices = np.where((freqs >= f_low) & (freqs < f_high))
            band_energy = np.sum(power_spectrum[indices])
            ersb.append(band_energy / (total_energy + 1e-10))  # ERSB1, 2, 3

        # Spectral Flatness Measure (SFM)
        geometric_mean = np.exp(np.mean(np.log(power_spectrum + 1e-10)))
        arithmetic_mean = np.mean(power_spectrum + 1e-10)
        sfm = geometric_mean / arithmetic_mean

        # Spectral Crest Factor (SCF)
        scf = np.max(power_spectrum) / (arithmetic_mean + 1e-10)

        features.append((volume, fc, bw, *ersb, sfm, scf))

    return features
 
def get_window(window_type, frame_length):
    """Returns the window function of the specified type."""
    if window_type == 'hanning':
        return np.hanning(frame_length)
    elif window_type == 'hamming':
        return np.hamming(frame_length)
    elif window_type == 'blackman':
        return np.blackman(frame_length)
    elif window_type == 'bartlett':
        return np.bartlett(frame_length)
    elif window_type == 'rectangular':
        return np.ones(frame_length)
    elif window_type == 'triangular':
        return np.bartlett(frame_length)
    else:
        raise ValueError(f"Unsupported window type: {window_type}")

def compute_real_cepstrum(signal):
    """
    Computes the real cepstrum of a signal using the FFT method.
    """
    spectrum = np.fft.fft(signal)
    log_magnitude = np.log(np.abs(spectrum) + 1e-10) 
    cepstrum = np.fft.ifft(log_magnitude).real
    
    return cepstrum

def estimate_f0_from_cepstrum(cepstrum, sr, min_freq=50, max_freq=400):
    """
    Estimates the fundamental frequency (F0) from the cepstrum using a peak-picking method.
    """
    min_quefrency = 1 / 400  # 0.0025 s
    max_quefrency = 1 / 50   # 0.02 s
    min_index = int(min_quefrency * sr)
    max_index = int(max_quefrency * sr)

    search_region = cepstrum[min_index:max_index]
    peak = np.argmax(search_region) + min_index
    f0 = sr / peak if peak > 0 else 0
    return f0, peak, min_index, max_index