import sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QLabel, QStyle, QSlider, QRadioButton, QButtonGroup, QHBoxLayout, QSpinBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import QUrl, QTimer, Qt
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.lines import Line2D as mlines
import os
import csv
import pandas as pd
from feature_extractor import format_time, extract_frequency_features, get_window, compute_real_cepstrum, estimate_f0_from_cepstrum

class AudioAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.audio = None
        self.sr = None
        self.duration = 0
        self.features = [] 
        self.clip_features = []
        self.mini_clip_features = [] 
        self.silence_threshold = 0.1
        self.frame_length = None
        self.selected_frame = None
        self.is_clips = False
        self.window_type = "hanning"
        self.overlapping = 0

        self.setWindowTitle("Audio Analyzer")
        self.setGeometry(100, 100, 1000, 900)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QHBoxLayout()
        self.central_widget.setLayout(self.layout)

        # loading section
        self.info_layout = QVBoxLayout()
        self.load_button = QPushButton("Load Audio File")
        self.load_button.clicked.connect(self.load_audio)
        self.info_layout.addWidget(self.load_button)

        # player section
        self.player_layout = QVBoxLayout()
        self.audio_name_label = QLabel("")
        self.bold_font = self.audio_name_label.font()
        self.bold_font.setBold(True)
        self.audio_name_label.setFont(self.bold_font)
        self.player_layout.addWidget(self.audio_name_label)
        
        self.player = QMediaPlayer(self)  
        self.audio_output = QAudioOutput() 
        self.player.setAudioOutput(self.audio_output)  
        self.player.playbackStateChanged.connect(self.audio_changed)
        
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_position)

        self.player_label = QLabel(text='0:00:00 / 0:00:00')
        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_button.clicked.connect(self.audio_play)
        self.player_layout.addWidget(self.player_label)
        self.player_layout.addWidget(self.play_button)
        self.info_layout.addLayout(self.player_layout)

        #add window options: rectangular, tringular, hamming, van hanning, blackman, hamming
        self.window_label = QLabel("Window Type")
        self.window_label.setFont(self.bold_font)
        self.info_layout.addWidget(self.window_label)
        self.window_button_layout = QVBoxLayout()
        self.rectangular_button = QRadioButton("Rectangular")
        self.triangular_button = QRadioButton("Triangular")
        self.hamming_button = QRadioButton("Hamming")
        self.blackman_button = QRadioButton("Blackman")
        self.hanning_button = QRadioButton("Hanning")
        self.window_button_group = QButtonGroup(self)
        self.window_button_group.addButton(self.rectangular_button)
        self.window_button_group.addButton(self.triangular_button)
        self.window_button_group.addButton(self.hamming_button)
        self.window_button_group.addButton(self.blackman_button)
        self.window_button_group.addButton(self.hanning_button)
        self.hanning_button.setChecked(True)
        self.window_button_layout.addWidget(self.rectangular_button)
        self.window_button_layout.addWidget(self.triangular_button)
        self.window_button_layout.addWidget(self.hamming_button)
        self.window_button_layout.addWidget(self.blackman_button)
        self.window_button_layout.addWidget(self.hanning_button)
        self.info_layout.addLayout(self.window_button_layout)
        self.rectangular_button.toggled.connect(self.update_window)
        self.triangular_button.toggled.connect(self.update_window)
        self.hamming_button.toggled.connect(self.update_window)
        self.blackman_button.toggled.connect(self.update_window)
        self.hanning_button.toggled.connect(self.update_window)
        self.window_button_group.setExclusive(True)
        

        #frame-length buttons
        self.selection_label = QLabel("Frame length:")
        self.selection_label.setFont(self.bold_font)
        self.info_layout.addWidget(self.selection_label)
        self.button_layout = QHBoxLayout()
        self.option1 = QRadioButton("10ms")
        self.option2 = QRadioButton("20ms")
        self.option3 = QRadioButton("30ms")
        self.option4 = QRadioButton("40ms")
        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.option1)
        self.button_group.addButton(self.option2)
        self.button_group.addButton(self.option3)
        self.button_group.addButton(self.option4)
        self.option2.setChecked(True)
        self.button_layout.addWidget(self.option1)
        self.button_layout.addWidget(self.option2)
        self.button_layout.addWidget(self.option3)
        self.button_layout.addWidget(self.option4)
        self.info_layout.addLayout(self.button_layout)
        self.option1.toggled.connect(self.update_selection)
        self.option2.toggled.connect(self.update_selection)
        self.option3.toggled.connect(self.update_selection)
        self.option4.toggled.connect(self.update_selection)


        # frame/clip info layout
        self.frame_info_label = QLabel("Frame Info")
        self.frame_info_label.setFont(self.bold_font)
        self.info_layout.addWidget(self.frame_info_label)
        self.frame_info_l2 = QLabel("")
        self.info_layout.addWidget(self.frame_info_l2)

        # whole audio info layout
        self.audio_info_label = QLabel("Audio Info")
        self.audio_info_label.setFont(self.bold_font)
        self.info_layout.addWidget(self.audio_info_label)
        self.audio_info_l2 = QLabel("")
        self.info_layout.addWidget(self.audio_info_l2)

        self.is_music_label = QLabel("")
        self.is_music_label.setFont(self.bold_font)
        self.info_layout.addWidget(self.is_music_label)

        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 4096)
        self.overlap_spin.setValue(0)
        self.overlap_spin.valueChanged.connect(self.update_overlapping)
        self.overlap_label = QLabel("Overlap (ms):")
        self.overlap_label.setFont(self.bold_font)
        self.info_layout.addWidget(self.overlap_label)
        self.info_layout.addWidget(self.overlap_spin)


        # plots - right side layout
        self.plots_layout = QVBoxLayout()
        self.figure, self.ax = plt.subplots(6, 1, figsize=(10, 14), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.plots_layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.plots_layout.addWidget(self.toolbar)

        self.file_name_path = None

        self.selected_option = "20ms"

        # right-side extra plots
        self.extra_plots_layout = QVBoxLayout()
        self.extra_figure, self.extra_ax = plt.subplots(3, 1, figsize=(5, 6), constrained_layout=True)
        self.extra_canvas = FigureCanvas(self.extra_figure)
        self.extra_toolbar = NavigationToolbar2QT(self.extra_canvas, self)

        self.extra_plots_layout.addWidget(self.extra_canvas)
        self.extra_plots_layout.addWidget(self.extra_toolbar)

        self.layout.addLayout(self.info_layout)
        self.layout.addLayout(self.plots_layout, 1)          # Main plot area: takes equal space
        self.layout.addLayout(self.extra_plots_layout, 1) 
    
    def load_audio(self):
        """Loads an audio file and processes it."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "WAV Files (*.wav)")
        if file_path:
            self.option2.setChecked(True)
            self.audio, self.sr = librosa.load(file_path, sr=None)
            self.frame_length = int(0.02 * self.sr)
            self.process_audio()
            self.player.setSource(QUrl.fromLocalFile(file_path))
            self.audio_name_label.setText(os.path.basename(file_path))
            self.file_name_path = os.path.basename(file_path)
    
    def process_audio(self):
        """Processes the audio file and extracts features."""
        self.features = extract_frequency_features(self.audio, self.sr, self.frame_length, window_type=self.window_type)
        self.plot_audio(self.features)
        self.plot_spectrum_for_selection(0, len(self.audio) // self.frame_length, frame_length=self.frame_length)
        self.plot_spectrogram_for_selection(0, len(self.audio) // self.frame_length, frame_length=self.frame_length)
        self.plot_cepstrum_for_selection(0, len(self.audio) // self.frame_length, frame_length=self.frame_length)
        self.reset_frame_info()
        self.duration = len(self.audio) / self.sr * 1000 

    def reset_frame_info(self):
        self.frame_info_label.setText("Frame Info")
        self.frame_info_l2.setText("")
        self.selected_frame = None
    
    def plot_audio(self, features):
        """Plots the audio waveform and frequency-domain features."""
        self.figure.clear()
        volume, fc, bw, ersb1, ersb2, ersb3, sfm, scf = zip(*features)

        time_axis = np.arange(len(volume)) * (self.frame_length / self.sr)

        ax = self.figure.subplots(7, 1, sharex=True)
        self.ax = ax
        window = get_window(self.window_type, self.frame_length)
        
        self.audio_windowed = np.zeros_like(self.audio)
        for i in range(0, len(self.audio) - self.frame_length + 1, self.frame_length):
            self.audio_windowed[i:i + self.frame_length] = self.audio[i:i + self.frame_length] * window

        ax[0].plot(np.linspace(0, len(self.audio_windowed) / self.sr, num=len(self.audio_windowed)), self.audio_windowed, label='Waveform (Windowed)', color='steelblue')
        ax[0].set_title("Waveform")
        ax[0].set_ylabel("Amplitude")



        if not self.is_clips:
            ax, time_axis = self.plot_wavelength_frames(
                ax, time_axis, volume, fc, bw, ersb1, ersb2, ersb3, sfm, scf
            )
        else:
            ax, time_axis = self.plot_wavelength_clips(ax, time_axis)

        self.canvas.mpl_connect('button_press_event', self.on_click_waveform)
        self.canvas.draw()


    
    def plot_wavelength_frames(self, ax, time_axis, volume, fc, bw, ersb1, ersb2, ersb3, sfm, scf):
        '''Plots all frequency-domain features with dedicated subplots.'''

        for i in range(7):
            ax[i].set_xlim(0, len(self.audio) / self.sr)
            ax[i].grid(True)
        ax[6].set_xlabel("Time (s)")

        ax[1].plot(time_axis, volume, color='orange')
        ax[1].set_title("Volume")

        ax[2].plot(time_axis, fc, color='blue')
        ax[2].set_title("Frequency Centroid (Hz)")

        ax[3].plot(time_axis, bw, color='green')
        ax[3].set_title("Effective Bandwidth (Hz)")

        ax[4].plot(time_axis, ersb1, label='ERSB1 (0–630 Hz)', color='purple')
        ax[4].plot(time_axis, ersb2, label='ERSB2 (630–1720 Hz)', color='pink')
        ax[4].plot(time_axis, ersb3, label='ERSB3 (1720–4400 Hz)', color='brown')
        ax[4].set_title("Energy Ratio in Sub-Bands (ERSB1–3)")
        ax[4].legend(loc='upper right')
        for text in ax[4].get_legend().get_texts():
            text.set_fontsize(5)


        ax[5].plot(time_axis, sfm, color='cyan')
        ax[5].set_title("Spectral Flatness")

        ax[6].plot(time_axis, scf, color='red')
        ax[6].set_title("Spectral Crest")

        self.ax[0].callbacks.connect('xlim_changed', self.zoom_on_plot)
        return ax, time_axis

    def on_click_waveform(self, event):
        """Handles waveform clicks, updating frame or mini-clip info based on self.is_clips."""
        if event.inaxes is not None:
            time_point = event.xdata
            
            if time_point is not None:
                if self.is_clips:
                    clip_index = self.get_clip_index(time_point)
                    self.update_frame_info(clip_index)
                else:
                    frame_index = self.get_frame_index(time_point)
                    self.update_frame_info(frame_index)

                # Move playback to the clicked position
                self.player.setPosition(int(time_point * 1000))
                self.display_time(int(time_point * 1000))

    def zoom_on_plot(self, event):
        '''Zooms in on the waveform plot and updates the audio-level features.'''
        xlim = self.ax[0].get_xlim() 
        start_time, end_time = xlim  
        #print(f"Zoomed from {start_time:.2f}s to {end_time:.2f}s")

        start_frame = self.get_frame_index(start_time)
        end_frame = self.get_frame_index(end_time)
        
        if start_frame < end_frame:
            self.audio_info_label.setText("Audio Info")
            audio_info_text = "Start Time: {:.2f}s\nEnd Time: {:.2f}s\n".format(start_time, end_time)
            self.audio_info_l2.setText(audio_info_text)
        frame_length = self.frame_length
        self.plot_spectrum_for_selection(start_frame, end_frame, frame_length=frame_length)
        self.plot_cepstrum_for_selection(start_frame, end_frame, frame_length=frame_length)
        self.extra_ax[1].set_xlim(start_time, end_time)
        self.extra_ax[1].set_ylim(0, min(self.sr / 2, 10000))
        self.extra_ax[1].set_xlabel("Time (s)")
        self.extra_ax[1].set_ylabel("Frequency (Hz)")
        self.extra_ax[1].set_title("Spectrogram")
        self.extra_canvas.draw()

    def plot_spectrum_for_selection(self, start_frame, end_frame, frame_length=441):
        """Plots frequency spectrum of the selected audio region on the first extra plot."""
        if self.audio_windowed is None:
            return

        selected_audio = self.audio_windowed[start_frame*frame_length:end_frame*frame_length + frame_length if end_frame*frame_length + frame_length < len(self.audio_windowed) else  len(self.audio_windowed)] 
        if selected_audio.ndim == 2: 
            selected_audio = selected_audio.mean(axis=1)

        if len(selected_audio) == 0:
            return
        # FFT
        spectrum = np.fft.fft(selected_audio)
        freqs = np.fft.fftfreq(len(selected_audio), d=1.0 / self.sr)

        magnitude = np.abs(spectrum[:len(spectrum)//2])
        freqs = freqs[:len(freqs)//2]

        self.extra_ax[0].clear()
        self.extra_ax[0].bar(freqs, magnitude, width=50)  # Adjust the width of the bars as necessary
        self.extra_ax[0].set_title("Frequency Spectrum")
        self.extra_ax[0].set_xlabel("Frequency [Hz]")
        self.extra_ax[0].set_ylabel("Magnitude")

        self.extra_canvas.draw()

    def plot_spectrogram_for_selection(self, start_frame, end_frame, frame_length=441, xlim=None, max_freq=10000):
        """Plots the spectrogram of the selected audio region on the second extra plot."""

        if self.audio is None:
            print("Audio data is not loaded.")
            return

        audio = self.audio.copy()
        start_sample = start_frame * frame_length
        end_sample = end_frame * frame_length + frame_length
        audio = audio[start_sample:end_sample if end_sample < len(audio) else len(audio)]

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        frame_length = self.frame_length
        overlap = self.overlapping
        window_type = self.window_type
        step = frame_length - overlap

        if step < 1 or frame_length > len(audio):
            print("Invalid step or frame length.")
            return

        window = get_window(window_type, frame_length)
        num_frames = 1 + (len(audio) - frame_length) // step
        spectrogram = []

        for i in range(num_frames):
            start = i * step
            frame = audio[start:start + frame_length] * window
            spectrum = np.abs(np.fft.rfft(frame))
            spectrogram.append(spectrum)

        spectrogram = np.array(spectrogram).T 
        spectrogram_db = np.log1p(spectrogram)

        time_axis = np.arange(num_frames) * step / self.sr
        freq_axis = np.fft.rfftfreq(frame_length, d=1.0 / self.sr)

        freq_mask = freq_axis <= max_freq
        freq_axis = freq_axis[freq_mask]
        spectrogram_db = spectrogram_db[freq_mask, :]
        self.extra_ax[1].clear()


        T, F = np.meshgrid(time_axis, freq_axis)
        pcm = self.extra_ax[1].pcolormesh(T, F, spectrogram_db, shading='auto', cmap='viridis')

        self.extra_ax[1].set_title("Spectrogram (Log Frequency)")
        self.extra_ax[1].set_xlabel("Time [s]")
        self.extra_ax[1].set_ylabel("Frequency [Hz]")
        self.extra_ax[1].set_yscale('log')
        self.extra_ax[1].set_ylim(freq_axis[1], max_freq)

        self.extra_ax[1].set_yticks([100, 1000, 10000])
        self.extra_ax[1].get_yaxis().set_major_formatter(plt.ScalarFormatter())

        self.extra_canvas.draw()

    def plot_cepstrum_for_selection(self, start_frame, end_frame, frame_length=441):
        """Plots the cepstrum of the selected audio region on the third extra plot."""
        if self.audio is None:
            print("Audio data is not loaded.")
            return

        audio = self.audio.copy()
        audio = audio[start_frame*frame_length:end_frame*frame_length + frame_length if end_frame*frame_length + frame_length < len(audio) else  len(audio)]
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        cepstrum = compute_real_cepstrum(audio)
        f0, peak_index, min_index, max_index = estimate_f0_from_cepstrum(cepstrum, self.sr)

        quefrency = np.arange(len(cepstrum)) / self.sr
        min_quefrency = min_index / self.sr
        max_quefrency = max_index / self.sr
        
        self.extra_ax[2].clear()
        self.extra_ax[2].plot(
            quefrency[min_index:max_index], 
            np.abs(cepstrum[min_index:max_index]), 
            label='Cepstrum', 
            color='blue'
        )
        self.extra_ax[2].plot(
            peak_index / self.sr, 
            np.abs(cepstrum[peak_index]), 
            'ro', 
            label='F0 Peak'
        )
        self.extra_ax[2].text(
            peak_index / self.sr, 
            np.max(np.abs(cepstrum[min_index:max_index])), 
            f'F0: {f0:.2f} Hz', 
            color='red', 
            fontsize=8, 
            ha='center'
        )
        self.extra_ax[2].set_title("Cepstrum")
        self.extra_ax[2].set_xlabel("Quefrency [s]")
        self.extra_ax[2].set_ylabel("Magnitude")
        self.extra_ax[2].legend()
        self.extra_ax[2].set_xlim(min_quefrency, max_quefrency)
        self.extra_ax[2].set_ylim(0, np.max(np.abs(cepstrum[min_index:max_index])) * 1.1)
        self.extra_canvas.draw()

    def get_frame_index(self, time_point):
        """ Returns the index of the frame corresponding to the given time point. """
        frame_index = int(time_point * self.sr / self.frame_length)
        if frame_index >= len(self.features):
            frame_index = len(self.features) - 1
        if frame_index < 0:
            frame_index = 0
        return frame_index

    def update_selection(self):
        """Updates the frame length based on the selected radio button."""
        if self.sr is None:
            return
        self.selected_option = "20ms"
        if self.option1.isChecked():
            self.selected_option = "10ms"
            self.frame_length = int(0.01 * self.sr)
        elif self.option2.isChecked():
            self.selected_option = "20ms"
            self.frame_length = int(0.02 * self.sr)
        elif self.option3.isChecked():
            self.selected_option = "30ms"
            self.frame_length = int(0.03 * self.sr)
        elif self.option4.isChecked():
            self.selected_option = "40ms"
            self.frame_length = int(0.04 * self.sr)
        self.process_audio()

    def update_overlapping(self, value):
        """Updates the overlapping value based on the spin box input."""
        self.overlapping = value
        xlim = self.ax[0].get_xlim()
        self.plot_spectrogram_for_selection(0, len(self.audio) // self.frame_length, frame_length=self.frame_length, xlim=xlim)
        self.extra_ax[1].set_xlim(xlim[0], xlim[1])
        self.extra_canvas.draw()
                
    def update_window(self):
        """Updates the window type based on the selected radio button."""
        if self.rectangular_button.isChecked():
            self.window_type = "rectangular"
        elif self.triangular_button.isChecked():
            self.window_type = "triangular"
        elif self.hamming_button.isChecked():
            self.window_type = "hamming"
        elif self.blackman_button.isChecked():
            self.window_type = "blackman"
        elif self.hanning_button.isChecked():
            self.window_type = "hanning"
        else:
            self.window_type = "hanning"
        self.process_audio()

    def update_frame_info(self, frame_index):
        """Updates the frame info based on the selected index using frequency-domain features."""
        if not self.is_clips:
            volume, fc, bw, ersb1, ersb2, ersb3, sfm, scf = self.features[frame_index]
            self.frame_info_label.setText(f"Frame {frame_index} Info:")
            frame_info_text = (
                f"Volume: {volume:.4f}\n"
                f"Frequency Centroid: {fc:.2f} Hz\n"
                f"Bandwidth: {bw:.2f} Hz\n"
                f"ERSB1 (0–630 Hz): {ersb1:.4f}\n"
                f"ERSB2 (630–1720 Hz): {ersb2:.4f}\n"
                f"ERSB3 (1720–4400 Hz): {ersb3:.4f}\n"
                f"Spectral Flatness: {sfm:.4f}\n"
                f"Spectral Crest: {scf:.4f}"
            )
            self.frame_info_l2.setText(frame_info_text)
        else:
            if len(self.mini_clip_features) > frame_index:  
                clip_features = self.mini_clip_features[frame_index]
                self.frame_info_label.setText(f"Clip {frame_index} Info:")
                audio_info_text = (
                    f"VSTD: {clip_features['VSTD']:.4f}\n"
                    f"VDR: {clip_features['VDR']:.4f}\n"
                    f"VU: {clip_features['VU']:.4f}\n"
                    f"LSTER: {clip_features['LSTER']:.4f}\n"
                    f"Energy Entropy: {clip_features['Energy_Entropy']:.4f}\n"
                    f"ZSTD: {clip_features['ZSTD']:.4f}\n"
                    f"HZCRR: {clip_features['HZCRR']:.4f}"
                )
                self.frame_info_l2.setText(audio_info_text)
            else:
                print(f"Error: Invalid clip index {frame_index}")

    def audio_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.timer.stop()
            self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        else:
            self.player.play()
            self.timer.start()
            self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))

    def audio_changed(self, state):
        if state == QMediaPlayer.PlaybackState.StoppedState:
            self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        elif state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        else:  # Paused
            self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def update_position(self):
        self.display_time(self.player.position())

    def display_time(self, ms):
        self.player_label.setText(f'{format_time(ms)} / {format_time(int(self.duration))}')

    def show_frames(self):
        self.is_clips = False
        self.plot_audio(self.features)
        self.reset_frame_info()
        self.duration = len(self.audio) / self.sr * 1000 

    def show_clips(self):
        self.is_clips = True
        self.plot_audio(self.features)
        self.reset_frame_info()
        self.duration = len(self.audio) / self.sr * 1000 

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioAnalyzer()
    window.show()
    sys.exit(app.exec())