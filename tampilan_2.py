import numpy as np
from tkinter import ttk
import tkinter as tk
from tkinter import filedialog, Label, StringVar, simpledialog, messagebox
from playsound import playsound
import librosa
import librosa.display
import matplotlib.pyplot as plt
import GUI


class ScreenManagerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tambahan Fitur")
        self.root.geometry("300x300")
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Helvetica", 12))
        self.adib_label = tk.Label(self.root, text="Adib Dzakwan Fawwaz ", font=("Helvetica", 9))
        self.adib_label.pack(pady=10)
        self.adib_label = tk.Label(self.root, text="Fitur Tambahan ", font=("Helvetica", 9))
        self.adib_label.pack(pady=10)
        self.Spektrum_button = ttk.Button(self.root, text="Tekan untuk Melihat sinyal Audio", command=self.spektrum)
        self.Spektrum_button.pack(pady=5)
        self.Play_button = ttk.Button(self.root, text="Tekan untuk Mendengar Audio", command=self.putarSuara)
        self.Play_button.pack(pady=5)

        self.HP_button = ttk.Button(self.root, text="Tekan untuk Kembali ke Homepage", command=self.Kembali)
        self.HP_button.pack(pady=5)
    def spektrum(self):
        filename = filedialog.askopenfilename(initialdir="/",
                                              title="Select a File",
                                              filetypes=(("Audio files",
                                                          "*.wav*"),
                                                         ("all files",
                                                          "*.*")))
        filename = filename  # Example audio file from librosa
        # Load the audio data and sample rate using librosa
        data, sr = librosa.load(filename)
        # Apply preemphasis
        preemphasized_data = np.append(data[0], data[1:] - 0.97 * data[:-1])
        # Compute MFCCs
        mfccs_original = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
        mfccs_preemphasized = librosa.feature.mfcc(y=preemphasized_data, sr=sr, n_mfcc=40)
        # Plotting
        plt.figure(figsize=(12, 8))
        # Waveform plot
        plt.subplot(2, 2, 1)
        plt.plot(data)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title("Original Waveform")
        # Preemphasized waveform plot
        plt.subplot(2, 2, 2)
        plt.plot(preemphasized_data)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title("Preemphasized Waveform")
        # MFCC plot for original waveform
        plt.subplot(2, 2, 3)
        librosa.display.specshow(mfccs_original, x_axis="time")
        plt.colorbar()
        plt.xlabel("Time")
        plt.ylabel("MFCC Coefficient")
        plt.title("MFCC - Original Waveform")
        # MFCC plot for preemphasized waveform
        plt.subplot(2, 2, 4)
        librosa.display.specshow(mfccs_preemphasized, x_axis="time")
        plt.colorbar()
        plt.xlabel("Time")
        plt.ylabel("MFCC Coefficient")
        plt.title("MFCC - Preemphasized Waveform")
        plt.tight_layout()
        plt.show()

    # Function to play sound
    def putarSuara(self):
        filename = filedialog.askopenfilename(
            initialdir="/",
            title="Select an Audio File",
            filetypes=(("Audio files", "*.wav"), ("All files", "*.*"))
        )
        if filename:
            playsound(filename)
    def Kembali(self):
        self.root.destroy()
        GUI.show_main()






def main():
    root = tk.Tk()
    app = ScreenManagerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
