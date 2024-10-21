import GUI
import warnings
from tkinter import ttk, simpledialog
import os
from tkinter import messagebox, filedialog
from tkinter import ttk
from sklearn import preprocessing
from tqdm import tqdm
import shutil
from KNN import KNN
from KNN_manhattan import KNN_manhattan
from KNN_minkowski import KNN_minkowski
import librosa
import librosa.display
import csv
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler




class Page1(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        label = tk.Label(self, text="Proses 1: Ekstraksi Data")
        label.pack(padx=10, pady=10)
        self.button_download = tk.Button(self, text="Download CSV", command=self.download_csv)
        self.button_download.pack( pady=7)
        Data_button=ttk.Button(self, text="Tekan untuk Mengekstrak data Dari Audio", command=self.data_training)
        Data_button.pack(pady=5)
        next_button = ttk.Button(self, text="Langkah Selanjutnya", command=self.next_page)
        next_button.pack(pady=10)


    def data_training(self):
        header = "filename  mfcc1_mean mfcc1_var mfcc2_mean mfcc2_var mfcc3_mean mfcc3_var mfcc4_mean mfcc4_var  mfcc_mean mfcc_var mfcc_mean mfcc_var mfcc_mean mfcc_var mfcc_mean mfcc_var mfcc_mean mfcc_var mfcc_mean mfcc_var label".split()
        file = open('data.csv', 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(header)
        marine_mammals = "BeardedSeal GraySeal HarbourSeal SpottedSeal StellerSeaLion Walrus WeddellSeal".split()

        total_files = sum(len(os.listdir(f"D:/5190411043_Setik/Data/{animal}/")) for animal in marine_mammals)

        progress_window = tk.Toplevel(self)
        progress_window.title("Progress")
        progress_window.geometry("300x100")

        progress_label = tk.Label(progress_window, text="Processing...")
        progress_label.pack(pady=10)

        progress_bar = ttk.Progressbar(progress_window, length=250, mode="determinate", maximum=total_files)
        progress_bar.pack()

        file_count = 0

        for animal in marine_mammals:
            for filename in os.listdir(f"D:/5190411043_Setik/Data/{animal}/"):
                sound_name = f"D:/5190411043_Setik/Data/{animal}/{filename}"
                y, sr = librosa.load(sound_name, mono=True, duration=30)
                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                to_append = f'{filename} '

                for e in mfcc:
                    to_append += f' {np.mean(e)}'
                    # print(np.mean(e))
                to_append += f' {animal}'
                file = open('data.csv', 'a', newline='')

                with file:
                    writer = csv.writer(file)
                    writer.writerow(to_append.split())

                file_count += 1
                progress_bar["value"] = file_count
                progress_bar.update()

        progress_window.destroy()

    def download_csv(self):
        source_file = "data.csv"

        if os.path.exists(source_file):
            destination_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])

            if destination_path:
                shutil.copy(source_file, destination_path)
                success_message = f"File downloaded to:\n{destination_path}"
                messagebox.showinfo("Download Successful", success_message)
            else:
                messagebox.showinfo("Download canceled.",warnings)
        else:
            messagebox.showinfo("File not found.",warnings)


    def next_page(self):
        self.controller.show_page(Page2)


class Page2(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        label = tk.Label(self, text="Proses 2: Menghasilkan Akurasi")
        label.pack(padx=10, pady=10)
        download_button = tk.Button(self, text="Unduh File", command=self.download_files)
        download_button.pack(pady=15)
        akurasi_button=ttk.Button(self, text="Tekan untuk Membuat akurasi", command=self.akurasi)
        akurasi_button.pack(pady=5)
        next_button = ttk.Button(self, text="Langkah Selanjutnya", command=self.next_page)
        next_button.pack(pady=10)


    def akurasi(self):
        global user_input
        try:
            result = simpledialog.askinteger("Input Angka", "Masukkan angka (3, 5, atau 7):")
            if result in (3, 5, 7):
                user_input = result
                messagebox.showinfo("Hasil", f"Angka dari user: {result}")
            else:
                user_input = None
                messagebox.showerror("Kesalahan", "Input harus 3, 5, atau 7.")
        except ValueError:
            user_input = None
            messagebox.showerror("Kesalahan", "Input harus berupa angka.")
        nilai_k = user_input
        df = pd.read_csv('data.csv')
        class_list = df.iloc[:, -1]
        converter = LabelEncoder()
        y = converter.fit_transform(class_list)
        fit = StandardScaler()
        X = fit.fit_transform(preprocessing.normalize(np.array(df.iloc[:, 1:21], dtype=float)))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234, stratify=y)

        # print("Tanpa Library:\n") Euclidean
        cl = KNN(nilai_k)
        cl.fit(X_train, y_train)
        prediksi = cl.predict(X_test)
        akurasi = np.sum(prediksi == y_test) / len(y_test)
        akurasi = "{:.1%}".format(akurasi)
        with open('hasil_prediksi.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Hasil Prediksi', 'Akurasi'])
            for i in range(len(prediksi)):
                hasil_prediksi = converter.inverse_transform([prediksi[i]])[0]
                writer.writerow([hasil_prediksi, akurasi])
        # manhattan
        clf = KNN_manhattan(nilai_k)
        clf.fit(X_train, y_train)
        predik = clf.predict(X_test)
        # print(converter.inverse_transform(predik))
        akurasi2 = np.sum(predik == y_test) / len(y_test)
        akurasi2 = "{:.1%}".format(akurasi2)
        with open('hasil_prediksi_Manhattan.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Hasil Prediksi', 'Akurasi'])
            for i in range(len(predik)):
                hasil_prediksi = converter.inverse_transform([predik[i]])[0]
                writer.writerow([hasil_prediksi, akurasi2])


        # minkowski
        clfj = KNN_minkowski(nilai_k)
        clfj.fit(X_train, y_train)
        predict = clfj.predict(X_test)
        # print(converter.inverse_transform(predict))
        akurasi3 = np.sum(predict == y_test) / len(y_test)
        akurasi3 = "{:.1%}".format(akurasi3)
        with open('hasil_prediksi_Minkowski.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Hasil Prediksi', 'Akurasi'])
            for i in tqdm(range(len(predict)), desc="Writing Progress", unit="row"):
                hasil_prediksi = converter.inverse_transform([predict[i]])[0]
                writer.writerow([hasil_prediksi, akurasi3])
        messagebox.showinfo("Selesai", "Proses penulisan selesai.")

    def download_files(self):
        # List of file names
        file_names = ['hasil_prediksi.csv', 'hasil_prediksi_Manhattan.csv', 'hasil_prediksi_minkowski.csv']

        # Choose download directory
        download_dir = filedialog.askdirectory(title="Pilih Direktori Untuk Unduhan")

        if download_dir:
            for file_name in file_names:
                # Generate full path
                file_path = f"{download_dir}/{file_name}"

                # Simulate downloading by creating empty files
                with open(file_path, 'w') as f:
                    pass

            # Show completion message
            messagebox.showinfo("Selesai", f"File-file telah diunduh ke:\n{download_dir}")


    def next_page(self):
        self.controller.show_page(Page3)


class Page3(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        label = tk.Label(self, text="Proses 3: Prediksi Data")
        label.pack(padx=10, pady=10)
        predict_button = tk.Button(self, text="Prediksi dan Tampilkan Hasil", command=self.klasifikasi)
        predict_button.pack(pady=20)
        next_button = ttk.Button(self, text="Proses 1", command=self.next_page)
        next_button.pack(pady=10)



    def klasifikasi(self):
        global user_input
        try:
            result = simpledialog.askinteger("Input Angka", "Masukkan angka (3, 5, atau 7):")
            if result in (3, 5, 7):
                user_input = result
                messagebox.showinfo("Hasil", f"Angka dari user: {result}")
            else:
                user_input = None
                messagebox.showerror("Kesalahan", "Input harus 3, 5, atau 7.")
        except ValueError:
            user_input = None
            messagebox.showerror("Kesalahan", "Input harus berupa angka.")
        df = pd.read_csv('data.csv')
        nilai_k = user_input
        class_list = df.iloc[:, -1]
        converter = LabelEncoder()
        fit = StandardScaler()
        y = converter.fit_transform(class_list)
        # fit = StandardScaler()
        X = fit.fit_transform(preprocessing.normalize(np.array(df.iloc[:, 1:21], dtype=float)))
        filename = filedialog.askopenfilename(initialdir="/",
                                              title="Select a File",
                                              filetypes=(("Audio files",
                                                          "*.wav*"),
                                                         ("all files",
                                                          "*.*")))
        nama_folder = os.path.dirname(filename)
        d, sr = librosa.load(filename, mono=True, duration=30)
        mfcc = np.array(librosa.feature.mfcc(y=d, sr=sr))
        test = []
        for e in mfcc:
            test.append(np.mean(e, dtype=float))
        features = fit.fit_transform(preprocessing.normalize([test]))

        # print(features)
        cl = KNN(nilai_k)
        cl.fit(X, y)
        prediksi = cl.predict(features)
        # prediksi_knn = StringVar()
        prediksi_knn=(converter.inverse_transform(prediksi))

        cls = KNN_manhattan(nilai_k)
        cls.fit(X, y)
        predict = cls.predict(features)
        # prediksi_manhattan = StringVar()
        prediksi_manhattan=(converter.inverse_transform(predict))

        clf = KNN_minkowski(nilai_k)
        clf.fit(X, y)
        predik = clf.predict(features)
        # prediksi_minkowski = StringVar()
        prediksi_minkowski=(converter.inverse_transform(predik))
        pesan = f"Kelas= {nama_folder} dan File= {filename}\n\nHasil Prediksi:\nEuclidean= {prediksi_knn}, Manhattan{prediksi_manhattan}, dan Minkowski{prediksi_minkowski}"

        # Tampilkan messagebox dengan hasil prediksi
        messagebox.showinfo("Hasil Prediksi", pesan)



    def next_page(self):
        self.controller.show_page(Page1)


# class Page4(tk.Frame):
#     def __init__(self, parent, controller):
#         super().__init__(parent)
#         self.controller = controller
#         label = tk.Label(self, text="Halaman 4")
#         label.pack(padx=10, pady=10)
#
#         back_button = ttk.Button(self, text="Halaman 3", command=self.previous_page)
#         back_button.pack(pady=10)
#
#     def previous_page(self):
#         self.controller.show_page(Page3)


class ScreenManagerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Proses Klasifikasi")
        self.root.geometry("300x300")

        self.screen_manager = ttk.Notebook(self.root)

        self.pages = {
            Page1: "Proses 1",
            Page2: "Proses 2",
            Page3: "Proses 3"
        }

        for page_class, page_name in self.pages.items():
            frame = page_class(self.screen_manager, self)
            self.screen_manager.add(frame, text=page_name)

        self.screen_manager.pack(expand=True, fill="both")

        self.HP_button = ttk.Button(self.root, text="Tekan untuk Kembali ke Homepage", command=self.Kembali)
        self.HP_button.pack(pady=5)
    def show_page(self, page_class):
        for index, (page, _) in enumerate(self.pages.items()):
            if page == page_class:
                self.screen_manager.select(index)
                break

    def Kembali(self):
        self.root.destroy()
        GUI.show_main()


def main():
    root = tk.Tk()
    app = ScreenManagerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
