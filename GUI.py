import tkinter as tk
from tkinter import ttk
import tampilan
import tampilan_2

class AplikasiUtama:
    def __init__(self, root):
        self.root = root
        self.root.geometry("300x300")
        self.label_aplikasi = tk.Label(self.root, text="Aplikasi Klasifikasi Suara\nAdib Dzakwan Fawwaz\n5190411043")
        self.label_aplikasi.pack(pady=10)

        self.tombol_tampilan1 = ttk.Button(self.root, text="Tekan Untuk Memulai Klasifikasi", command=self.buka_tampilan1)
        self.tombol_tampilan1.pack(pady=10)

        self.tombol_tampilan2 = ttk.Button(self.root, text="Tekan Untuk Membuka Fitur Tambahan", command=self.buka_tampilan2)
        self.tombol_tampilan2.pack(pady=10)

    def buka_tampilan1(self):
        self.root.destroy()  # Menutup aplikasi utama
        tampilan.main()

    def buka_tampilan2(self):
        self.root.destroy()  # Menutup aplikasi utama
        tampilan_2.main()

def show_main():
    root = tk.Tk()
    app = AplikasiUtama(root)
    root.mainloop()

if __name__ == "__main__":
    show_main()
