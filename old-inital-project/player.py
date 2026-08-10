import os
import pygame
import tkinter as tk
from tkinter import filedialog
from threading import Thread
import time

class MusicPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Python Music Player")
        self.playlist = []
        self.current_index = 0
        self.playing = False

        # Pygame mixer
        pygame.mixer.init()

        # GUI
        self.load_button = tk.Button(root, text="Load Folder", command=self.load_music)
        self.load_button.pack()

        self.skip_button = tk.Button(root, text="Skip", command=self.skip_song)
        self.skip_button.pack()

        self.status_label = tk.Label(root, text="No music loaded.")
        self.status_label.pack()

        # Start monitoring thread
        self.monitor_thread = Thread(target=self.monitor_music, daemon=True)
        self.monitor_thread.start()

    def load_music(self):
        folder = filedialog.askdirectory()
        if folder:
            self.playlist = [os.path.join(folder, f) for f in os.listdir(folder)
                             if f.endswith((".mp3", ".wav", ".ogg"))]
            self.playlist.sort()
            self.current_index = 0
            self.status_label.config(text="Loaded {} songs.".format(len(self.playlist)))
            if self.playlist:
                self.play_song()

    def play_song(self):
        if self.current_index < len(self.playlist):
            pygame.mixer.music.load(self.playlist[self.current_index])
            pygame.mixer.music.play()
            self.playing = True
            self.status_label.config(text=f"Playing: {os.path.basename(self.playlist[self.current_index])}")
        else:
            self.status_label.config(text="End of playlist.")
            self.playing = False

    def skip_song(self):
        if self.current_index + 1 < len(self.playlist):
            self.current_index += 1
            self.play_song()

    def monitor_music(self):
        while True:
            if self.playing and not pygame.mixer.music.get_busy():
                self.current_index += 1
                self.play_song()
            time.sleep(1)

if __name__ == "__main__":
    root = tk.Tk()
    player = MusicPlayer(root)
    root.mainloop()
