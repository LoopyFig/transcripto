import os
import csv
import mpv
import time
import queue
import threading
import numpy as np
import sounddevice as sd

from scipy.io.wavfile import write
from pynput import keyboard
from datetime import datetime
from faster_whisper import WhisperModel

import tkinter as tk
from tkinter import filedialog


# =====================
# CONFIG
# =====================

OUTPUT_DIR = "output_annotations"
CSV_PATH = os.path.join(OUTPUT_DIR, "annotations.csv")

REWIND_SECONDS = 5
FORWARD_SECONDS = 5
SAMPLE_RATE = 16000

# =====================
# INIT
# =====================

os.makedirs(OUTPUT_DIR, exist_ok = True)

# Load Whisper model (runs locally)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "faster-whisper-base")
model = WhisperModel(model_path, compute_type="float32")

# =====================
# GUI
# =====================

def launch_gui():

  config = {
    "video_path": None,
    "csv_path": None
  }

  def select_video():
    path = filedialog.askopenfilename(
      filetypes = [("MP4 files", "*.mp4")]
    )
    config["video_path"] = path
    video_label.config(text = path)

  def select_csv():
    path = filedialog.askopenfilename(
      filetypes = [("CSV files", "*.csv")]
    )
    config["csv_path"] = path
    csv_label.config(text = path)

  def start():

    if not config["video_path"] or not config["csv_path"]:
      return

    key_map = load_key_map(config["csv_path"])

    # 🔥 Run session in background thread
    threading.Thread(
      target = run_session,
      args = (config["video_path"], key_map),
      daemon = True
    ).start()

  root = tk.Tk()
  root.title("Transcripto Setup")

  tk.Button(root, text = "Select Video", command = select_video).pack(pady = 5)
  video_label = tk.Label(root, text = "No file selected")
  video_label.pack()

  tk.Button(root, text = "Select Key CSV", command = select_csv).pack(pady = 5)
  csv_label = tk.Label(root, text = "No file selected")
  csv_label.pack()

  tk.Button(root, text = "Start", command = start).pack(pady = 10)

  root.mainloop()

def load_key_map(csv_path):

  key_map = {}

  with open(csv_path, newline = "") as f:
    reader = csv.DictReader(f)
    for row in reader:
      key_map[row["key"]] = row["phrase"]

  return key_map

# =====================
# VIDEO PLAYER
# =====================

class VideoPlayer:

  def __init__(self, path):
    self.player = mpv.MPV(
      input_default_bindings = False,
      input_vo_keyboard = False,
      osc = False,
      keep_open = True,
      force_window=True,
      idle = False
    )

    self.player.play(path)

    self.paused = False
    self.recording = False
    self.working = False
    self.running = True

    # Detect user closing
    @self.player.event_callback("shutdown")
    def on_shutdown(event):
      self.running = False
    # Detect end of video
    @self.player.event_callback("end-file")
    def on_end(event):
      self.player.pause = True

  def get_timestamp(self):
    return self.player.time_pos or 0.0

  def seek(self, seconds):
    self.player.seek(seconds, reference = "relative")

  def toggle(self, state):
    self.paused = state
    self.player.pause = self.paused

  def play(self):
    while True:
      try:
        self.player.wait_for_event(0.1)
      except:
        break

# =====================
# AUDIO RECORDING
# =====================

def record_audio(player):

  print("Recording... press SPACE to stop")

  frames = []

  def callback(indata, frames_count, time_info, status):
    if status:
      print(f"Audio status: {status}")
    frames.append(indata.copy())

  stream = sd.InputStream(
    samplerate = SAMPLE_RATE,
    channels = 1,
    callback = callback
  )

  stream.start()

  while player.recording:
    time.sleep(0.05)

  stream.stop()
  stream.close()

  if not frames:
    return np.zeros((0, 1), dtype = np.float32)

  audio = np.concatenate(frames, axis = 0)

  return audio


# =====================
# TRANSCRIPTION
# =====================

def transcribe_audio(audio_np):
  if audio_np.size == 0:
    return ""

  # Ensure float32
  if audio_np.dtype != np.float32:
    audio_np = audio_np.astype(np.float32)

  # Ensure mono (flatten if needed)
  if len(audio_np.shape) > 1:
    audio_np = audio_np.squeeze()

  # Normalize (important)
  max_val = np.max(np.abs(audio_np))
  if max_val > 0:
    audio_np = audio_np / max_val

  segments, _ = model.transcribe(audio_np)

  text = " ".join(seg.text for seg in segments)

  return text.strip()

# =====================
# Handle Recording and Transcription Workflow
# =====================

def record_and_process(player, phrase, timestamp, saves):

  audio = record_audio(player)

  player.working = False
  player.toggle(False)

  text = transcribe_audio(audio)

  saves.put((timestamp, phrase, text, audio))

  print(f"\n[{timestamp:.2f}s] {phrase}: {text}\n")

# =====================
# KEYBOARD HANDLER
# =====================

def keyboard_handler(player, key_map, saves):

  def on_press(key):

    try:
      k = key.char
    except:
      k = key

    # Play/Pause Mode
    if not player.working:

      # SPACE → play/pause
      if key == keyboard.Key.space:
        player.toggle(not player.paused)

      # SEEK
      elif key == keyboard.Key.left:
        player.seek(-REWIND_SECONDS)

      elif key == keyboard.Key.right:
        player.seek(FORWARD_SECONDS)

      # CUSTOM KEY
      elif isinstance(k, str) and k in key_map:

        player.toggle(True)
        player.working = True
        player.recording = True
        print("Recording started... press SPACE to stop")

        phrase = key_map[k]
        timestamp = player.get_timestamp()

        threading.Thread(
          target = record_and_process,
          args = (player, phrase, timestamp, saves,),
          daemon = True
          ).start()

      elif key == keyboard.Key.esc:
        player.player.command("quit")

    # Working Mode
    else:
      # SPACE → end recording
      if key == keyboard.Key.space:
        player.recording = False

  listener = keyboard.Listener(on_press = on_press)
  listener.start()
  return listener


# =====================
# SAVER THREAD
# =====================

def saver(saves):

  idx = 0

  with open(CSV_PATH, "w", newline = "") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "timestamp", "phrase", "transcription", "audio_file"])

    while True:
      item = saves.get()

      if item is None:
        break

      timestamp, phrase, text, audio = item

      filename = f"{idx}_{int(timestamp)}.wav"
      filepath = os.path.join(OUTPUT_DIR, filename)

      write(filepath, SAMPLE_RATE, audio)

      writer.writerow([idx, timestamp, phrase, text, filename])
      f.flush()

      idx += 1

# =====================
# SESSION LAUNCHER
# =====================

def run_session(video_path, key_map):

  player = VideoPlayer(video_path)

  saves = queue.Queue()

  listener = keyboard_handler(player, key_map, saves)

  saver_thread = threading.Thread(
    target = saver,
    args=(saves,),
    daemon = False
  )
  saver_thread.start()

  player.play()

  saves.put(None)
  saver_thread.join()
  listener.stop()
  try:
    player.player.terminate()
  except:
    pass

# =====================
# MAIN
# =====================

def main():
  launch_gui()

if __name__ == "__main__":
  main()