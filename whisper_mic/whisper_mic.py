import torch
import whisper
import queue
import speech_recognition as sr
import threading
import io
import numpy as np
from pydub import AudioSegment
import os
import tempfile
import time

from whisper_mic.utils import get_logger


class WhisperMic:
    def __init__(self,model="base",device=("cuda" if torch.cuda.is_available() else "cpu"),english=False,verbose=False,energy=300,pause=0.8,dynamic_energy=False,save_file=False):
        self.logger = get_logger("whisper_mic", "info")
        self.energy = energy
        self.pause = pause
        self.dynamic_energy = dynamic_energy
        self.save_file = save_file
        self.verbose = verbose
        self.english = english

        if model != "large" and self.english:
            model = model + ".en"

        self.audio_model = whisper.load_model(model).to(device)
        self.temp_dir = tempfile.mkdtemp() if save_file else None

        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()

        self.break_threads = False
        self.mic_active = False

        self.banned_results = [""," ","\n",None]

        self.setup_mic()


    def setup_mic(self):
        self.source = sr.Microphone(sample_rate=16000)

        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = self.energy
        self.recorder.pause_threshold = self.pause
        self.recorder.dynamic_energy_threshold = self.dynamic_energy

        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)

        self.recorder.listen_in_background(self.source, self.record_callback, phrase_time_limit=2)
        self.logger.info("Mic setup complete, you can now talk")


    def preprocess(self, data):
        return torch.from_numpy(np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0)
    
    def get_all_audio(self,min_time=-1):
        audio = bytes()
        got_audio = False
        time_start = time.time()
        while not got_audio or time.time() - time_start < min_time:
            while not self.audio_queue.empty():
                audio += self.audio_queue.get()
                got_audio = True

        data = sr.AudioData(audio,16000,2)
        data = data.get_raw_data()
        return data


    def record_callback(self,_, audio:sr.AudioData) -> None:
        data = audio.get_raw_data()
        self.audio_queue.put_nowait(data)


    def transcribe_forever(self):
        while True:
            if self.break_threads:
                break
            self.transcribe()


    def transcribe(self,data=None,realtime=False):
        if data is None:
            # audio_data = self.audio_queue.get()
            audio_data = self.get_all_audio()
        else:
            audio_data = data
        audio_data = self.preprocess(audio_data)
        if self.english:
            result = self.audio_model.transcribe(audio_data,language='english')
        else:
            result = self.audio_model.transcribe(audio_data)

        predicted_text = result["text"]
        if not self.verbose:
            if predicted_text not in self.banned_results:
                self.result_queue.put_nowait(predicted_text)
        else:
            if predicted_text not in self.banned_results:
                self.result_queue.put_nowait(result)

        if self.save_file:
            os.remove(audio_data)


    def listen_loop(self):
        threading.Thread(target=self.transcribe_forever).start()
        while True:
            result = self.result_queue.get()
            print(result)
                

    def listen(self,timout=3):
        audio_data = self.get_all_audio(timout)
        self.transcribe(data=audio_data)
        while True:
            if not self.result_queue.empty():
                return self.result_queue.get()
            
    def toggle_microphone(self):
        #TO DO: make this work
        self.mic_active = not self.mic_active
        if self.mic_active:
            print("Mic on")
        else:
            print("turning off mic")
            self.mic_thread.join()
            print("Mic off")
    