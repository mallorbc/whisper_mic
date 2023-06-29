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


class WhisperMic:
    def __init__(self,model="base",device=("cuda" if torch.cuda.is_available() else "cpu"),english=False,verbose=False,energy=300,pause=0.8,dynamic_energy=False,save_file=False):
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

        self.mic_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.mic_thread.start()


    def record_audio(self):
        r = sr.Recognizer()
        r.energy_threshold = self.energy
        r.pause_threshold = self.pause
        r.dynamic_energy_threshold = self.dynamic_energy
        
        with sr.Microphone(sample_rate=16000) as source:
            self.mic_active = True
            print("Say something!")
            i = 0
            while True:
                if not self.mic_active:
                    break
                #get and save audio to wav file
                audio = r.listen(source)
                if self.save_file:
                    data = io.BytesIO(audio.get_wav_data())
                    audio_clip = AudioSegment.from_file(data)
                    filename = os.path.join(self.temp_dir, f"temp{i}.wav")
                    audio_clip.export(filename, format="wav")
                    audio_data = filename
                else:
                    torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
                    audio_data = torch_audio
                
                self.audio_queue.put_nowait(audio_data)
                i += 1



    def transcribe_forever(self):
        while True:
            if self.break_threads:
                break
            self.transcribe()


    def transcribe(self,data=None):
        if data is None:
            audio_data = self.audio_queue.get()
        else:
            audio_data = data
        if self.english:
            result = self.audio_model.transcribe(audio_data,language='english')
        else:
            result = self.audio_model.transcribe(audio_data)

        if not self.verbose:
            predicted_text = result["text"]
            self.result_queue.put_nowait(predicted_text)
        else:
            self.result_queue.put_nowait(result)

        if self.save_file:
            os.remove(audio_data)


    def listen_loop(self):
        threading.Thread(target=self.transcribe_forever).start()
        while True:
            print(self.result_queue.get())

    def listen(self):
        audio_data = self.audio_queue.get()
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
    