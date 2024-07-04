import torch
import queue
import speech_recognition as sr
import threading
import numpy as np
import os
import time
import tempfile
import platform
import pynput.keyboard
from typing import Optional
# from ctypes import *

from whisper_mic.utils import get_logger

#TODO: This is a linux only fix and needs to be testd.  Have one for mac and windows too.
# Define a null error handler for libasound to silence the error message spam
# def py_error_handler(filename, line, function, err, fmt):
#     None

# ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
# c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

# asound = cdll.LoadLibrary('libasound.so')
# asound.snd_lib_error_set_handler(c_error_handler)
class WhisperMic:
    def __init__(self,model="base",device=("cuda" if torch.cuda.is_available() else "cpu"),english=False,verbose=False,energy=300,pause=2,dynamic_energy=False,save_file=False, model_root="~/.cache/whisper",mic_index=None,implementation="whisper",hallucinate_threshold=300):

        self.logger = get_logger("whisper_mic", "info")
        self.energy = energy
        self.hallucinate_threshold = hallucinate_threshold
        self.pause = pause
        self.dynamic_energy = dynamic_energy
        self.save_file = save_file
        self.verbose = verbose
        self.english = english
        self.keyboard = pynput.keyboard.Controller()

        self.platform = platform.system().lower()
        if self.platform == "darwin":
            if device == "cuda" or device == "mps":
                self.logger.warning("CUDA is not supported on MacOS and mps does not work. Using CPU instead.")
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if (model != "large" and model != "large-v2" and model!= "large-v3") and self.english:
            model = model + ".en"

        model_root = os.path.expanduser(model_root)

        self.faster = False
        if (implementation == "faster_whisper"):
            try:
                from faster_whisper import WhisperModel
                self.audio_model = WhisperModel(model, download_root=model_root, device="auto", compute_type="int8")            
                self.faster = True    # Only set the flag if we succesfully imported the library and opened the model.
            except ImportError:
                self.logger.error("faster_whisper not installed, falling back to whisper")
                self.logger.info("To install faster_whisper, run 'pip install faster_whisper'")
                import whisper
                self.audio_model = whisper.load_model(model, download_root=model_root).to(device)

        else:
            import whisper
            self.audio_model = whisper.load_model(model, download_root=model_root).to(device)
        
        self.temp_dir = tempfile.mkdtemp() if save_file else None

        self.audio_queue = queue.Queue()
        self.result_queue: "queue.Queue[str]" = queue.Queue()
        
        self.break_threads = False
        self.mic_active = False

        self.banned_results = [""," ","\n",None]

        if save_file:
            self.file = open("transcribed_text.txt", "w+", encoding="utf-8")

        self.__setup_mic(mic_index)


    def __setup_mic(self, mic_index):
        if mic_index is None:
            self.logger.info("No mic index provided, using default")
        self.source = sr.Microphone(sample_rate=16000, device_index=mic_index)

        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = self.energy
        self.recorder.pause_threshold = self.pause
        self.recorder.dynamic_energy_threshold = self.dynamic_energy

        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)

        self.logger.info("Mic setup complete")

    # Whisper takes a Tensor while faster_whisper only wants an NDArray
    def __preprocess(self, data):
        is_audio_loud_enough = self.is_audio_loud_enough(data)
        if self.faster:
            return np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0,is_audio_loud_enough
        else:
            return torch.from_numpy(np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0),is_audio_loud_enough
        
    def is_audio_loud_enough(self, frame) -> bool:
        audio_frame = np.frombuffer(frame, dtype=np.int16)
        amplitude = np.mean(np.abs(audio_frame))
        return amplitude > self.hallucinate_threshold

    
    def __get_all_audio(self, min_time: float = -1.):
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
    

    # Handles the task of getting the audio input via microphone. This method has been used for listen() method
    def __listen_handler(self, timeout, phrase_time_limit):
        try:
            with self.source as microphone:
                audio = self.recorder.listen(source=microphone, timeout=timeout, phrase_time_limit=phrase_time_limit)
            self.__record_load(0, audio)
            audio_data = self.__get_all_audio()
            self.__transcribe(data=audio_data)
        except sr.WaitTimeoutError:
            self.result_queue.put_nowait("Timeout: No speech detected within the specified time.")
        except sr.UnknownValueError:
            self.result_queue.put_nowait("Speech recognition could not understand audio.")


    # This method is similar to the __listen_handler() method but it has the added ability for recording the audio for a specified duration of time
    def __record_handler(self, duration=2, offset=None):
        with self.source as microphone:
            audio = self.recorder.record(source=microphone, duration=duration, offset=offset)
        
        self.__record_load(0, audio)
        audio_data = self.__get_all_audio()
        self.__transcribe(data=audio_data)


    # This method takes the recorded audio data, converts it into raw format and stores it in a queue. 
    def __record_load(self,_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        self.audio_queue.put_nowait(data)


    def __transcribe_forever(self) -> None:
        while True:
            if self.break_threads:
                break
            self.__transcribe()


    def __transcribe(self,data=None, realtime: bool = False) -> None:
        if data is None:
            audio_data = self.__get_all_audio()
        else:
            audio_data = data
        audio_data,is_audio_loud_enough = self.__preprocess(audio_data)

        if is_audio_loud_enough:
            predicted_text = ''
            # faster_whisper returns an iterable object rather than a string
            if self.faster:
                segments, info = self.audio_model.transcribe(audio_data)
                for segment in segments:
                    predicted_text += segment.text
            else:
                if self.english:
                    result = self.audio_model.transcribe(audio_data,language='english',suppress_tokens="")
                else:
                    result = self.audio_model.transcribe(audio_data,suppress_tokens="")
                predicted_text = result["text"]

            if not self.verbose:
                if predicted_text not in self.banned_results:
                    self.result_queue.put_nowait(predicted_text)
            else:
                if predicted_text not in self.banned_results:
                    self.result_queue.put_nowait(result)


            if self.save_file:
                # os.remove(audio_data)
                self.file.write(predicted_text)
        else:
            # If the audio is not loud enough, we put None in the queue to indicate that we need to listen again or return None
            self.result_queue.put_nowait(None)

    async def listen_loop_async(self, dictate: bool = False, phrase_time_limit=None) -> Optional[str]:
        for result in self.listen_continuously(phrase_time_limit=phrase_time_limit):
            if dictate:
                self.keyboard.type(result)
            else:
                yield result


    def listen_loop(self, dictate: bool = False, phrase_time_limit=None) -> None:
        for result in self.listen_continuously(phrase_time_limit=phrase_time_limit):
            if result is not None:
                if dictate:
                    self.keyboard.type(result)
                else:
                    print(result)


    def listen_continuously(self, phrase_time_limit=None):
        self.recorder.listen_in_background(self.source, self.__record_load, phrase_time_limit=phrase_time_limit)
        self.logger.info("Listening...")
        threading.Thread(target=self.__transcribe_forever, daemon=True).start()

        while True:
            yield self.result_queue.get()

            
    def listen(self, timeout = None, phrase_time_limit=None,try_again=True):
        self.logger.info("Listening...")
        self.__listen_handler(timeout, phrase_time_limit)
        while True:
            if not self.result_queue.empty():
                result = self.result_queue.get()
                if result is None and try_again:
                    self.logger.info("Too quiet, listening again...")
                    result = self.listen(timeout=timeout, phrase_time_limit=phrase_time_limit,try_again=True)
                    return result
                else:
                    return result


    # This method is similar to the listen() method, but it has the ability to listen for a specified duration, mentioned in the "duration" parameter.
    def record(self, duration=2, offset=None,try_again=True):
        self.logger.info("Listening...")
        if duration is None:
            self.logger.warning("Duration not provided, may hang indefinitely.")
        self.__record_handler(duration, offset)
        while True:
            if not self.result_queue.empty():
                result = self.result_queue.get()
                if result is None and try_again:
                    self.logger.info("Too quiet, listening again...")
                    result = self.record(duration=duration, offset=offset,try_again=True)
                    return result
                else:
                    return result


    def toggle_microphone(self) -> None:
        #TO DO: make this work
        self.mic_active = not self.mic_active
        if self.mic_active:
            print("Mic on")
        else:
            print("turning off mic")
            self.mic_thread.join()
            print("Mic off")
