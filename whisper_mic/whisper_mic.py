import torch
import queue
import speech_recognition as sr
import threading
import numpy as np
import time
import platform


class WhisperMic:
    def __init__(
        self,
        model='base',
        device=('cuda' if torch.cuda.is_available() else 'cpu'),
        language=None,
        energy=300,
        pause=2,
        dynamic_energy=False,
        mic_index=None,
        implementation='whisper',
        hallucinate_threshold=300
    ):
        self.energy = energy
        self.hallucinate_threshold = hallucinate_threshold
        self.pause = pause
        self.dynamic_energy = dynamic_energy
        self.language = language

        self.platform = platform.system()

        self.faster = False
        if (implementation == 'faster_whisper'):
            try:
                from faster_whisper import WhisperModel
                self.audio_model = WhisperModel('large-v3', device='cuda', compute_type='float16')            
                self.faster = True    # Only set the flag if we succesfully imported the library and opened the model.

            except ImportError:
                print('[ERROR] faster_whisper not installed, falling back to whisper')
                self.audio_model = None

        if self.audio_model is None:
            import whisper
            self.audio_model = whisper.load_model(model).to(device)

        self.audio_queue = queue.Queue()
        self.result_queue: 'queue.Queue[str]' = queue.Queue()

        self.break_threads = False
        self.mic_active = False

        self.banned_results = ['',' ','\n',None]

        self.__setup_mic(mic_index)

    def __setup_mic(self, mic_index):
        if mic_index is None:
            print('[INFO] No mic index provided, using default')

        self.source = sr.Microphone(sample_rate=16000, device_index=mic_index)
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = self.energy
        self.recorder.pause_threshold = self.pause
        self.recorder.dynamic_energy_threshold = self.dynamic_energy

        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)

        print('[INFO] Mic setup complete')

    # Whisper takes a Tensor while faster_whisper only wants an NDArray
    def __preprocess(self, data):
        is_audio_loud_enough = self.is_audio_loud_enough(data)
        if self.faster:
            return np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0,is_audio_loud_enough

        else:
            return torch.from_numpy(np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0),is_audio_loud_enough

    def is_audio_loud_enough(self, frame):
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

    # This method takes the recorded audio data, converts it into raw format and stores it in a queue. 
    def __record_load(self, _, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        self.audio_queue.put_nowait(data)

    def __transcribe_forever(self) -> None:
        while True:
            if self.break_threads:
                break

            self.__transcribe()

    def __transcribe(self, data=None) -> None:
        if data is None:
            audio_data = self.__get_all_audio()

        else:
            audio_data = data

        audio_data, is_audio_loud_enough = self.__preprocess(audio_data)

        if is_audio_loud_enough:
            predicted_text = ''
            # faster_whisper returns an iterable object rather than a string
            if self.faster:
                segments, _ = self.audio_model.transcribe(audio_data)
                for segment in segments:
                    predicted_text += segment.text

            else:
                if self.language is not None:
                    result = self.audio_model.transcribe(audio_data, language=self.language, suppress_tokens='')

                else:
                    result = self.audio_model.transcribe(audio_data,suppress_tokens='')

                predicted_text = result['text']

            if predicted_text not in self.banned_results:
                self.result_queue.put_nowait(predicted_text)

    def listen_loop(self, phrase_time_limit=None) -> None:
        for result in self.listen_continuously(phrase_time_limit=phrase_time_limit):
            print(result)

    def listen_continuously(self, phrase_time_limit=None):
        self.recorder.listen_in_background(self.source, self.__record_load, phrase_time_limit=phrase_time_limit)
        print('[INFO] Listening...')
        threading.Thread(target=self.__transcribe_forever, daemon=True).start()

        while True:
            yield self.result_queue.get()
