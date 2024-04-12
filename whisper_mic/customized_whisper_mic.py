import torch
import queue
import typing
import speech_recognition as sr
import threading
import numpy as np
import time
import whisper_mic_service_pb2
import whisper_mic_service_pb2_grpc
import grpc


class CustomizedWhisperMic:
    def __init__(
        self,
        energy,
        pause,
        dynamic_energy,
        mic_index,
        hallucinate_threshold,
    ):
        if mic_index is None:
            print('[INFO] No mic index provided, using default')

        self.source = sr.Microphone(sample_rate=16000, device_index=mic_index)
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = energy
        self.recorder.pause_threshold = pause
        self.recorder.dynamic_energy_threshold = dynamic_energy
        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)

        print('[INFO] Mic setup complete')

        self.hallucinate_threshold = hallucinate_threshold
        self.audio_queue = queue.Queue()

    def listen_and_transcribe(self, phrase_time_limit=None) -> None:
        self.recorder.listen_in_background(self.source, self._record_load, phrase_time_limit=phrase_time_limit)
        print('[INFO] Listening...')
        self._transcribe_by_grpc()

    # This method takes the recorded audio data, converts it into raw format and stores it in a queue. 
    def _record_load(self, _, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        self.audio_queue.put_nowait(data)

    def _transcribe_by_grpc(self) -> None:
        with grpc.insecure_channel("localhost:50051") as channel:
            stub = whisper_mic_service_pb2_grpc.WhisperMicStub(channel)

            def request_generator():
                while True:
                    audio_data: np.ndarray = self._get_audio_ndarray()
                    if audio_data is None:
                        continue

                    yield whisper_mic_service_pb2.AudioData(ndarray_bytes=audio_data.tobytes())

            response_iterator = stub.StreamData(request_generator())
            for response in response_iterator:
                print(response.prediction)

    def _get_audio_ndarray(self) -> typing.Optional[np.ndarray]:
        return self._get_audio_ndarray_from_raw_data(self._get_audio_raw_data())

    def _get_audio_raw_data(self, min_time: float = -1.):
        audio_bytes = bytes()
        got_audio = False
        time_start = time.time()
        while not got_audio or time.time() - time_start < min_time:
            while not self.audio_queue.empty():
                audio_bytes += self.audio_queue.get_nowait()
                got_audio = True

        return sr.AudioData(audio_bytes, sample_rate=16000, sample_width=2).get_raw_data()

    def _get_audio_ndarray_from_raw_data(self, audio_raw_data) -> typing.Optional[np.ndarray]:
        if not self._is_audio_loud_enough(audio_raw_data):
            return

        return np.frombuffer(audio_raw_data, np.int16).flatten().astype(np.float32) / 32768.0

    def _is_audio_loud_enough(self, frame):
        audio_frame = np.frombuffer(frame, dtype=np.int16)
        amplitude = np.mean(np.abs(audio_frame))

        return amplitude > self.hallucinate_threshold
