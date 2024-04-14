import queue
import threading
import time
import typing

import grpc
import numpy as np
import speech_recognition as sr
from faster_whisper import WhisperModel

import faster_whisper_transcription_pb2
import faster_whisper_transcription_pb2_grpc


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
        # The following attributes are only used in local mode
        self.banned_results = None
        self.result_queue: None
        self.faster_whisper_model = None
        self.predicted_language = None

    def listen_and_transcribe(
        self,
        language: typing.Optional[str],
        grpc_address: typing.Optional[str],
        model: typing.Optional[str],
        device: typing.Optional[str],
        precision: typing.Optional[str],
        phrase_time_limit: typing.Optional[int] = None,
    ) -> None:
        self.recorder.listen_in_background(self.source, self._record_load, phrase_time_limit=phrase_time_limit)
        print('[INFO] Listening...')
        if grpc_address is None:
            self._transcribe_locally(language, model, device, precision)

        else:
            self._transcribe_by_grpc(language, grpc_address)

    # This method takes the recorded audio data, converts it into raw format and stores it in a queue.
    def _record_load(self, _, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        self.audio_queue.put_nowait(data)

    def _transcribe_locally(
        self,
        language: typing.Optional[str],
        model: typing.Optional[str],
        device: typing.Optional[str],
        precision: typing.Optional[str],
    ):
        self.banned_results = {'', ' ', '\n', None}
        self.result_queue: 'queue.Queue[str]' = queue.Queue()
        self.faster_whisper_model = WhisperModel(model_size_or_path=model, device=device, compute_type=precision)
        self.predicted_language = language

        threading.Thread(target=self._push_predicted_result, daemon=True).start()

        def pop_predicted_result():
            while True:
                yield self.result_queue.get()

        for predicted_text in pop_predicted_result():
            print(predicted_text)

    def _push_predicted_result(self) -> None:
        while True:
            audio_ndarray: np.ndarray = self._get_audio_ndarray()
            if audio_ndarray is None:
                continue

            segments, _ = self.faster_whisper_model.transcribe(audio_ndarray, language=self.predicted_language)
            predicted_text = ''.join(segment.text for segment in segments)
            if predicted_text not in self.banned_results:
                self.result_queue.put_nowait(predicted_text)

    def _transcribe_by_grpc(self, language: typing.Optional[str], grpc_address: typing.Optional[str]) -> None:
        if language is None:
            language = ''

        with grpc.insecure_channel(grpc_address) as channel:
            stub = faster_whisper_transcription_pb2_grpc.FasterWhisperTranscriptionStub(channel)

            def request_generator():
                while True:
                    audio_ndarray: np.ndarray = self._get_audio_ndarray()
                    if audio_ndarray is None:
                        continue

                    nonlocal language
                    yield faster_whisper_transcription_pb2.AudioData(ndarray_bytes=audio_ndarray.tobytes(), language=language)

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
            return None

        return np.frombuffer(audio_raw_data, np.int16).flatten().astype(np.float32) / 32768.0

    def _is_audio_loud_enough(self, frame):
        audio_frame = np.frombuffer(frame, dtype=np.int16)
        amplitude = np.mean(np.abs(audio_frame))

        return amplitude > self.hallucinate_threshold
