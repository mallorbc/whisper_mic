import os
import platform
import queue
import tempfile
import threading
import time

from collections.abc import AsyncGenerator, Generator
from typing import cast

import numpy as np
import pynput.keyboard
import speech_recognition as sr
import torch

from numpy.typing import NDArray

from whisper_mic.utils import get_logger


# TODO: This is a linux only fix and needs to be testd.  Have one for mac and windows too.
# Define a null error handler for libasound to silence the error message spam
# def py_error_handler(filename, line, function, err, fmt):
#     None

# ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
# c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)


# asound = cdll.LoadLibrary('libasound.so')
# asound.snd_lib_error_set_handler(c_error_handler)
class WhisperMic:
    def __init__(
        self,
        model: str = "base",
        device: str = ("cuda" if torch.cuda.is_available() else "cpu"),
        english: bool = False,
        verbose: bool = False,
        energy: int = 300,
        pause: float = 2,
        dynamic_energy: bool = False,
        save_file: bool = False,
        model_root: str = "~/.cache/whisper",
        mic_index: int | None = None,
        implementation: str = "whisper",
        hallucinate_threshold: int = 300,
    ) -> None:
        self.logger = get_logger("whisper_mic", "info")
        self.energy = energy
        self.hallucinate_threshold = hallucinate_threshold
        self.pause = pause
        self.dynamic_energy = dynamic_energy
        self.save_file = save_file
        self.verbose = verbose
        self.english = english
        self.keyboard = pynput.keyboard.Controller()

        self.platform = platform.system()

        if self.platform == "darwin" and device == "mps":
            self.logger.warning(
                "Using MPS for Mac, this does not work but may in the future",
            )
            device = "mps"
            device = torch.device(device)

        if (model != "large" and model != "large-v2") and self.english:
            model = model + ".en"

        model_root = os.path.expanduser(model_root)

        self.faster = False
        if implementation == "faster_whisper":
            try:
                from faster_whisper import WhisperModel

                self.audio_model = WhisperModel(
                    model,
                    download_root=model_root,
                    device="auto",
                    compute_type="int8",
                )
                self.faster = (
                    True  # Only set the flag if we succesfully imported the library and opened the model.
                )
            except ImportError:
                self.logger.exception(
                    "faster_whisper not installed, falling back to whisper",
                )
                import whisper

                self.audio_model = whisper.load_model(
                    model,
                    download_root=model_root,
                ).to(device)

        else:
            import whisper

            self.audio_model = whisper.load_model(model, download_root=model_root).to(
                device,
            )

        self.temp_dir = tempfile.mkdtemp() if save_file else None

        self.audio_queue: queue.Queue[str] = queue.Queue()
        self.result_queue: queue.Queue[str] = queue.Queue()

        self.break_threads = False
        self.mic_active = False

        self.banned_results = ["", " ", "\n", None]

        if save_file:
            with open("transcribed_text.txt", "w+", encoding="utf-8") as file:
                self.file = file

        self.__setup_mic(mic_index)

    def __setup_mic(self, mic_index: int | None) -> None:
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
    def __preprocess(self, data: bytes | memoryview) -> tuple[torch.Tensor | NDArray[np.float32], bool]:
        is_audio_loud_enough = self.is_audio_loud_enough(data)
        if self.faster:
            return np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0, is_audio_loud_enough

        return torch.from_numpy(
            np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0,
        ), is_audio_loud_enough

    def is_audio_loud_enough(self, frame: bytes | memoryview) -> bool:
        audio_frame = np.frombuffer(frame, dtype=np.int16)
        amplitude = np.mean(np.abs(audio_frame))
        return cast(bool, amplitude > self.hallucinate_threshold)

    def __get_all_audio(self, min_time: float = -1.0) -> bytes:
        got_audio = False
        time_start = time.time()
        while not got_audio or time.time() - time_start < min_time:
            while not self.audio_queue.empty():
                data_audio = cast(bytes, self.audio_queue.get())
                got_audio = True

        data = sr.AudioData(data_audio, 16000, 2)
        return cast(bytes, data.get_raw_data())

    # Handles the task of getting the audio input via microphone. This method has been used for listen() method
    def __listen_handler(self, timeout: int | None, phrase_time_limit: int | None) -> None:
        try:
            with self.source as microphone:
                audio = self.recorder.listen(
                    source=microphone,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit,
                )
            self.__record_load(0, audio)
            audio_data = self.__get_all_audio()
            self.__transcribe(data=audio_data)
        except sr.WaitTimeoutError:
            self.result_queue.put_nowait(
                "Timeout: No speech detected within the specified time.",
            )
        except sr.UnknownValueError:
            self.result_queue.put_nowait(
                "Speech recognition could not understand audio.",
            )

    # This method is similar to the __listen_handler() method but it has the added ability for recording the audio for a specified duration of time
    def __record_handler(self, duration: int | None, offset: int | None) -> None:
        with self.source as microphone:
            audio = self.recorder.record(
                source=microphone,
                duration=duration,
                offset=offset,
            )

        self.__record_load(0, audio)
        audio_data = self.__get_all_audio()
        self.__transcribe(data=audio_data)

    # This method takes the recorded audio data, converts it into raw format and stores it in a queue.
    def __record_load(self, _: int, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        self.audio_queue.put_nowait(data)

    def __transcribe_forever(self) -> None:
        while True:
            if self.break_threads:
                break
            self.__transcribe()

    # TODO: Allow realtime transcription
    # def __transcribe(self, data: bytes | memoryview | None = None, realtime: bool = False) -> None:
    def __transcribe(self, data: bytes | memoryview | None = None) -> None:
        audio_data = self.__get_all_audio() if data is None else data
        audio_data, is_audio_loud_enough = self.__preprocess(audio_data)

        if is_audio_loud_enough:
            predicted_text = ""
            # faster_whisper returns an iterable object rather than a string
            if self.faster:
                segments, _ = self.audio_model.transcribe(audio_data)
                for segment in segments:
                    predicted_text += segment.text
            else:
                if self.english:
                    result = self.audio_model.transcribe(
                        audio_data,
                        language="english",
                        suppress_tokens="",
                    )
                else:
                    result = self.audio_model.transcribe(audio_data, suppress_tokens="")
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

    async def listen_loop_async(
        self,
        dictate: bool = False,
        phrase_time_limit: int | None = None,
    ) -> AsyncGenerator[str, None]:
        for result in self.listen_continuously(phrase_time_limit=phrase_time_limit):
            if dictate:
                self.keyboard.type(result)
            else:
                yield result

    def listen_loop(self, dictate: bool = False, phrase_time_limit: int | None = None) -> None:
        for result in self.listen_continuously(phrase_time_limit=phrase_time_limit):
            if dictate:
                self.keyboard.type(result)
            else:
                self.logger.info(result)

    def listen_continuously(self, phrase_time_limit: int | None = None) -> Generator[str, str, None]:
        self.recorder.listen_in_background(
            self.source,
            self.__record_load,
            phrase_time_limit=phrase_time_limit,
        )
        self.logger.info("Listening...")
        threading.Thread(target=self.__transcribe_forever, daemon=True).start()

        while True:
            yield self.result_queue.get()

    def listen(self, timeout: int | None = None, phrase_time_limit: int | None = None) -> str:
        self.logger.info("Listening...")
        self.__listen_handler(timeout, phrase_time_limit)
        while True:
            if not self.result_queue.empty():
                return self.result_queue.get()

    # This method is similar to the listen() method, but it has the ability to listen for a specified duration, mentioned in the "duration" parameter.
    def record(self, duration: int | None = None, offset: int | None = None) -> str:
        self.logger.info("Listening...")
        self.__record_handler(duration, offset)
        while True:
            if not self.result_queue.empty():
                return self.result_queue.get()

    def toggle_microphone(self) -> None:
        # TODO: make this work
        self.mic_active = not self.mic_active
        if self.mic_active:
            self.logger.debug("Mic on")
        else:
            self.logger.debug("turning off mic")
            # self.mic_thread.join()  # FIXME: doesn't exist
            self.logger.debug("Mic off")
