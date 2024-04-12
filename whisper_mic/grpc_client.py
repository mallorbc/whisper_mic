import torch
import typing
import click
import pyaudio
from customized_whisper_mic import CustomizedWhisperMic


@click.command()
@click.option('--language', default=None, help='Determine the prior language to detect', type=str)
@click.option('--energy', default=300, help='Energy level for mic to detect', type=int)
@click.option('--dynamic_energy', default=False, is_flag=True, help='Flag to enable dynamic energy')
@click.option('--pause', default=0.8, help='Pause time before entry ends', type=float)
@click.option('--mic_index', default=None, help='Mic index to use', type=int)
@click.option('--list_devices', default=False, help='Flag to list devices', is_flag=True)
@click.option('--hallucinate_threshold',default=400, help='Raise this to reduce hallucinations. Lower this to activate more often.', type=int)
def main(
    energy: int,
    pause: float,
    dynamic_energy: bool,
    mic_index: typing.Optional[int],
    list_devices: bool,
    hallucinate_threshold: int,
    language: str,
    model: str ='base',
    device: str =('cuda' if torch.cuda.is_available() else 'cpu'),
    faster: bool = True,
) -> None:
    if list_devices:
        py_audio_instance = pyaudio.PyAudio()
        for mic_index in range(py_audio_instance.get_device_count()):
            print(f'{mic_index}: {py_audio_instance.get_device_info_by_index(mic_index).get("name")}')

        return

    mic = CustomizedWhisperMic(
        energy=energy,
        pause=pause,
        dynamic_energy=dynamic_energy,
        mic_index=mic_index,
        hallucinate_threshold=hallucinate_threshold,
    )

    try:
        mic.listen_and_transcribe(phrase_time_limit=2)

    except KeyboardInterrupt:
        print('Interrupted by keyboard.')


if __name__ == '__main__':
    main()
