import typing

import click
import pyaudio

from customized_whisper_mic import CustomizedWhisperMic


# pylint: disable=no-value-for-parameter
@click.command()
@click.option('--language', default=None, help='Determine the prior language to detect', type=str)
@click.option('--energy', default=300, help='Energy level for mic to detect', type=int)
@click.option('--dynamic_energy', default=False, is_flag=True, help='Flag to enable dynamic energy')
@click.option('--pause', default=0.8, help='Pause time before entry ends', type=float)
@click.option('--mic_index', default=None, help='Mic index to use', type=int)
@click.option('--list_devices', default=False, help='Flag to list devices', is_flag=True)
@click.option('--hallucinate_threshold',default=400, help='Raise this to reduce hallucinations. Lower this to activate more often.', type=int)
@click.option('--grpc_address', default=None, help='<ip_addr>:<port> or run locally by default', type=str)
@click.option('--model', default=None, help='Model (`distil-large-v3` if en only)', type=click.Choice(['medium', 'large-v3', 'distil-large-v3']))
@click.option('--device', default=None, help='Device', type=click.Choice(['cpu', 'cuda']))
@click.option('--precision', default=None, help='Precision level', type=click.Choice(['int8', 'float16']))
def main(
    language: str,
    energy: int,
    pause: float,
    dynamic_energy: bool,
    mic_index: typing.Optional[int],
    list_devices: bool,
    hallucinate_threshold: int,
    grpc_address: typing.Optional[str],
    model: str,
    device: str,
    precision: str,
) -> None:
    if list_devices:
        py_audio_instance = pyaudio.PyAudio()
        for hardware_mic_index in range(py_audio_instance.get_device_count()):
            print(f'{hardware_mic_index}: {py_audio_instance.get_device_info_by_index(hardware_mic_index).get("name")}')

        return

    mic = CustomizedWhisperMic(
        energy=energy,
        pause=pause,
        dynamic_energy=dynamic_energy,
        mic_index=mic_index,
        hallucinate_threshold=hallucinate_threshold,
    )

    try:
        mic.listen_and_transcribe(
            language=language,
            grpc_address=grpc_address,
            model=model,
            device=device,
            precision=precision,
            phrase_time_limit=2,
        )

    except KeyboardInterrupt:
        print('Interrupted by keyboard.')


if __name__ == '__main__':
    main()
