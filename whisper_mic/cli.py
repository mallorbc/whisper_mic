#!/usr/bin/env python3

import click
import torch
import pyaudio
from typing import Optional

from whisper_mic import WhisperMic

@click.command()
@click.option('--model', default='base', help='Model to use', type=click.Choice(['tiny','base', 'small','medium','large','large-v2','large-v3']))
@click.option('--device', default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device to use', type=click.Choice(['cpu','cuda','mps']))
@click.option('--language', default=None, help='Determine the prior language to detect', type=str)
@click.option('--energy', default=300, help='Energy level for mic to detect', type=int)
@click.option('--dynamic_energy', default=False, is_flag=True, help='Flag to enable dynamic energy')
@click.option('--pause', default=0.8, help='Pause time before entry ends', type=float)
@click.option('--mic_index', default=None, help='Mic index to use', type=int)
@click.option('--list_devices', default=False, help='Flag to list devices', is_flag=True)
@click.option('--faster', default=False, help='Use faster_whisper implementation', is_flag=True)
@click.option('--hallucinate_threshold',default=400, help='Raise this to reduce hallucinations.  Lower this to activate more often.', type=int)
def main(
    model: str,
    language: str,
    energy: int,
    pause: float,
    dynamic_energy: bool,
    device: str,
    mic_index: Optional[int],
    list_devices: bool,
    faster: bool,
    hallucinate_threshold: int,
) -> None:
    if list_devices:
        py_audio_instance = pyaudio.PyAudio()
        for mic_index in range(py_audio_instance.get_device_count()):
            print(f'{mic_index}: {py_audio_instance.get_device_info_by_index(mic_index).get("name")}')

        return

    mic = WhisperMic(
        model=model,
        language=language,
        energy=energy,
        pause=pause,
        dynamic_energy=dynamic_energy,
        device=device,
        mic_index=mic_index,
        implementation=('faster_whisper' if faster else 'whisper'),
        hallucinate_threshold=hallucinate_threshold,
    )

    try:
        mic.listen_loop(phrase_time_limit=2)

    except KeyboardInterrupt:
        print('Interrupted by keyboard.')


if __name__ == '__main__':
    main()
