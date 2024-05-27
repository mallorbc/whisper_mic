import click
import speech_recognition as sr
import torch

from whisper_mic.utils import get_logger
from whisper_mic.whisper_mic import WhisperMic


@click.command()
@click.option(
    "--model",
    default="base",
    help="Model to use",
    type=click.Choice(
        ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
    ),
)
@click.option(
    "--device",
    default=("cuda" if torch.cuda.is_available() else "cpu"),
    help="Device to use",
    type=click.Choice(["cpu", "cuda", "mps"]),
)
@click.option(
    "--english",
    default=False,
    help="Whether to use English model",
    is_flag=True,
    type=bool,
)
@click.option(
    "--verbose",
    default=False,
    help="Whether to print verbose output",
    is_flag=True,
    type=bool,
)
@click.option(
    "--energy",
    default=300,
    help="Energy level for mic to detect",
    type=int,
)
@click.option(
    "--dynamic_energy",
    default=False,
    is_flag=True,
    help="Flag to enable dynamic energy",
    type=bool,
)
@click.option(
    "--pause",
    default=0.8,
    help="Pause time before entry ends",
    type=float,
)
@click.option(
    "--save_file",
    default=False,
    help="Flag to save file",
    is_flag=True,
    type=bool,
)
@click.option(
    "--loop",
    default=False,
    help="Flag to loop",
    is_flag=True,
    type=bool,
)
@click.option(
    "--dictate",
    default=False,
    help="Flag to dictate (implies loop)",
    is_flag=True,
    type=bool,
)
@click.option(
    "--mic_index",
    default=None,
    help="Mic index to use",
    type=int,
)
@click.option(
    "--list_devices",
    default=False,
    help="Flag to list devices",
    is_flag=True,
    type=bool,
)
@click.option(
    "--faster",
    default=False,
    help="Use faster_whisper implementation",
    is_flag=True,
    type=bool,
)
@click.option(
    "--hallucinate_threshold",
    default=400,
    help="Raise this to reduce hallucinations.  Lower this to activate more often.",
    is_flag=True,
    type=int,
)
def main(
    model: str,
    english: bool,
    verbose: bool,
    energy: int,
    pause: float,
    dynamic_energy: bool,
    save_file: bool,
    device: str,
    loop: bool,
    dictate: bool,
    mic_index: int | None,
    list_devices: bool,
    faster: bool,
    hallucinate_threshold: int,
) -> None:
    logger = get_logger("cli", "debug")

    if list_devices:
        logger.debug("Possible devices: ", sr.Microphone.list_microphone_names())
        return
    mic = WhisperMic(
        model=model,
        english=english,
        verbose=verbose,
        energy=energy,
        pause=pause,
        dynamic_energy=dynamic_energy,
        save_file=save_file,
        device=device,
        mic_index=mic_index,
        implementation=("faster_whisper" if faster else "whisper"),
        hallucinate_threshold=hallucinate_threshold,
    )

    if not loop:
        try:
            result = mic.listen()
            logger.info(f"You said: {result}")
        except KeyboardInterrupt:
            logger.info("Operation interrupted successfully")
        finally:
            if save_file:
                mic.file.close()
    else:
        try:
            mic.listen_loop(dictate=dictate, phrase_time_limit=3)
        except KeyboardInterrupt:
            logger.info("Operation interrupted successfully")
        finally:
            if save_file:
                mic.file.close()


if __name__ == "__main__":
    main()
