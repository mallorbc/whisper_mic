import click
import torch

from whisper_mic.whisper_mic import WhisperMic

@click.command()
@click.option("--model", default="base", help="Model to use", type=click.Choice(["tiny","base", "small","medium","large"]))
@click.option("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), help="Device to use", type=click.Choice(["cpu","cuda","mps"]))
@click.option("--english", default=False, help="Whether to use English model",is_flag=True, type=bool)
@click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True,type=bool)
@click.option("--energy", default=300, help="Energy level for mic to detect", type=int)
@click.option("--dynamic_energy", default=False,is_flag=True, help="Flag to enable dynamic energy", type=bool)
@click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)
@click.option("--save_file",default=False, help="Flag to save file", is_flag=True,type=bool)
@click.option("--loop", default=False, help="Flag to loop", is_flag=True,type=bool)
def main(model, english,verbose, energy, pause,dynamic_energy,save_file,device,loop):
    mic = WhisperMic(model=model, english=english, verbose=verbose, energy=energy, pause=pause, dynamic_energy=dynamic_energy, save_file=save_file, device=device)
    if not loop:
        result = mic.listen()
        print("You said: " + result)
    else:
        mic.listen_loop()

if __name__ == "__main__":
    main()
