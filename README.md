# Whisper Mic
This repo is based on the work done [here](https://github.com/openai/whisper) by OpenAI.  This repo allows you use use a mic as demo. This repo copies some of the README from the original project.

## Video Tutorial

The latest video tutorial for this repo can be seen [here](https://youtu.be/S58MGCU7Wgg)

An older video tutorial for this repo can be seen [here](https://www.youtube.com/watch?v=nwPaRSlDSaY)

### Professional Assistance

If are in need of paid professional help, that is available through this [email](mailto:blakecmallory@gmail.com)

## Setup

Now a pip package!

1. Create a venv of your choice.
2. Run ```pip install whisper-mic```

## Available models and languages

There are five model sizes, four with English-only versions, offering speed and accuracy tradeoffs. Below are the names of the available models and their approximate memory requirements and relative speed. 


|  Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
|:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
|  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~32x      |
|  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~16x      |
| small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~6x       |
| medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
| large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |

For English-only applications, the `.en` models tend to perform better, especially for the `tiny.en` and `base.en` models. We observed that the difference becomes less significant for the `small.en` and `medium.en` models.

## Microphone Demo

You can use the model with a microphone using the ```whisper_mic``` program.  Use ```-h``` to see flag options.

Some of the more important flags are the ```--model``` and ```--english``` flags.

## Transcribing To A File

Using the command: ```whisper_mic --loop --dictate``` will type the words you say on your active cursor.

## Usage In Other Projects

You can use this code in other projects rather than just use it for a demo.  You can do this with the ```listen``` method.

```python
from whisper_mic import WhisperMic

mic = WhisperMic()
result = mic.listen()
print(result)
```

Check out what the possible arguments are by looking at the ```cli.py``` file

## Troubleshooting

If you are having issues, try the following:
```
sudo apt install portaudio19-dev python3-pyaudio
```

## Contributing

Some ideas that you can add are:
1. Supporting different implementations of Whisper
2. Adding additional optional functionality.
3. Add tests

## License

The model weights of Whisper are released under the MIT License. See their repo for more information.

This code under this repo is under the MIT license.  See [LICENSE](LICENSE) for further details.

## Thanks
Until recently, access to high performing speech to text models was only available through paid serviecs.  With this release, I am excited for the many applications that will come.
