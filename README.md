# Whisper Mic
This repo is based on the work done [here](https://github.com/openai/whisper) by OpenAI.  This repo allows you use use a mic as demo. This repo copies some of the README from original project.

## Video Tutorial

See the video tutorial for this repo [here](https://www.youtube.com/watch?v=nwPaRSlDSaY)

### Professional Assistance

If are in need of paid professional help, that is available through this [email](mailto:blakecmallory@gmail.com)

## Setup

1. Create a venv of your choice.
2. Run ```pip install -r requirements.txt```


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

You can use the model with a microphone using the ```mic.py``` program.  Use ```-h``` to see flag options.

Some of the more important flags are the ```--model``` and ```--english``` flags.

## Troubleshooting

If you are having issues with the ```mic.py``` not running try the following:
```
sudo apt install portaudio19-dev python3-pyaudio
```


## License

The code and the model weights of Whisper are released under the MIT License. See their repo for more information.

The code under this repo is under the AGPL license.  See [LICENSE](LICENSE) for further details.

## Thanks
Until recently, access to high performing speech to text models was only available through paid serviecs.  With this release, I am excited for the many applications that will come.
