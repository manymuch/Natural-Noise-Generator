import os
import numpy as np
from scipy.io.wavfile import read as wav_r
from scipy.io.wavfile import write as wav_w


def main():
    whole_audio_path = "beach/beach.wav"

    output_dir = "beach_sliced"
    sample_rate = 11025
    output_lenght = 32768
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    whole_audio = wav_r(whole_audio_path)[1]
    print(whole_audio.shape)
    print(whole_audio.dtype)

    one_wave = whole_audio[output_lenght*4:output_lenght*5]
    print(one_wave.shape)
    wav_w("demo.wav",sample_rate,one_wave)





if __name__ == '__main__':
    main()
