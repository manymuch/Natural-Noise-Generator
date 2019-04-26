import os
import numpy as np
from scipy.io.wavfile import read as wav_r
from scipy.io.wavfile import write as wav_w


def main():
    whole_audio_path = "beach/beach.wav"

    output_dir = "beach_sliced"
    sample_rate = 11025
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    whole_audio = wav_r(whole_audio_path)[1]
    birds = wav_r("birds/birds11025.wav")[1]
    print(whole_audio.shape)
    print(whole_audio.dtype)
    print(birds.shape)
    print(birds.dtype)
    one_wave = whole_audio[sample_rate*1,sample_rate*2]





if __name__ == '__main__':
    main()
