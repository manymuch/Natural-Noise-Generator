import os
import numpy as np
from librosa.output import write_wav as wav_w
from librosa.core import load as wav_r


def main():
    onewave_path = "beach_sliced/one_wave.wav"
    whole_audio_path = "beach/beach.wav"
    output_dir = "beach_sliced"
    sample_rate = 11025
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    onewave = wav_r(onewave_path,sr=sample_rate)[0]
    whole_audio = wav_r(whole_audio_path,sr=sample_rate)[0]
    print(onewave.shape)
    print(whole_audio.shape)





if __name__ == '__main__':
    main()
