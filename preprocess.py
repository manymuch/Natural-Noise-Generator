import numpy as np
from librosa.output import write_wav as wav_w
from librosa.core import load as wav_r


def main():
    file_path = "beach/beach.wav"
    demo_wave = wav_r(file_path,sr=11025,duration=32768/11025)
    print(demo_wave.shape)
    

if __name__ == '__main__':
    main()
