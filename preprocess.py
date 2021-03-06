import os
import numpy as np
from scipy.io.wavfile import read as wav_r
from scipy.io.wavfile import write as wav_w
from scipy.signal import find_peaks #require scipy>=1.10


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

    convolve_result = np.convolve(np.flip(one_wave,axis=0),whole_audio[:output_lenght*10],mode='valid')
    print(convolve_result.shape)
    print(convolve_result.max(axis=0))
    print(convolve_result.mean())
    peaks = find_peaks(convolve_result,height=20000,distance=5000)[0]
    total_num = peaks.shape[0]
    for i in range(total_num):
        begin_index = peaks[i]
        wav_w(output_dir+str(i)+".wav",sample_rate,whole_audio[begin_index:begin_index+output_lenght])









if __name__ == '__main__':
    main()
