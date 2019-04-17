import os
import numpy as np
import tensorflow as tf

import argparse

from train import train
from infer import infer
from generate import generate

class argument:
    def __init__(self, parser):
        self.parser=parser

        global_args = self.parser.add_argument_group('global')
        global_args.add_argument('--train_dir', type=str, help='training directory')
        global_args.add_argument('--wavegan_latent_dim', type=int,help='number of dimensions of the latent space')
        global_args.add_argument('--wavegan_kernel_len', type=int,help='length of 1D filter kernels')
        global_args.add_argument('--wavegan_dim', type=int,help='dimensionality multiplier for model of G and D')
        global_args.add_argument('--wavegan_batchnorm', action='store_true', dest='wavegan_batchnorm',help='enable batchnorm')
        global_args.add_argument('--wavegan_smooth_len', type=int, help='length of the pervious audio used to smooth the connection')
        
        self.parser.set_defaults(train_dir = './train',
                                 wavegan_latent_dim=100,
                                 wavegan_kernel_len=25,
                                 wavegan_dim=64,
                                 wavegan_batchnorm=False,
                                 wavegan_smooth_len=4096)

    def train(self):
        data_args = self.parser.add_argument_group('data for train')
        data_args.add_argument('--data_dir', type=str,help='data directory containing *only* audio files to load')
        data_args.add_argument('--data_sample_rate', type=int,help='number of audio samples per second')
        data_args.add_argument('--data_num_channels', type=int,help='number of audio channels to generate (for >2, must match that of data)')
        data_args.add_argument('--data_first_slice', action='store_true', dest='data_first_slice',help='if set, only use the first slice each audio example')
        data_args.add_argument('--data_normalize', action='store_true', dest='data_normalize',help='if set, normalize the training examples')
        data_args.add_argument('--data_fast_wav', action='store_true', dest='data_fast_wav',help='if your data is comprised of standard WAV files (16-bit signed PCM or 32-bit float), set to decode audio using scipy (faster) instead of librosa')
        data_args.add_argument('--data_prefetch_gpu_num', type=int,help='if nonnegative, prefetch examples to this GPU (Tensorflow device num)')
        
        train_args = self.parser.add_argument_group('train')
        train_args.add_argument('--train_batch_size', type=int,help='batch size')
        train_args.add_argument('--train_save_secs', type=int,help='how often to save model')
        train_args.add_argument('--train_summary_secs', type=int,help='how often to report summaries')
        train_args.add_argument('--verbose', action='store_true',dest='verbose',help='if set, print G and D loss to stdout')
        train_args.add_argument('--wavegan_disc_nupdates', type=int,help='number of discriminator updates per generator update')
        train_args.add_argument('--wavegan_disc_phaseshuffle', type=int,help='radius of phase shuffle operation')
        
        self.parser.set_defaults(data_dir=None,
                                 data_sample_rate=11025,
                                 data_num_channels=1,
                                 data_first_slice=False,
                                 data_normalize=False,
                                 data_fast_wav=False,
                                 data_prefetch_gpu_num=0,
                                 train_batch_size=64,
                                 train_save_secs=300,
                                 train_summary_secs=120,
                                 verbose=False,
                                 wavegan_disc_nupdates=5,
                                 wavegan_disc_phaseshuffle=2)

    def generate(self):
        generate_args = self.parser.add_argument_group('generate')
        generate_args.add_argument('--ckpt_path', type=str, help='use chosen checkpoint to generate')
        generate_args.add_argument('--wav_out_path', type=str, help='path to output wav file')
        
        self.parser.set_defaults(ckpt_path = './train/model.ckpt-0',
                                 wav_out_path = './gen.wav')
    def main(self):
        self.train()
        self.generate()
        main_args = self.parser.add_argument_group('main')
        main_args.add_argument('mode', type=str, choices=['train', 'infer', 'generate'], help='choose mode')

    def args_init(self):
        return self.parser.parse_args()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main script for training and generating')
    arg = argument(parser)
    arg.main()
    args = arg.args_init()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'generate':
        infer(args)
        generate(args)
    elif args.mode == 'infer':
        infer(args)
