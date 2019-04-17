import os
import numpy as np
import tensorflow as tf

def train(args):
    from functools import reduce
    from model import WaveGANGenerator, WaveGANDiscriminator
    import glob
    import loader
    
    # Make train dir
    if not os.path.isdir(args.train_dir):
        os.makedirs(args.train_dir)

    fps = glob.glob(os.path.join(args.data_dir, '*'))

    if len(fps) == 0:
       raise Exception('Did not find any audio files in specified directory')
    print('Found {} audio files in specified directory'.format(len(fps)))
    
    with tf.name_scope('loader'):
        x = loader.decode_extract_and_batch(fps,
                                          batch_size=args.train_batch_size,
                                          slice_len=32768,
                                          decode_fs=args.data_sample_rate,
                                          decode_num_channels=args.data_num_channels,
                                          decode_fast_wav=args.data_fast_wav,
                                          decode_parallel_calls=4,
                                          slice_randomize_offset=False,
                                          slice_first_only=args.data_first_slice,
                                          slice_overlap_ratio=0.,
                                          slice_pad_end=True,
                                          repeat=True,
                                          shuffle=True,
                                          shuffle_buffer_size=4096,
                                          prefetch_size=args.train_batch_size * 4,
                                          prefetch_gpu_num=args.data_prefetch_gpu_num)
        x = x[:, :, 0]

    # Make z vector
    z = tf.random_uniform([args.train_batch_size, args.wavegan_latent_dim], -1., 1., dtype=tf.float32)

    # Make generator
    with tf.variable_scope('G'):
        # use first 512 point from real data as y
        y = tf.slice(x, [0, 0, 0], [-1, args.wavegan_smooth_len, -1])
        G_z = WaveGANGenerator(y, z,
                               args.wavegan_kernel_len,
                               args.wavegan_smooth_len,
                               args.wavegan_dim,
                               args.wavegan_batchnorm,
                               train=True)
    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')
    # Print G summary
    print('-' * 80)
    print('Generator vars')
    nparams = 0
    for v in G_vars:
        v_shape = v.get_shape().as_list()
        v_n = reduce(lambda x, y: x * y, v_shape)
        nparams += v_n
        print('{} ({}): {}'.format(v.get_shape().as_list(),v_n,v.name))
    print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))

    # Make real discriminator
    with tf.name_scope('D_x'), tf.variable_scope('D'):
        D_x = WaveGANDiscriminator(x,
                                   args.wavegan_kernel_len,
                                   args.wavegan_dim,
                                   args.wavegan_batchnorm,
                                   args.wavegan_disc_phaseshuffle)
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')

    # Print D summary
    print('-' * 80)
    print('Discriminator vars')
    nparams = 0
    for v in D_vars:
        v_shape = v.get_shape().as_list()
        v_n = reduce(lambda x, y: x * y, v_shape)
        nparams += v_n
        print('{} ({}): {}'.format(v.get_shape().as_list(),v_n,v.name))
    print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))
    print('-' * 80)

    # Make fake discriminator
    with tf.name_scope('D_G_z'), tf.variable_scope('D', reuse=True):
        yG_z = tf.concat([y, G_z], 1)
        print("yG_z shape:")
        print(yG_z.get_shape())
        D_G_z = WaveGANDiscriminator(yG_z,
                                     args.wavegan_kernel_len,
                                     args.wavegan_dim,
                                     args.wavegan_batchnorm,
                                     args.wavegan_disc_phaseshuffle)

    # Create loss
    G_loss = -tf.reduce_mean(D_G_z)
    D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)

    alpha = tf.random_uniform(shape=[args.train_batch_size, 1, 1], minval=0., maxval=1.)
    differences = yG_z - x
    interpolates = x + (alpha * differences)
    with tf.name_scope('D_interp'), tf.variable_scope('D', reuse=True):
        D_interp = WaveGANDiscriminator(interpolates,
                                        args.wavegan_kernel_len,
                                        args.wavegan_dim,
                                        args.wavegan_batchnorm,
                                        args.wavegan_disc_phaseshuffle)
    LAMBDA = 10
    gradients = tf.gradients(D_interp, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)
    D_loss += LAMBDA * gradient_penalty

    # Create (recommended) optimizer
    G_opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)
    D_opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)

    # Create training ops
    G_train_op = G_opt.minimize(G_loss, var_list=G_vars, global_step=tf.train.get_or_create_global_step())
    D_train_op = D_opt.minimize(D_loss, var_list=D_vars)

    # Summarize
    tf.summary.audio('x', x, args.data_sample_rate)
    tf.summary.audio('G_z', G_z, args.data_sample_rate)
    tf.summary.audio('yG_z', yG_z, args.data_sample_rate)

    tf.summary.scalar('G_loss', G_loss)
    tf.summary.scalar('D_loss', D_loss)

    # Run training
    with tf.train.MonitoredTrainingSession(checkpoint_dir=args.train_dir,
                                         save_checkpoint_secs=args.train_save_secs,
                                         save_summaries_secs=args.train_summary_secs) as sess:
        while True:
            # Train discriminator
            from six.moves import xrange
            
            for i in xrange(args.wavegan_disc_nupdates):
                sess.run(D_train_op)

            # Train generator
            sess.run(G_train_op)
            if args.verbose:
                eval_loss_D = D_loss.eval(session=sess)
                eval_loss_G = G_loss.eval(session=sess)
                print(str(eval_loss_D)+","+str(eval_loss_G))


if __name__ == '__main__':
    import argparse
    from main import argument
    
    parser = argparse.ArgumentParser(description='script for training')
    arg = argument(parser)
    arg.train()
    args = arg.args_init()

    train(args)
