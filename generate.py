import os
import numpy as np
import tensorflow as tf


def generate(args):
    if args.wav_out_time is None:
        out_length=float('inf')
    else:   
        out_length=args.wav_out_time * 16000
        
    infer_dir = os.path.join(args.train_dir, 'infer')
    infer_metagraph_fp = os.path.join(infer_dir, 'infer.meta')
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(infer_metagraph_fp)
    graph = tf.get_default_graph()
    
    with tf.Session() as sess:
        from librosa.output import write_wav as wav_w
        
        saver.restore(sess, args.ckpt_path)
        z = graph.get_tensor_by_name('z:0')
        y = graph.get_tensor_by_name('y:0')
        G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]
        
        # Loop_Init
        print("Generating Initialized!")
        _y = np.zeros([1, args.wavegan_smooth_len,1])
        wv = np.zeros([1,1])
        gen_count = 3
        
        # Loop
        try: 
            while wv.shape[1]<out_length:
                _z = (np.random.rand(1, 100) * 2.) - 1.
                wv = np.concatenate((wv,sess.run(G_z, {y: _y, z: _z})), axis = 1)
                _y = np.reshape(wv[:,-1-args.wavegan_smooth_len:-1], (1,-1,1))
                gen_count = gen_count+1
                
                if gen_count==4:
                    wav_w(args.wav_out_path, wv[0, :], 16000)
                    gen_count=0
            wav_w(args.wav_out_path, wv[0, 0:out_length-1], 16000)
            print("Generating Finished!")
            
        except KeyboardInterrupt:
            wav_w(args.wav_out_path, wv[0, 0:min(wv.shape[1],out_length-1)], 16000)
            print("KeyboardInterrupt Called!")
            


if __name__ == '__main__':
    import argparse
    from main import argument
	
    parser = argparse.ArgumentParser(description='script for generating')
    arg = argument(parser)
    arg.generate()
    args = arg.args_init()
	
    from infer import infer
    infer(args)
    generate(args)
