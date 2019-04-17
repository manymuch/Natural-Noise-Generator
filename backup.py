import os
import tensorflow as tf

if __name__ == '__main__':
  import argparse
  import glob
  import shutil
  import sys
  import time
  
  parser = argparse.ArgumentParser(description='script for backing up checkpoints')
	
  parser.add_argument('--train_dir', type=str, help='training directory')
  parser.add_argument('--backup_time', type=int, help='time before next backup task begins in minutes')

  parser.set_defaults(train_dir='./train',backup_time=60)
	
  args = parser.parse_args()

  backup_dir = os.path.join(args.train_dir, 'backup')

  if not os.path.exists(backup_dir):
    os.makedirs(backup_dir)

  while tf.train.latest_checkpoint(args.train_dir) is None:
    print('Waiting for first checkpoint')
    time.sleep(2)

  while True:
    latest_ckpt = tf.train.latest_checkpoint(args.train_dir)

    # Sleep for four seconds in case file flushing
    time.sleep(4)

    for fp in glob.glob(latest_ckpt + '*'):
      _, name = os.path.split(fp)
      backup_fp = os.path.join(backup_dir, name)
      print('{}->{}'.format(fp, backup_fp))
      shutil.copyfile(fp, backup_fp)
    print('-' * 80)

    # Sleep for next backup
    time.sleep(int(float(args.backup_time) * 60.))