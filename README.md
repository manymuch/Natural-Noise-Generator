# Natural-Noise-Generator

## Intuition

There are many applications that play natural noise like forest, stove, forest birds singing, sea wave, etc.  
But actually they are playing a audio clip again and again, which means they repeat every several minutes.  
Sometimes it may be a little bit annoying when noticing.

So we want to create a Natural Noise Generator, basically based on [waveGAN](https://github.com/chrisdonahue/wavegan), to generate natural noise continously without repeating.

## BU SCC Environment Setup

Since there is no tensorflow=1.12 on SCC, we need to use anaconda to setup environment
1. ```module load anaconda3```
2. ```conda create -n my_root --clone="/share/pkg/anaconda3/4.4.0/install"```
3. ```source activate my_root```
4. ```conda install -c anaconda tensorflow-gpu```
5. ```pip install --user librosa==0.6.2```  
Done!  
Then you can submit the job to SCC using  
```qsub train_birds.sh```  
remember to modify project name

Generator Inferencing
```python train_wavegan.py preview ./train --wavegan_genr_pp```

## Remote Tensorboard
open a terminal  
```ssh -NfL localhost:16006:localhost:6006 jiaxin@scc1.bu.edu```  
open another terminal  
```ssh jiaxin@scc1.bu.edu```  
navigate to the working directory  
```module load python/3.6.2 tensorflow/r1.10```  
```tensorboard --logdir=./train --port 6006```  
in the web browser [localhost:16006](http://localhost:16006)  


## SCC5
```module load python3/3.6.5```  
```module load tensorflow/r1.12```  
```module load cuda/9.0```  
