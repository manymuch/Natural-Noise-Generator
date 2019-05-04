# Natural-Noise-Generator

## Intuition

There are many applications that play natural noise like forest, stove, forest birds singing, sea wave, etc.  
But actually they are playing a audio clip again and again, which means they repeat every several minutes.  
Sometimes it may be a little bit annoying when noticing.

So we want to create a Natural Noise Generator, basically based on [waveGAN](https://github.com/chrisdonahue/wavegan), to generate natural noise continously without repeating.

## Model

Based on WaveGAN, the Generator takes additional input from last part of previous audio sample as Y. See [model.py](https://github.com/manymuch/Natural-Noise-Generator/blob/master/model.py#L64) for details

## Running  
Trainning  
```
python main.py train --data_dir ./data/ --data_fast_wav --verbose
```

Generator Inferencing  
```
python main.py generate --wav_out_time 150
```


## BU SCC Environment Setup

Since there is no tensorflow=1.12 on SCC, we need to use anaconda to setup environment
1. ```module load anaconda3```
2. ```conda create -n my_root --clone="/share/pkg/anaconda3/4.4.0/install"```
3. ```source activate my_root```
4. ```conda install -c anaconda tensorflow-gpu```
5. ```pip install --user librosa==0.6.2```  
Done!  
Then you can submit the job to SCC using ```qsub train.sh```  
remember to modify project name


## Remote Tensorboard
open a terminal  
```ssh -NfL localhost:16007:localhost:6007 jiaxin@scc1.bu.edu```  
open another terminal  
```ssh jiaxin@scc1.bu.edu```  
navigate to the working directory  
```module load python/3.6.2 tensorflow/r1.10```  
```tensorboard --logdir=./train --port 6007```  
in the web browser [localhost:16007](http://localhost:16007)  
