# Natural-Noise-Generator

## Intuition

There are many applications that play natural noise like forest, stove, forest birds singing, sea wave, etc.  
But actually they are playing a audio clip again and again, which means they repeat every several minutes.  
Sometimes it may be a little bit annoying when noticing.

So we want to create a Natural Noise Generator, basically based on [waveGAN](https://github.com/chrisdonahue/wavegan), to generate natural noise continously without repeating.

## BU SCC Environment Setup
Since there is no tensorflow=1.12 on SCC, we need to use anaconda to setup environment
1. module load anaconda3
2. conda install tensorflow
3. conda will claim you have no root access, follow the instruction to clone root environment
4. source activate my_root
5. conda install -c anaconda tensorflow-gpu
6. pip install --user librosa==0.6.2  
Done!  
Then you can submit the job to SCC using 
```gpu.sh```  
remember to modify project name
