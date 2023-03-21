# robot-pattern_gen


This is a repository for studying generative pattern models for the robot. 

Now, we implemented a score-conditioned generative model (by leveraging conditional VAE) and proxy score function. 


## Training proxy and generator
First, we train proxy score function $f_{\phi}(x) \approx f(x)$ using offline data $D_{\text{off}} = (x_i,f(x_i))_{i=1}^{N}$, where $x_i$ is pattern data (represented as image tensor) and $f(x_i)$ stands for score from robot simulation function $f$. 

Second, we train conditional VAE to attain score conditioned generator which satisfies $p(x|y,z) \propto 1_{f(x)=y}$ and $z \sim \mathcal{N}(0,1)$ using offline data $D_{\text{off}}$. 


## Samples pattern from trained generator and proxy

First, we samples $M$ candidates of patterns $X_{\text{candidates}}$ as: $x \sim p(x|y^{\text{max}},z)$ where $z \sim \mathcal{N}(0,1)$. Note $y^{\text{max}}$ is some high value we desire to get. 

Note that $X_{\text{candidates}}$ stands for the set of sampled $x$ where $|X_{\text{candidates}}| = M$.


Second, we screen the $M$ candidates as TopK patterns based on the score evaluated by the proxy function $f_{\phi}$: $X_{\text{final}}:= \text{TopK}(f_{\phi}, X_{\text{candidates}})$.
Specifically, for each sample $x \in X_{\text{candidates}}$, (1) we assign pseudo-score using $f_{\phi}(x)$, (2) sort it based on the suede-score and (3) collect Top K samples among them. 


## Dependancies

* Python>=3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.1
* tqdm
* Matplotlib 


## How to run?

```bash
python train.py
```

