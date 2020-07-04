# Disentangled Representation Learning with Applications in Natural Language Processing
Source code for my Master Thesis at [University of Zagreb, Faculty of Electrical Engineering and Computing](https://www.fer.unizg.hr/en).

## Disentangled Representation Learning
Representations learned by deep neural networks are hardly interpretable, which might obstruct reusing them for some tasks.
The goal of disentangled representation learning is to learn separate informative features, which should create an interpretable representation.
Most prominent disentangled representation learning approaches rely on variational autoencoders and modifying the variational lower bound, focusing mostly on the computer vision domain.
Repositories providing a good overview of recent methods and experiments on datasets in the computer vision domain:
 * [disentanglement_lib](https://github.com/google-research/disentanglement_lib/)
 * [PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE )

Some relevant papers for getting into disentangled representations: 
 * [Recent advances in Autoencoder-based representation learning](https://arxiv.org/pdf/1812.05069.pdf)
 * [&beta;-VAE: Learning basic visual concepts with a constrained variational framework](https://openreview.net/pdf?id=Sy2fzU9gl)
 * [ELBO surgery: yet another way to carve up the variational evidence lower bound](http://approximateinference.org/accepted/HoffmanJohnson2016.pdf)  
 * [Structured disentangled representations](https://arxiv.org/pdf/1804.02086v4.pdf)
 * [Challenging common assumptions in the unsupervised learning of disentangled representations](https://arxiv.org/pdf/1811.12359.pdf)

## Topic Modeling
The goal of this thesis was to research the disentanglement methods based on variational autoencoders and extend the applications to the NLP domain.
Topic modeling is a task that has a goal of discovering hidden semantic structures, or topics, that occur in a collection of documents, and are widely 
applied to help better organize and understand large unstructured text, and has been chosen as the area of application of disentangled representation learning in NLP.
Inference methods used in traditional topic models such as [LDA](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) have become increasingly complex when
dealing with large datasets. As topic modeling can be seen as inferring the posterior distribution over the latent topics, the task can be successfully approached
using variational inference and variational autoencoders.
Relevant neural topic models based on variational autoencoders:
 * [Neural Variational Document Model](https://arxiv.org/pdf/1511.06038.pdf)
 * [Discovering Discrete Latent Topics with Neural Variational Inference](https://dl.acm.org/doi/pdf/10.5555/3305890.3305930)
 * [Coherence-Aware Neural Topic Modeling](https://arxiv.org/pdf/1809.02687.pdf)
 * [Autoencoding Variational Inference for Topic Models](https://arxiv.org/pdf/1703.01488.pdf)
 
Learning disentangled representations in the domain of topic modeling should produce more coherent, independent and interpretable topics.
The focus in experiments in this thesis was to modify the objective-variatinal lower bound of the mentioned
neural topic models to induce disentanglement which should lead to more independent and coherent topics.
To do so the total correlation term in the further decomposition of the ELBO was penalized, following 
[&beta;-TCVAE](https://arxiv.org/pdf/1802.04942.pdf) and [HFVAE](https://arxiv.org/abs/1804.02086).

## Usage
Install the conda environment from environment.yml, or create a new virtual environment and install requirements from requirements.txt
Models currently implemented are : [NVDM](https://arxiv.org/pdf/1511.06038.pdf), [NTM (Neural topic model)](https://arxiv.org/pdf/1809.02687.pdf), [GSM (Gaussian softmax model)](https://dl.acm.org/doi/pdf/10.5555/3305890.3305930).
Use [run.py](run.py) for running the experiments. The program expects the input dataset in a BoW (Bag of words) format, either as a numpy array (.npy) or a sparse array (.npz). 
The program expects a .yaml config file (see [configs](configs)).
For evaluating the trained model use [evaluation/evaluate_model.py](evaluation/evaluate_model.py).


## Results
The experiments were performed on the [20Newsgroups](http://qwone.com/~jason/20Newsgroups/) dataset.
The preprocessed dataset provided by [Autoencoding Variational Inference for Topic Models](https://arxiv.org/pdf/1703.01488.pdf) was used to avoid preprocessing differences and enable comparison with related work.
The HFVAE an &beta;-TCVAE objectives were implemented using [ProbTorch](https://github.com/probtorch/probtorch), a probabilistic library for deep generative models that extends PyTorch. 
The results have shown that neural topic models whose objective is modified with disentangled variants not only produce better results in terms of the average topic normalized mutual pointwise information (NPMI) (seen in tables),
but also produce more independent and coherent topics (discussed in more detail in the thesis).
The values in the brackets () indicate the values of &beta; and &gamma; in used in the HFVAE objective. 


| num topics    | 50   |  20  |
| ---           | :---:| :---:|
| NVDM          | 0.15 | 0.15 |
| +HFVAE(20,20) | **0.20** | **0.20** |
| +HFVAE(10,5)  | 0.20 | 0.18 |
| NTM           | 0.21 | 0.19 |
| +HFVAE(20,20) | O.24 | 0.20 |
| +HFVAE(10,5)  | **0.25** | **0.21** |
| GSM           | 0.23 | 0.22 |
| +HFVAE(20,20) | 0.23 | 0.23 |
| +HFVAE(10,5)  | **0.24** | **0.24** | 
Some issues with the evaluation of some neural topic models were also identified and addressed.
Finally, it has also been shown that the average topic NPMI is not a fully sufficient metric for evaluating topic models as it does not include all relevant factors, where proposing a new metric that does is left for future work.
The methods show a promising future step for neural topic modeling and learning disentangled representations in NLP.

## Project status
This repository is no longer actively maintained.