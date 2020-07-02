# Disentangled representation learning with applications in Natural Language Processing
Source code for my Master Thesis at [University of Zagreb, Faculty of Electrical Engineering and Computing](https://www.fer.unizg.hr/en).

## Disentangled representation learning
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

The goal of this thesis was to research the disentanglement methods based on variational autoencoders and extend the applications to the NLP domain. The task of topic modeling was chosen as an application, where learning disentangled representations should produce more coherent, independent and interpretable topics.
The results have shown that neural topic models whose objective is modified with disentangled variants not only produce better results in terms of the average topic normalized mutual pointwise information (NPMI), but also produce more independent and coherent topics. 
Some issues with the evaluation of some neural topic models were also identified and addressed.
Finally, it has also been shown that the average topic NPMI is not a fully sufficient metric for evaluating topic models as it does not include all relevant factors, where proposing a new metric that does is left for future work.
----TODO finish this up
This repository is no longer actively maintained.