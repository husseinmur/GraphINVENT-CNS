# GraphINVENT-CNS
An implementation of the GraphINVENT framework using curated data of CNS molecules. The framework was optimized for the drug discovery of Parkinson's disease. 

## Prerequisites
* Anaconda or Miniconda with Python 3.6 or 3.8.
* (for GPU-training only) CUDA-enabled GPU.

## References
This work is based on GraphINVENT framework:

https://github.com/MolecularAI/GraphINVENT

```
@article{mercado2020graph,
  author = "Rocío Mercado and Tobias Rastemo and Edvard Lindelöf and Günter Klambauer and Ola Engkvist and Hongming Chen and Esben Jannik Bjerrum",
  title = "{Graph Networks for Molecular Design}",
  journal = {Machine Learning: Science and Technology},
  year = {2020},
  publisher = {IOP Publishing},
  doi = "10.1088/2632-2153/abcf91"
}

@article{mercado2020practical,
  author = "Rocío Mercado and Tobias Rastemo and Edvard Lindelöf and Günter Klambauer and Ola Engkvist and Hongming Chen and Esben Jannik Bjerrum",
  title = "{Practical Notes on Building Molecular Graph Generative Models}",
  journal = {Applied AI Letters},
  year = {2020},
  publisher = {Wiley Online Library},
  doi = "10.1002/ail2.18"
}
```


It also makes use of the following filters/benchmarkers to test resulting molecules:
* DeePred-BBB: A Blood Brain Barrier Permeability Prediction Model With Improved Accuracy
https://github.com/12rajnish/DeePred-BBB

* GuacaMol:

https://github.com/BenevolentAI/guacamol

The weights of the embedding layer in the LSTM QSAR model were initialized to the weights of the pretrained word2vec model "Mol2Vec":

https://github.com/samoturk/mol2vec


### Related work
#### MPNNs
The MPNN implementations used in this work were pulled from Edvard Lindelöf's repo in October 2018, while he was a masters student in the MAI group. This work is available at
https://github.com/edvardlindelof/graph-neural-networks-for-drug-discovery.

His master's thesis, describing the EMN implementation, can be found at

https://odr.chalmers.se/handle/20.500.12380/256629.

RL-GraphINVENT
https://github.com/olsson-group/RL-GraphINVENT and(https://doi.org/10.33774/chemrxiv-2021-9w3tc).

Graph traversal algorithms in GraphINVENT
(https://doi.org/10.33774/chemrxiv-2021-5c5l1)

## License
GraphINVENT is licensed under the MIT license and is free and provided as-is.
