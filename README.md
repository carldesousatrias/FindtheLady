[![arXiv](https://img.shields.io/badge/arXiv-2312.14182-red.svg)](https://arxiv.org/abs/2312.14182) 
# Find the Lady: Permutation and Re-Synchronization of Deep Neural Networks

This is the source code to obtain the different results and figure of the paper. Bellow you can find the details on how each '[file].py' files are used.


## Install environment 

```
conda create -n Findthelady python=3.6
conda activate Findthelady
pip3 install torch torchvision
pip3 install tqdm scikit-image opencv-python numpy pillow pandas
```

## Descriptions of the files
We present the code only for the CIFAR10 dataset. The code for the other datasets/tasks is similar.
### main.py
Proceeds the experiments to test the robustness of our method against Gaussian noise addition, fine-tuning, quantization and pruning.
### Permutation.py
Proceeds the permutation attack against a VGG16 model trained on CIFAR10. A dedicated library about permutation can be found at The library for permutation can be found at [https://github.com/giommariapilo/neuronswap](https://github.com/giommariapilo/neuronswap).
### Basic_ranking.py
Implements the "Finding the canonical space: rank the neurons" method (see paper) and tests its robustness against quantization. (Figure 4 and Figure 5 of the paper).
### Active_ranking.py
Implements the "Learn a specific input to rank the neurons" (see paper) and tests its robustness against fine-tuning.
### Training.py
Permits the training and saving of the models, you can obtain vgg16_Uchi by adding the watermarking line to this file. 
### Permutation.py
Proceeds the permutation attack against a VGG16 model trained on CIFAR10. 
### ProofofConcept.py
Studies the correlation between outputs of a layer (Figure 3 of the paper).
### utils.py
Contains some functions to run the other scripts.

## Experimental results
### Datasets of the paper
* [ImageNet](http://www.image-net.org/)
* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [Cityscapes](https://www.cityscapes-dataset.com/)
* [CoCo](https://cocodataset.org/#home)

### Tables
![Table 1: Robustness to Gaussian noise addition, fine-tuning, quantization, and magnitude pruning](https://github.com/carldesousatrias/FindtheLady/blob/main/data/AAAI24_table_page-0001.jpg?raw=true)

#### Citations and contact
```bibtex
@inproceedings{trias2023lady,
      title={Find the Lady: Permutation and Re-Synchronization of Deep Neural Networks}, 
      author={Carl De Sousa Trias and Mihai Petru Mitrea and Attilio Fiandrotti and Marco Cagnazzo and Sumanta Chaudhuri and Enzo Tartaglione},
      year={2024},
      booktitle={Thirty-Eighth {AAAI} Conference on Artificial Intelligence, {AAAI}
                  2024}
      publisher={{AAAI} Press},
}
```

Contact: [De Sousa Trias Carl](mailto:carl.de-sousa-trias@telecom-sudparis.eu)
