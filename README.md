# Find the Lady: permutation and re-synchronization of deep neural networks

This is the source to obtain the different results and figure in the paper. Bellow we detailed how each .py files are used.


## Install environment 

```
conda create -n Findthelady python=3.6
conda activate Findthelady
pip3 install torch torchvision
pip3 install tqdm scikit-image opencv-python numpy pillow pandas
```
## Dataset
* [ImageNet](http://www.image-net.org/)
* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [Cityscapes](https://www.cityscapes-dataset.com/)
* [CoCo](https://cocodataset.org/#home)

## Descriptions of the files
We present the code for the CIFAR10 dataset. The code for the other datasets/tasks is similar.
### Active_ranking.py
Implement the "Learn a specific input to rank the neurons" (see paper) and tests its robustness against fine-tuning.
### Active_ranking_one.py
Implement the "Learn to distinguish one neuron at a time" (see paper) and tests its robustness against quantization.
### Basic_ranking.py
Implement the "Finding the canonical space: rank the neurons" method (see paper) and tests its robustness against quantization. This file permits producing Figure 4 and Figure 5 of the paper.
### Correlation_noise.py
Proceeds the experiments to test the robustness of our method against Gaussian noise addition. 
### Correlation_ft.py
Proceeds the experiments to test the robustness of our method against fine-tuning. 
### Correlation_ws.py
Proceeds the experiments to test the robustness of our method against quantization and pruning.
### main.py
Permits the training from scratch a model on a dataset (only use for CIFAR10 experiments).
### Pandas_to_plot_ebar.py
Plots all the results (with error bar) which permits producing Figure 7, Figure 8 ad Figure 9 of the paper.
### Permutation.py
Proceeds the permutation attack against a VGG16 model trained on CIFAR10. This file permits producing Table 1 of the paper.
### Proof_of_concept.py
Studies the correlation between outputs of a layer which permits producing Figure 3 of the paper.
### utils.py
Contains some functions to run the other scripts.