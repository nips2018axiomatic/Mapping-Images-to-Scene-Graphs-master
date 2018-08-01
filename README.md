# Mapping-Images-to-Scene-Graphs
Mapping Images to Scene Graphs with Permutation-Invariant Structured Prediction

<!---[alt text](sg_example_final.png)--->
<img src="sg_example_final.png" width="750">


## Introduction
Scene graph prediction is the task of mapping an image into a set of bounding boxes, along with their categories and relations.
<!--- In the scene graph prediction task, the input is an image annotated with a set of bounding boxes. The goal is to label each bounding box with the correct entity category, and every pair of entities with their relation, such that they form a coherent graph, known as a scene graph. --->

We present a new architecture for graph inference that has the **following structural property**:
on the one hand, the architecture is invariant to input permutations; 
on the other hand, every permutation-invariant function can be implemented via this architecture.

In this repository, we share our architecture implementation for the task of scene graph prediction.

## Model implementation
**Scene Graph Predictor (SGP)** gets as an input inital confidience distributions per entity and relation and processes these to obtain new labels. SGP satisfies the graph permutation invariance property intoduced in the paper.
The model is implemented in [TensorFlow](https://www.tensorflow.org/).
For the initial confidence distributions per entity and relation, we simply re-use features learned by the baseline model from Zellers et al. (2017). (git repositiry https://github.com/rowanz/neural-motifs)



## SGP architecture
Our SGP implementation is using an iteratively RNN to process predictions. Each step outputs an improved predictions.

<img src="sgp_arch_git.png" width="750">

A schematic representation of the architecture. Given an image, a Label predictor outputs initial predictions ![equation](http://latex.codecogs.com/gif.latex?Z_%7Bi%7D%2C%20Z_%7Bi%2Cj%7D). Then, our SGP model, computes each ![equation](http://latex.codecogs.com/gif.latex?%5Cphi_%7Bi%2Cj%7D) element wise. Next, they are summed to create vector ![equation](http://latex.codecogs.com/gif.latex?S_%7Bi%7D), which is concatenated with ![equation](http://latex.codecogs.com/gif.latex?Z_%7Bi%7D). Then, ![equation](http://latex.codecogs.com/gif.latex?%5Calpha) is applied, and another summation creates the graph representation. Finally, ![equation](http://latex.codecogs.com/gif.latex?%5Crho_%7Bentity%7D) classifies objects and ![equation](http://latex.codecogs.com/gif.latex?%5Crho_%7Brelation%7D) classifies relation. The process of SGP could be repeated iteratively (in the paper we repeat it 2 times).

For more information, please look at the code (Module/Module.py file) and the paper.


## Attention with SGP architecture
Our SGP architecture uses attention at the feature-level for each node during inference. 
We weight the significance of each feature per node, such that the network can choose which features from adjacent nodes contributes the most information.

<img src="qualitive_results_att_boxes.png" width="750">

An example of attention per entities and global attention over all nodes. The size and location of objects provide a key signal to the attention mechanism. The model assigns higher confidence for the label "tie" when the label "shirt" is detected (third panel from the left). Similarly, the model assigns a higher confidence for the label "eye" when it is located near "hair".

## Dependencies
To get started with the framework, install the following dependencies:
- [Python 2.7](https://www.python.org/)
- [tensorflow-gpu 1.0.1](https://www.tensorflow.org/)
- [matplotlib 2.0.2](http://matplotlib.org/)
- [h5py 2.7.0](http://www.h5py.org/)
- [numpy 1.12.1](http://www.numpy.org/)
- [pyyaml 3.12](https://pypi.python.org/pypi/PyYAML)

Run `"pip install -r requirements.txt"`  - to install all the requirements.


## Usage
1. Run `"python Run.py download"` to download and extract train, validation and test data. The data already contains the result of applying the baseline detecor over the VisualGenome data. 
2. Run `"python Run.py eval gpi_linguistic_pretrained <gpu-number>"` to evaluate the pre-trained model of our best variant, linguistic with multi-head attention. (recall@100 SG Classification).
3. Run `"python Run.py train gpi_linguistic <gpu-number>"` to train a new model (linguistic with multi-head attention).
4. Run `"python Run.py eval gpi_linguistic_best <gpu-number>"` to evaluate the new model. (recall@100 SG Classification).


