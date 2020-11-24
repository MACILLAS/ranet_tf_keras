# RaNet (Resolution Adaptive Networks for Efficient Inference) on TF 2
## Code By: The Einsteins,  Paper By: Yang et al. 2020

### Term Project for Academic Credit: University of Waterloo, SYDE 671, F20 

#### Overview
RaNet is end-to-end trainable, multi-output, image classifier, that dynamically varies depth depending on classification confidence.

#### Workings
* Set desired hyperparameters in hyperparameters.py (i.e. num_epochs=25 used for debug)
* Run run.py
* Weights are save each epoch in ./models/model_checkpoints

#### Limitations
* RN only hardcoded for cifar10 dataset
* We use 3 classifier blocks instead of the original 6
* We guestimated the details of the original network (since not stated in paper)
* Will make runnable version for Jupyter 

#### To Do
* Implement Densenet and Resnet 
* Compare performance (accuracy and computation speed)