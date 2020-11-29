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
* Will make runnable version for Jupyter (Optional)

#### Experiment
We compare our Ranet model with VGG16 and Resnet50... Ranet and VGG16 are trained from scratch and have a validation 
accuracies of 87% and 86% respectively. Resnet50 was transfer learned with weights from Imagenet with a validation accuracy of 
88%.

We randomly choose 100 images on which each model will predict. The average prediction times per image are recorded.
This is repeated three times for each model. We calculate the accuracy of Ranet but assume the average validation accuracy
is same as prediction. 

Inference times will depend on specific hardware. These models are trained on Ubuntu 18.04 with TF2-GPU, single Nvidia Titan V (CUDA 9) 

Resnet50 average per image inference times (seconds): 0.04398, 0.04471 and 0.04402 avg time: 0.04423 acc: 88%

VGG16 average per image inference times (seconds): 0.0325, 0.0324 and 0.03442 avg time: 0.03311 acc: 86%

Ranet average per image inference time (seconds): 0.017 (85%), 0.0162 (89%) and 0.0157 (87%) avg time: 0.0163 acc: 87%

Ranet is on average 2x faster than VGG16 and 2.8x faster than Resnet50 with relative similar accuracy on CIFAR10. 







