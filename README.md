# RaNet (Resolution Adaptive Networks for Efficient Inference) on TF 2
## Code By: The Einsteins,  Paper By: Yang et al. 2020

### Term Project for Academic Credit: University of Waterloo, SYDE 671, F20 

#### Overview
Ranet is end-to-end trainable, multi-output, image classifier, that dynamically varies depth depending on classification confidence.
We also train vgg16 and resnet and compare the inference speed and accuracy of these models on the CIFAR10 dataset. 

#### Implementation Details
Our implementation of Ranet has three interlinked models (small, med and large)...
We inherit from the Model from tf.keras.models and overwrite call(), build() and predict()
The layers of Ranet are initialized in constructor and passed as class variables, this is necessary to facilitate adaptive inference. 

small_net, med_net and large_net are defined in functions: build_small_net, build_med_net and build_large_net

call() automatically runs build, which returns a model instance one input and three outputs (output_1, output_2, output_3)
corresponding to (small_net, med_net and large_net). This is necessary to train ranet in an end to end fashion and allow use of predict(). 

predict() is the forward pass of ranet which is distinct from training behaviour.
predict() will first run small_net, then if the confidence of the prediction is below some threshold (self.threshold = 0.85)
med_model will be run etc. When confidence reaches threshold the prediction will be returned. 

Ranet is trained with image augmentation and learning rate scheduler for 200 epochs. Results are saved using tensorboard. 

#### Run Code
###### Ranet
* Set desired hyperparameters in hyperparameters.py
* Ranet is executed through run.py
* run.py has two functions: main() and ranet_prediction()
    * main() will train the ranet model from scratch
        * Weights are save each epoch in ./models/model_checkpoints
        * Weights are also saved at the end of training
    * ranet_prediction () is the experiment script
        * ranet_prediction will train one epoch of ranet then load weights this is a work around due to the unique architecture
        of ranet
        * ranet weights loaded into this script is ranet_model.h5 in ./Ranet

###### VGG16
* VGG16 script (by others) is contained in ./models/vgg116_run.py 
* The vgg script also has two main functions: run_test_harness() and vgg_prediction()
    * run_test_harness() trains vgg16 model from scratch on CIFAR10 
    * vgg_prediction() loads weights from ./models/vgg_final_model.h5 and executes the experiment

###### Resnet50
* Resnet50 script (by others) is contained in ./models/resnet_run.py 
* The resnet script has two main functions: run_test_harness() and resnet_prediction()
    * run_test_harness() transfer learns a resnet50 model pre-trained on imagenet for CIFAR10
        * This is because we could not get very high accuracies training from scratch...
    * resnet_prediction() loads weights from ./models/resnet_model.h5 (not included due to size) and executes experiment 

#### Limitations
* This code currently only works for cifar10 dataset
* We use 3 classifier blocks instead of the original 6
* We guestimated the details of the original network (since not stated in paper)

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







