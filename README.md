# COOOL-PiVaAI

## Installation Guide
All dependencies are available in the ```requirements.txt```.
Simple ```pip install -r requirements.txt``` should work.

## Video pre-processing
The approach is based on a few steps. First, we process the video data with available models, and then we work with the predictions while considering all three metrics. 

We provide notebooks to pre-process the data and to allow full reproducibility; you must run all the notebooks, and the inference could take a few hours. For example, the MOLMO captioning requires 40Gb of vRAM and runs for 7 hours.

To allow easier "results verification," we include all the pre-processing results in a separate pickle file.

⚠️ **Use LFS to clone the repo** ⚠️

### 1. OpticalFlow
An OpticalFlow for each image is run, and results are stored in a [pickle file](https://github.com/picekl/COOOL-PiVaAI/tree/main/resources/optical-flow). 

⚠️ While testing the "reproducibility," we have noticed that for some versions of the OpenCV or HW, the OpticalFlow might result in an inf value. This can be fixed by upgrading or downgrading the OpenCV. However, it worked on 4 out of 5 machines pretty well.

### 2. Image Classification with Cifar pre-trained model
🚧⏳🔄🛠️ 

### 3. MOLMO captioning
🚧⏳🔄🛠️ 

### Depth - Estimation [Not used yet] 
TBD

## Making a Submission

Run the following notebooks that will continuously update the submission:
```[1.DriverState+HazardTrack-Estimation-BboxSizes.ipynb[()```
```[2.DriverState-Estimation-OpticalFlow.ipynb]()```
```[3.DriverState+HazardTrack-Estimation-Ensemble.ipynb]()```
```[4.HazardCaptioning.ipynb]()```