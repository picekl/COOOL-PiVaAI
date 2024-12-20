# COOOL-PiVaAI

## Installation Guide
To set up the project, simply install the dependencies listed in `requirements.txt` using the following command:

```bash
pip install -r requirements.txt
```
## Video pre-processing
The approach is based on a few steps. First, we process the video data with available models, and then we work with the predictions while considering all three metrics. 

We provide notebooks to pre-process the data and to allow full reproducibility; you must run all the notebooks, and the inference could take a few hours. For example, the MOLMO captioning requires 40Gb of vRAM and runs for 7 hours.

To allow easier "results verification," we include all the pre-processing results in a separate pickle file.

⚠️ **Use LFS to clone the repo** ⚠️

---

### 1. OpticalFlow 🌊

In this step, we calculate **OpticalFlow** for each image in the video, tracking the motion between consecutive frames. The results are stored in a [pickle file](https://github.com/picekl/COOOL-PiVaAI/tree/main/resources/optical-flow).

The notebook to prepare the OpticalFlow outputs can be found [here](https://github.com/picekl/COOOL-PiVaAI/blob/main/video-preprocessing/optical-flow/run-optical-flow.ipynb).

⚠️ **Note on Compatibility:** During our testing, we found that some versions of **OpenCV** or certain hardware configurations might cause OpticalFlow to output `inf` values. This can be resolved by upgrading or downgrading OpenCV. Despite this, it worked well on **4 out of 5** machines.

---

### 2. Image Classification with CIFAR Pre-trained Model 📸

Each bounding box identified in the video is classified using a **CIFAR pre-trained model**. This classification helps filter out non-hazardous objects.

The classification results are stored in a [pickle file](https://github.com/picekl/COOOL-PiVaAI/tree/main/resources/cifar-classification).

To process this data, use the notebook located [here](https://github.com/picekl/COOOL-PiVaAI/blob/main/video-preprocessing/cifar-classification/run-cifar-obj-cls.ipynb).

---

### 3. MOLMO Captioning 📝

For each object in the video, we generate captions for the **top 5 largest bounding boxes**. To avoid distortion, each object is cropped before caption generation.

The captioning results are stored in a [pickle file](https://github.com/picekl/COOOL-PiVaAI/tree/main/resources/molmo-captions).

The notebook for processing this data is available [here](https://github.com/picekl/COOOL-PiVaAI/blob/main/video-preprocessing/molmo-captioning/run-molmo-largest-obj-cap.ipynb).

Additionally, we tested several alternative prompts to generate captions. While the results varied, the performance was not satisfactory compared to the largest objects. You can explore these experiments in the notebook located [here](https://github.com/picekl/COOOL-PiVaAI/blob/main/video-preprocessing/molmo-captioning/run-molmo-obj-cap.ipynb).

---

### Depth - Estimation [Not used yet] 
TBD

---

## Making a Submission 🚀

To make a submission, run the following notebooks. These will automatically update the submission with the results of each step:

1. [DriverState Estimation (Bounding Box Sizes)](1.DriverState+HazardTrack-Estimation-BboxSizes.ipynb)
2. [DriverState Estimation (OpticalFlow)](2.DriverState-Estimation-OpticalFlow.ipynb)
3. [DriverState Estimation (Ensemble)](3.DriverState+HazardTrack-Estimation-Ensemble.ipynb)
4. [Hazard Captioning + HazardTrack Filtering](4.HazardCaptioning.ipynb)