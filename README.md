# COOOL-PiVaAI


## Installation Guide
 - Install requirements #TODO

## Video Processing

### Cifar Klasifikace
  
### MOLMO captioning

### OpticalFlow

### Depth - Estimation
Produce depth maps for frames of videos in specified frequency. Save them as .jpegs.
- ```video-process/depth_anything```
- Run ```depth-anything.ipynb```
    
## Driver state change

### Bboxes method
Produce driver state change predictions based on change of total amount of bounding box pixels between frames of videos.
- ```driver-state/*```
- Run ```run_bboxsizes.ipynb```

### opticalflow method

### ensembling

## Hazard recognition

### Predict all available tracks as hazard tracks - Extended baseline 
Produce hazard tracks predictions based on extension of the baseline approach where all available tracks for each frame are predicted as hazard tracks.
Hazard track IDs are sorted based on the proximity to the center of the video frame.
- ```driver-state/*```
- Run ```run_bboxsizes_alltracks.ipynb```



### all - cifar classification

## Hazard Captioning
    Prostě molmo.