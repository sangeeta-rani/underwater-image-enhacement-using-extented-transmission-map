
# Haze, color cast and texture enhancement using extented transmission map from the underwater images  

This code is useful for removing dense haze, color cast and texture improvent from the underwater images caused by challenging environment.


1 Removing color cast issues using color correction model from the underwater images

We can utilize the color correction  model to eliminate color cast issues  from underwater images

2  Texture improvement using texture preserving module of Underwater Images
The texture preserving module enables the recovery of texture details from the high frequency region of the underwater images

3  Haze removal module

The Haze removal module network is employed to remove haze from the  underwater images.







Here is the list of libraries you need to install to execute the code:

python = 3.6
cv2
numpy
scipy
matplotlib
scikit-image
natsort
math
datetime
```
    
1 Complete the running environment configuration;
2 Replace  the inputs images path to corresponding place given in  the code
3  Run Python texture preserving.py, haze removal.py and color correction.py;
4 Find the enhanced/restored images 
Datasets can be downloaded using the provided links:
(i) https://paperswithcode.com/dataset/uieb
(ii) https://paperswithcode.com/dataset/uccs   
(iii)  https://www.kaggle.com/datasets/pamuduranasinghe/euvp-dataset   
