# model-for-isles
Models for [Ischemic Stroke Lesion Segmentation](http://www.isles-challenge.org/)  
This project has 3 steps to do:   
* step 1, train and estimate the segmentation model on mri, get the seg result on my own data  
* step 2, do image registration between mri(DWI) and ct, map the mri result on ct for I don't have ct gt data  
* step 3, do as step 1, test how it would perform only using CT data    

step 1 process has finished, future version will apply more model, evaluate and plot performance.  
step 2 is on working.  
step 3 needs CT gt data, get from step 1 and 2 or ask for hospital to help  annotation.  
## Data Prepared  
* MR or CT nii.gz file, better skull stripped, better 1*1*1 mm  
* 0 mean and 1 std  
* modalities + mask + gt
## Data Sample
* ### For training using patch
random choose some samples in roi, each class (central voxel labeled 0 or 1) should have the same number of samples.  
then argument and shuffle before training.  
* ### For testing using patch 
split the image and merge the patch result, when overlap, average them.  
* ### For training using full image   
2d method can use this, todo.  
## Model
### 3D U-net
Encoder: 3 down samples, each down sample, conv-bn-relu-conv-bn-relu-pooling, bn is optional.  
Decoder: 3 up samples, each do upsample/deconv-concate-conv-relu-conv-relu.  
Final: conv(kernel size = 1)-softmax(output p for each class)/sigmoid(output p for foreground).  
### 3D Deep Medic
[Deep medic](https://github.com/Kamnitsask/deepmedic), multi-scale input, no pooling and padding. 
### 2D model todo
U-net/Deconv-net/DeepLab/PSP-net  
## Loss  
* cross-entropy-loss  
* weighted-ce-loss    
## Metric  
* dice coefficient
## Post process  
* [3D dense crf](https://github.com/Kamnitsask/dense3dCrf)  
* classifier to remove false positive. 
