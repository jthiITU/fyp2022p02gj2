import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import feature, img_as_ubyte,measure,color,morphology, io, exposure
from skimage.transform import rescale, resize, downscale_local_mean,rotate
from skimage.segmentation import slic, mark_boundaries
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, auc, confusion_matrix, precision_score, pairwise
from imblearn.over_sampling import RandomOverSampler
import os
import cv2
import math
import joblib

#greyscaling function
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


#Attempts to create mask with better defined borders using decolored versions of im and based on seg
def create_custom_mask(im,seg):
    #Slightly increase contrast for better color detection 
    im = exposure.rescale_intensity(im) 

    #Keep green values, blue values and greyscale images in an attempt
    #to make red colors as low as possible (since most of the lesions are red)
    #This also removes any colored circles (if present)
    gray_1 = exposure.rescale_intensity(im[:,:,1]) #green only
    gray_2 = exposure.rescale_intensity(im[:,:,2]) #blue only
    gray_3 = exposure.rescale_intensity(rgb2gray(im)) #all greyscaled
    
    imlist = []
    difflist = []
    same_counter = 0
    index = 0
    #For loop that creates a mask at different thresholds of color intensity
    #quant is a quantile threshold based on the color histogram of gray_1, gray_2, and gray_3
    for quant in np.arange(0.15,1,0.075): 
        
        #first mask with green only
        mymask_1 = gray_1 < np.quantile(gray_1,quant) 
        #second mask with blue only
        mymask_2 = gray_2 < np.quantile(gray_2,quant) 
        #third mask with greyscaled image
        mymask_3 = gray_3 < np.quantile(gray_3,quant)

        #Superpose all masks (including original segment) and keep joint elements
        better_mask = mymask_1.astype('int32')+mymask_2.astype('int32')+mymask_3.astype('int32')
        fixed_mask = better_mask.copy() + seg

        #make mask boolean
        fixed_mask[fixed_mask < 4] = 0
        fixed_mask[fixed_mask == 4] = 1
        fixed_mask = fixed_mask.astype("bool")
        
        #Morphology transformations to generalize mask
        fixed_mask = morphology.binary_opening(fixed_mask,morphology.disk(2)) #Removes hairs
        fixed_mask = morphology.binary_closing(fixed_mask,morphology.disk(3)) #Closes small disconnected areas
        fixed_mask = morphology.remove_small_holes(fixed_mask,100000) #Fills in holes and gaps in the mask
        fixed_mask = morphology.remove_small_objects(fixed_mask,3000) #Removes small clusters outside the main masked area

        #calculate size difference between current mask and original segmentation
        diff = np.abs((np.sum(fixed_mask))/np.sum(seg)) 

        #keep the fixed_mask if size is at least 5% different than the original segmentation
        #mask cannot take 100% of the image size (768x1024) and connot be completely black
        if((diff > 1.05 or diff < 0.95) and np.sum(fixed_mask)!=786432 and np.sum(fixed_mask!=0)) :
            imlist.append(fixed_mask)
            difflist.append(np.abs(1-diff))
            index+=1
            same_counter = 0
        else:
            same_counter += 1

        #break when best mask has been found
        if index >=2:
            if difflist[index-1] > difflist[index-2] or same_counter >= 3:
                break
    
    custom_mask = imlist[difflist.index(min(difflist))] if imlist else seg
    return custom_mask


def create_features_list(im,mask):
    mask = mask.copy() #Allows for easier editing
    
    ### Area of lesion using segment
    nb_pixels = mask.shape[0] * mask.shape[1] 
    area = np.sum(mask)
    
    #Avoids image borders to interfer with perimeter calculations
    mask[0:3,:], mask[-4:-1,:], mask[:,0:3], mask[:,-4:-1] = False,False,False,False
    
    ### Calculation perimeter using a brush
    # Erode the image - eat away at the borders
    mask_eroded = morphology.binary_erosion(mask, morphology.disk(3))
    # As the new area is smaller, the perimeter is calculated by subtracting the og mask from the mask_eroded
    image_perimeter = mask.astype("int32")-mask_eroded
    perimeter = np.sum(image_perimeter)
    
    compactness = (perimeter)**2/(4*math.pi*area)
    
    masked_im = im.copy()
    masked_im[mask==0] = 0
    
    #Crops masked_im to the edges of the mask
    array = mask.copy().astype("int32")
    H,W = array.shape
    left_edges = np.where(array.any(axis=1),array.argmax(axis=1),W+1)
    flip_lr = cv2.flip(array,1) #1 horz vert 0
    right_edges = W - np.where(flip_lr.any(axis=1),flip_lr.argmax(axis=1),W+1)
    top_edges = np.where(array.any(axis=0),array.argmax(axis=0),H+1)
    flip_ud = cv2.flip(array,0) #1 horz vert 0
    bottom_edges = H - np.where(flip_ud.any(axis=0),flip_ud.argmax(axis=0),H+1)
    leftmost = left_edges.min()
    rightmost = right_edges.max()
    topmost = top_edges.min()
    bottommost = bottom_edges.max()
    
    #cropped mask and masked_image
    masked_im = masked_im[topmost:bottommost,leftmost:rightmost,:]
    mask = mask.copy()[topmost:bottommost,leftmost:rightmost]
    
    ### Asymmetry
    #if rotated lesion and lesion are overlapped and there exists a high value of gray, then the lesion is assymetric
    h, w = map(int, mask.shape)
    left = mask[0:, 0:math.floor(w/2)]
    right = mask[0:, math.ceil(w/2):]
    rot_im = rotate(right, 180)         
    new_im = rot_im + left
    new_im[new_im == 2] = 0
    h2, w2 = map(int, new_im.shape)
    top = new_im[0: math.floor(h2/2), 0:]
    bottom = new_im[math.ceil(h2/2):, 0:]
    rot_im2 = rotate(top, 270)
    new_im2 = rot_im2 + bottom
    new_im2[new_im2 == 2] = 0
    
    asymmetry = np.sum(new_im2)/area
    
    #color intensity of lesion area
    colors_of_lesion = masked_im[mask==1]
    x_R, x_G, x_B = np.mean(colors_of_lesion, axis = 0)
    avg_color = (x_R + x_G + x_B)/3
    
    features = [compactness,asymmetry,avg_color]
    return features
    
def normalize_features(features):
    #The values below are the maximum from the ISIC_2017 database
    compactness_max = 13524303.661695108
    asymmetry_max = 0.6949959819161972
    average_color_max = 241.51572327044025
    
    return [features[0]/compactness_max,features[1]/asymmetry_max,features[2]/average_color_max]



def main():
    #Specify image filepath here
    image_filepath = "example.jpg"
    segmentation_filepath = "example_segmentation.png"

    #import and resize image and segmentation
    im = plt.imread(image_filepath)
    im = resize(im, (768, 1024),anti_aliasing=True)
    seg = plt.imread(segmentation_filepath)
    seg = resize(seg, (768, 1024),anti_aliasing=True)

    mask = create_custom_mask(im,seg)
    features = create_features_list(im,mask)

    norm_features = normalize_features(features) #normalize features

    knn_model = joblib.load("knn_trained.joblib")

    probabilities = knn_model.predict_proba(np.asarray(norm_features).reshape(1,-1))
    print("Melanoma Probability:",round(probabilities[0,1]*100,3),"%")

if __name__ == "__main__" : main()