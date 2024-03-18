import matplotlib.image as img
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

 
# read an image
imageMat = img.imread('out.png')
print(imageMat)
print("Image shape:", 
      imageMat.shape)
 
# # if image is colored (RGB)

# if(imageMat.shape[2] == 3): #probably remove this part later
   
#   # reshape it from 3D matrice to 2D matrice
#   imageMat_reshaped = imageMat.reshape(imageMat.shape[0],
#                                       -1)
#   print("Reshaping to 2D array:", 
#         imageMat_reshaped.shape)
 
# # if image is grayscale
# else:
#   # remain as it is
imageMat_reshaped = imageMat.reshape(-1, 1)
     
# converting it to dataframe.
mat_df = pd.DataFrame(imageMat_reshaped)
 
# exporting dataframe to CSV file.
mat_df.to_csv('hed3.csv',header = None,
              index = None)
 


def image_to_csv(image:img, filename:str= "skeletonized.csv"):
      image = image.reshape(-1, 1)
     
      # converting it to dataframe.
      image_df = pd.DataFrame(image)
      
      # exporting dataframe to CSV file.
      image_df.to_csv(filename,header = None,
              index = None)



# # getting matrice values.
# loaded_2D_mat = loaded_df.values
 




# # reshaping it to 3D matrice
# loaded_mat = loaded_2D_mat.reshape(0)
# original
# loaded_mat = loaded_2D_mat.reshape(loaded_2D_mat.shape[0],
#                                    imageMat.shape[1])
 
# print("Image shape of loaded Image :",
#       loaded_mat.shape)
 
# check if both matrice have same shape or not
# if((imageMat == loaded_mat).all()):
#   print("\n\nYes",
#         "The loaded matrice from CSV file is same as original image matrice")
