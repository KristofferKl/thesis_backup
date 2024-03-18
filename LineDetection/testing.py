from tkinter import *
from PIL import Image, ImageTk
import skimage as ski
import numpy as np
import pandas as pd
from dataManipulation import get_skeletonized_image_from_pointcloud, filter_df_by_df, sort_linesegments    


# camera_matrix = np.array([[-0.08859395, -0.995804, -0.0229247, 175.7406],
#                      [0.9922861, -0.09023783, 0.08500189, -59.54109],
#                      [-0.0867139, -0.01521721, 0.9961171, -286.7653],
#                      [0, 0, 0, 1]])

df1 = pd.DataFrame([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]])
# print(df1)

camera_matrix= [[-0.09208491, -0.9944386, -0.05110936, 175.9321],
                [0.9925474, -0.09578168, 0.07533592, -56.53379],
                [-0.07981229, -0.04379116, 0.9958475, -282.3045],
                [0, 0, 0, 1]]

pos01 = np.array([[4.98318650e-01, -4.52953633e-01, 7.39264181e-01, 8.75285000e+02],
             [-1.89132288e-01, -8.88932145e-01, -4.17167375e-01, 3.15947000e+02],
             [8.46113173e-01, 6.80635573e-02, -5.28639623e-01, -6.92896000e+02],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

pos02 = np.array([[4.20848843e-01, -5.75051039e-01, 7.01571489e-01, 8.90010000e+02],
             [-3.13090458e-01, -8.17950301e-01, -4.82629951e-01, 3.31501000e+02],
             [8.51387466e-01, -1.65410825e-02, -5.24276431e-01, -6.95828000e+02],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

pos03 = np.array([[4.22271284e-01, -4.05924079e-01, 8.10501453e-01, 8.11877000e+02],
             [-1.98467283e-01, -9.13837467e-01, -3.54276476e-01, 3.06629000e+02],
             [8.84475947e-01, -1.12572386e-02, -4.66449969e-01, -6.86442000e+02],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])



mat1= np.array([[1,2,3,4],
                [5,6,7,8],
            [9,10,11,12],
            [0, 0, 0, 1]])

mat2= np.array([[13, 14,15,16],
               [17, 18, 19, 20],
               [21, 22, 23, 24],
               [0, 0, 0, 1]])

def df_transformation(matrix_rob:np.ndarray, matrix_cam:np.ndarray, points:pd.DataFrame):
    """
    Takes in two matrixe, robot and camera, and the dataframe containing the position vectors
    """
    def multiply(matrix, row):
        # print(f"row: {row}")
        return pd.Series(matrix @ row)

    padding = pd.Series(np.ones(points.shape[0]))
    points = pd.concat([points, padding], axis=1)
    # print(points)
    comb_matrix = matrix_rob @ matrix_cam
    print(points)

    points = points.apply(lambda row: multiply(comb_matrix, row), axis= 1)
    # print(points)
    # print(points)
    # print(comb_matrix)

    points = points.drop(index= 3, axis = 0) #drops the "1" padding
    # print(points)

    return points



df= pd.read_csv("weld_path.csv")
df= df.head(10)
# print(df)
    
# dfT= df_transformation(matrix_rob=pos01, matrix_cam= camera_matrix, points= df)
alla= df_transformation(matrix_rob=mat1, matrix_cam=mat2,points=df1 )
# print(mat1@mat2@[1,2,3,1])
print(alla)
# print(dfT)