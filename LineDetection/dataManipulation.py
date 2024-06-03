import matplotlib.image as img
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
from PIL import Image
import skimage as ski
from scipy import ndimage as ndi
from run2 import run_hed

image_size = [1944, 1200]
# from skimage.morphology import skeletonize


# def hough_transform(image): #leaving this here in case for later use
#     tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
#     h, theta, d = ski.transform.hough_line(image, theta=tested_angles)

#     # Generating figure 1
#     fig, axes = plt.subplots(1, 3, figsize=(15, 6))
#     ax = axes.ravel()

#     ax[0].imshow(image, cmap=cm.gray)
#     ax[0].set_title('Input image')
#     ax[0].set_axis_off()

#     angle_step = 0.5 * np.diff(theta).mean()
#     d_step = 0.5 * np.diff(d).mean()
#     bounds = [np.rad2deg(theta[0] - angle_step),
#               np.rad2deg(theta[-1] + angle_step),
#               d[-1] + d_step, d[0] - d_step]
#     ax[1].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
#     ax[1].set_title('Hough transform')
#     ax[1].set_xlabel('Angles (degrees)')
#     ax[1].set_ylabel('Distance (pixels)')
#     ax[1].axis('image')

#     ax[2].imshow(image, cmap=cm.gray)
#     ax[2].set_ylim((image.shape[0], 0))
#     ax[2].set_axis_off()
#     ax[2].set_title('Detected lines')

#     for _, angle, dist in zip(*ski.transform.hough_line_peaks(h, theta, d)):
#         (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
#         ax[2].axline((x0, y0), slope=np.tan(angle + np.pi/2))

#     plt.tight_layout()
#     plt.show()
#     return


# def raw_to_xyz(filename:str = 'results2.csv'):
#     """
#     input: string containing the filename or path to a csv file,
#         the format of the first three columns needs
#         to be x, y, z, The rest is arbitrary

#     output: a pandas dataframe containing x,y,z, including NaN values.
#         (the first threee colums are returned)
#     """
#     df_pointcloud = pd.read_csv(filename,
#                              sep = ',',
#                              header= None,
#                             #  keep_default_na= True
#                              )
#     df_pts = pd.concat([df_pointcloud.iloc[:, :3]], axis = 1) # keeps only xyz, removes everything else
#     return df_pts

def raw_to_xyz(df:pd.DataFrame):
    """
    input: Pandas Dataframe
        the format of the first three columns needs
        to be x, y, z, The rest is arbitrary

    output: a pandas dataframe containing x,y,z, including NaN values.
        (the first threee colums are returned)
    """
    df = pd.concat([df.iloc[:, :3]], axis = 1) # keeps only xyz, removes everything else
    return df

def raw_to_rgb(df_raw: pd.DataFrame)-> pd.DataFrame:
    """
    input: pandas dataframe containing x,y,z,r,g,b, can be longer
    Output: pandas dataframe containing r,g,b
    !!!It might be bgr right now!!!
    """
    df_rgb= df_raw.iloc[:, [3,4,5]]
    return df_rgb

def threshold_gray(x, val, to_binary:bool = False):
    """
    input: x:int or float,
        val: thresholding value: int or float
        to_binary: bool, that if set to True returns 1 instead of x
    output: 0 if x is lower than threshold, else 1 or x depending on to_binary
    """
    if x <= val:
        return 0
    if to_binary:
        return 1
    else:
        return x


def df_thresholding_hed(df_hed:pd.DataFrame, threshold:int = 155, to_binary:bool = False)-> pd.DataFrame:
    """
    input: pandas dataframe containing the detected edges, threshold value: int
        Assumes values between 0 and 255, but not neccesary
    """
    df_hed = df_hed.round(1)
    df_hed = df_hed.apply(lambda x: threshold_gray(x.item(), threshold, to_binary= to_binary), axis = 1)
    # df_hed = df_hed.drop(df_hed[df_hed.values <= threshold].index)
    return df_hed



def filter_df_by_df(df:pd.DataFrame, df_filter:pd.DataFrame, remove_na:bool = True)-> pd.DataFrame:
    """
    Input: pandas dataframe containing x, y, z coordinates with NaN values
        and pandas dataframe containing pixel-corresponding HED values
    Output: pandas dataframe with x, y, z coordinates that have a
        grayscale value
    """
    assert (len(df) == len(df_filter)), f"ERROR: not same number of rows: length {len(df)} != {len(df_filter)}"
    if df.shape[1] >3:
        print("converting to only xyz-values")
        df= raw_to_xyz(df)
    df = df.drop(df_filter[df_filter.values == 0].index)
    if remove_na:
        df = df.dropna()
    return df

# df_pc_edges = pd.concat([df_pc_stripped.iloc[:, :3], df_edges , df_pc_stripped.iloc[:, 3:]], axis = 1) #keeps all info, changes rgb to grayscale
# df_pc_edges = pd.concat([df_pointcloud.iloc[:, :3], df_edges], axis = 1) #keeps x,y,z and grayscale, removes alfa channel and SNR
# df_pts = pd.concat([df_pointcloud.iloc[:, :3]], axis = 1) # keeps only xyz, removes everything else


# df_pc_edges = pd.concat([df_pc_stripped.iloc[:, :2], df_edges , df_pc_stripped.iloc ])
# df_pc_edges.insert(3, df_edges)


# print(df_pointcloud)

# def save_dataframe(df:pd.DataFrame , name: str ):
#     """
#     input: dataframe to be saved
#         string containing the full name of the file
#         on the form: 'examplename.csv'

#     out: None
#     """
#     df.to_csv(name,header = None, index = None)


# df_pc_edges.to_csv('pc_hed.csv',header = None, index = None)


def filter_thresholding_PIL_low(intensity):
    if intensity <= 50:
        return 0
    return intensity


def image_to_csv(image:img, filename:str= "skeletonized.csv"):
      image = image.reshape(-1, 1)
     
      # converting it to dataframe.
      image_df = pd.DataFrame(image)
      
      # exporting dataframe to CSV file.
      image_df.to_csv(filename,header = None,
              index = None)




####converting back to images to view the results

# array_color = df_color.to_numpy().astype(int)
# df_edges = df_edges.iloc[[df_edges[df_edges.values <= threshold].index], 0] = 0 #droping values below threshold


# df_edges.to_csv('pc_hed.csv',header = None, index = None)
      






def df_to_image(df:pd.DataFrame, size:np.array): #needs to add color or no_color
    """
    """
    if df.shape[1]==3:#color image
        arr = df.to_numpy().astype(np.uint8)
        image = arr.reshape((size[1], size[0],3))
    else: #intensity only
        arr = df.to_numpy().astype(np.uint8)
        image = arr.reshape((size[1], size[0]))
    return image


def skeletonizing_ed(image_path:str = "/home/zivid/pytorch_env/out.png", threshold:int = 140, save:bool= True, name_str="HED"): #100 in threshold has previously been used as a baseline
    """
    threshold is for now simply manually set to reduce background noise.
    returns the thresholded and skeletonized image
    """
    image = ski.io.imread(image_path)
    image_thresh = image > threshold
    # thresh_otsu = ski.filters.threshold_otsu(image2)
    skeleton = ski.morphology.skeletonize(image_thresh)
    if save:
        ski.io.imsave("thresholded_" + name_str + ".png", ski.util.img_as_ubyte(image_thresh))
        ski.io.imsave("skeleton.png", ski.util.img_as_ubyte(skeleton))
    return skeleton




def get_color_img_from_df(df:pd.DataFrame, image_size:list, save:bool=True, save_image_name:str= "/home/zivid/pytorch_env/LineDetection/images/results3.png"):
    """
    optional to save, set to save the file to 'color_image.png' as default
    """
    df_colors = raw_to_rgb(df)
    # df= df.round(0)
    image = df_to_image(df_colors, size=image_size)
    if save:
        ski.io.imsave(save_image_name, image)
    return image


def get_skeletonized_image_from_pointcloud(df:pd.DataFrame, image_size:list, image_name_in:str, save:bool=True, threshold =100):
    image= get_color_img_from_df(df = df, image_size=image_size, save= True, save_image_name=image_name_in)
    skel_path = run_hed(args_strIn=image_name_in)
    skeleton= skeletonizing_ed(image_path=skel_path, threshold=threshold)

def canny(image_name_in):
    # Generate noisy image of a square
    # image = np.zeros((128, 128), dtype=float)
    # image[32:-32, 32:-32] = 1
    image = ski.io.imread(image_name_in, as_gray=True)

    # image = ndi.rotate(image, 15, mode='constant')
    # image = ndi.gaussian_filter(image, 4) # its already included
    # image = random_noise(image, mode='speckle', mean=0.1)

    # applying canny with gaussian sigma = 1, 3, 5
    max_val = np.amax(np.array(image))
    edges1 = ski.feature.canny(image, sigma=0.6, low_threshold=0.2*max_val, high_threshold=0.3*max_val) # hysteresis thresholding
    # edges2 = ski.feature.canny(image, sigma=0.7, low_threshold=0.2*max_val, high_threshold=0.3*max_val)
    # edges3 = ski.feature.canny(image, sigma=0.8, low_threshold=0.2*max_val, high_threshold=0.3*max_val)
    # edges4 = ski.feature.canny(image, sigma=0.9, low_threshold=0.2*max_val, high_threshold=0.3*max_val)

    ski.io.imsave("canny_sigma06-02.png", edges1)
    # ski.io.imsave("canny_sigma07-02.png", edges2)
    # ski.io.imsave("canny_sigma08-02.png", edges3)
    # ski.io.imsave("canny_sigma09-02.png", edges4)
    return ["canny_sigma1.png", "canny_sigma1.png", "canny_sigma3.png", "canny_sigma5.png"]

def get_skeletonized_image_from_pointcloud_canny(df:pd.DataFrame, image_size:list, image_name_in:str, save:bool=True, threshold =100):
    image= get_color_img_from_df(df = df, image_size=image_size, save= True, save_image_name=image_name_in)

    skel_path = canny(image_name_in)#run_hed(args_strIn=image_name_in)

def sort_linesegments(line1:pd.DataFrame, line2:pd.DataFrame):
    """
    
    """
    lines= [np.array(line1), np.array(line2)]
    #first line:
    sorted_lines= []
    first= lines[0]
    second = lines[1]
    if first[0].all()== second[0].all() or first[0].all() ==second[-1].all():
        sorted_lines.append(first[::-1])
    else:
        sorted_lines.append(first)
    for i in range(len(lines)-1):
        if sorted_lines[i][-1].all() == lines[i+1][0].all():
            sorted_lines.append(lines[i+1])
        else: #reverse the order of the next line (this is done blindly), should add check if they corrolate, and maybe a "closest" if none are the same
            sorted_lines.append(lines[i+1][::-1])
    return sorted_lines


def combined_image(color_image, overlay_image1, overlay_image2, wide:bool=True):
    image = color_image.copy()
    size= 1
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1), (1, 0), (1, 1)]
    xy1= np.column_stack(np.where(overlay_image1>0))
    xy2= np.column_stack(np.where(overlay_image2>0))
    if wide:
        for i in range(size):
            for point in directions:
                image[ xy1[:,0]+point[0]*(i+1), xy1[:,1]+point[1]*(i+1) ,:]= [0, 0, 204]
                image[ xy2[:,0]+point[0]*(i+1), xy2[:,1]+point[1]*(i+1) ,:]= [204, 0, 0]
    else:
        image[ xy1[:,0], xy1[:,1] ,:]= [0, 0, 204]
        image[ xy2[:,0], xy2[:,1] ,:]= [204, 0, 0]
    return image


########  creating plots  ########
def create_combined_layer_visualization():
    color_image = ski.io.imread("/home/zivid/pytorch_env/LineDetection/images/results3.png")
    skeleton= ski.io.imread("skeleton.png")
    lines = ski.io.imread("weld_lines.png")
    print(color_image.shape)
    print(skeleton.shape)
    comb= combined_image(color_image, skeleton, lines)
    comb_thin = combined_image(color_image, skeleton, lines, wide=False)
    ski.io.imsave("combined_image_thin.png", comb_thin)
    ski.io.imsave("combined_image_thick.png", comb)
    return
#######  end of plots  ###########


######## Transformation ##########

def df_transformation(matrix_rob:np.ndarray, matrix_cam:np.ndarray, points:pd.DataFrame):
    """
    Takes in two matrixe, robot and camera, and the dataframe containing the position vectors
    """
    comb_matrix = matrix_rob @ matrix_cam
    def multiply(matrix, row):
        return pd.Series(matrix @ row)
    
    if points.shape[1] == comb_matrix.shape[0]-1:
        padding = pd.Series(np.ones(points.shape[0]))
        points = pd.concat([points, padding], axis=1)


    points = points.apply(lambda row: multiply(comb_matrix, row), axis= 1)

    # points = points.drop(index= 4, axis = 1) #drops the "1" padding, does not work
    return points.round(4)

# def translation(vector:list, point:list):
#     return [v + p for v, p in zip(vector, point)]

def translation(vector:list, point:pd.DataFrame):
    return point + vector
def df_translation(vector:list, points:pd.DataFrame) -> pd.DataFrame:
    """
    takes in a vector containing the offset and a dataframe the vector should be applied to
    returnes the dataframe with the offset compensated for
    """
    # if isinstance(vector, np.array)
    if points.shape[1] > len(vector):
        points= raw_to_xyz(points)
    # else:
    #     print(f"didnt reshape:{points.shape()}")
    points =  points.apply(lambda row: translation(vector, row), axis= 1)
    return points

# Convert degrees to radians


# Function to prompt user for input
# def get_input(prompt):
#     return float(input(prompt))

# # Prompt user for position (translation) in mm along x, y, and z axes
# position_x = get_input("Enter position along X-axis (in mm): ")
# position_y = get_input("Enter position along Y-axis (in mm): ")
# position_z = get_input("Enter position along Z-axis (in mm): ")

# # Prompt user for rotations in degrees along x, y, and z axes
# rotation_x_deg = get_input("Enter rotation angle around X-axis (in degrees): ")
# rotation_y_deg = get_input("Enter rotation angle around Y-axis (in degrees): ")
# rotation_z_deg = get_input("Enter rotation angle around Z-axis (in degrees): ")

# # Convert degrees to radians
# rotation_x = deg2rad(rotation_x_deg)
# rotation_y = deg2rad(rotation_y_deg)
# rotation_z = deg2rad(rotation_z_deg)

# # Define rotation matrices
# Rx = np.array([[1, 0, 0],
#                [0, np.cos(rotation_x), -np.sin(rotation_x)],
#                [0, np.sin(rotation_x), np.cos(rotation_x)]])

# Ry = np.array([[np.cos(rotation_y), 0, np.sin(rotation_y)],
#                [0, 1, 0],
#                [-np.sin(rotation_y), 0, np.cos(rotation_y)]])

# Rz = np.array([[np.cos(rotation_z), -np.sin(rotation_z), 0],
#                [np.sin(rotation_z), np.cos(rotation_z), 0],
#                [0, 0, 1]])

# # Combine rotation matrices
# R = np.dot(np.dot(Rz, Ry), Rx)

# # Create homogeneous transformation matrix
# T = np.eye(4)   
# T[:3, :3] = R
# T[:3, 3] = [position_x, position_y, position_z]

# output = np.array2string(T, separator=', ', formatter={'all':lambda x: '{:.8e}'.format(x)})
# print("Homogeneous Transformation Matrix:")
# print(output)

def screw_to_homogeneus(screw:np.array):
    """
    screw axis from robot end-effector to second to last joint, 
    is given as a vector of rotation and then translation
    where the rotation is in degrees, NOT radians!!
    """
    def deg2rad(deg):
        return deg * np.pi / 180.0

    position_x= screw[0]
    position_y= screw[1]
    position_z= screw[2]

    rotation_x_deg= screw[3]
    rotation_y_deg= screw[4]
    rotation_z_deg= screw[5]



    rotation_x = deg2rad(rotation_x_deg)
    rotation_y = deg2rad(rotation_y_deg)
    rotation_z = deg2rad(rotation_z_deg)
    
    Rx = np.array([[1, 0, 0],
               [0, np.cos(rotation_x), -np.sin(rotation_x)],
               [0, np.sin(rotation_x), np.cos(rotation_x)]])

    Ry = np.array([[np.cos(rotation_y), 0, np.sin(rotation_y)],
               [0, 1, 0],
               [-np.sin(rotation_y), 0, np.cos(rotation_y)]])

    Rz = np.array([[np.cos(rotation_z), -np.sin(rotation_z), 0],
               [np.sin(rotation_z), np.cos(rotation_z), 0],
               [0, 0, 1]])
    
    R = np.dot(np.dot(Rz, Ry), Rx)

    T = np.eye(4)   
    T[:3, :3] = R
    T[:3, 3] = [position_x, position_y, position_z]

    return np.array(T)
######## T-END ###############


#########  Bounding Box  ############
def create_bounding_box(center:list, size:int=None):
    if not size:  #checking if there is enterede a value
        size = 50
    else:
        size=size//2
    print(f"center is: {center}")
    x= center[0]
    y= center[1]
    top_left= [max(x-size, 0), max(y-size, 0)]
    bot_right= [min(x+size, image_size[0]-1), min(y+size, image_size[1]-1)]
    return top_left, bot_right

def create_template(image:np.ndarray, center:list, iter,  size:int=None):
    top_left, bot_right= create_bounding_box(center= center, size=size)
    print(f"top_left: {top_left}")
    print(f"bot_right: {bot_right}")
    template = image[top_left[1]: bot_right[1], top_left[0]:bot_right[0]]
    ski.io.imsave("template"+str(iter) + ".png", template)
    return template

#########  Bounding Box End  ##########

#########  Template Matching  #########
def match_template(template:np.array, image:np.array) -> list[int]:
    result = ski.feature.match_template(image, template=template, pad_input=True)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]
    return [x,y]

def template_matching_extended(image:np.ndarray, center:list, iter:str, size:int=None, show:bool=False):
    template= create_template(image=image, center=center, iter=iter, size=size)
    result = ski.feature.match_template(image, template=template, pad_input=True)
    result[result<0]=0
    result= result*255
    result= result.round(0).astype(np.int32)

    # template_y, template_x = template.shape
    # back_y, back_x= backboard.shape
    # backboard[template_y//2: back_y-template_y//2, template_x//2: back_x-template_x//2]=result
    # backboard[top_left[1]: bot_right[1], top_left[0]:bot_right[0]]

    # ski.io.imsave("template_result"+str(iter)+".png", result) dosent show anything usefull unless plotted
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]
    if show:
        print(f"result shape: {result.shape}")
        print(f"max value: {np.max(result)}")
        print(f"min value: {np.min(result)}")

        print(f"result: {result}")
        print(f"ij: {ij}")
        print(f"x:{x}")
        print(f"y:{y}")
        fig = plt.figure(figsize=(8, 3))
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)
        ax1.imshow(template, cmap=plt.cm.gray)
        ax1.set_axis_off()
        ax1.set_title('template')
        ax2.imshow(image, cmap=plt.cm.gray)
        ax2.set_axis_off()
        ax2.set_title('image')
        htemplate, wtemplate = template.shape
        rect = plt.Rectangle((x-wtemplate//2, y-htemplate//2), wtemplate, htemplate, edgecolor='r', facecolor='none')
        ax2.add_patch(rect)
        ax2.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
        ax3.imshow(result, cmap=plt.cm.gray)
        ax3.set_axis_off()
        ax3.set_title('`match_template`\nresult')
        # highlight matched regiondef match_template(template:np.array[np.array[int]], image:np.array[np.array[int]]):
    result = ski.feature.match_template(image, template=template, pad_input=True)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]
    return [x,y]

    # ski.io.imsave("template_result.png", result)
#TODO #continue this (check for mistakes) and continue automating the procedure( tranform the point and create a tranformation between it and a new point from the next image)
def pixel_to_point(pixel_loc:list, df:pd.DataFrame):
    """
    Inputs:
    location of a pixel (from a image extracted from/corrolating to the datafame!)
    Dataframe to match the pixcel with

    Outputs:
    x,y,z coordinates of the real world location corresponding to the pixel
    """
    x, y = pixel_loc
    array_pos= y* image_size[0] + x #do tests on this logic!!!!
    print(f"aray position: {array_pos}")
    coords= df.iloc[array_pos, 0:3]
    # print(f"{np.array([coords]) = }")
    return np.array(coords)

def transform_point(matrix_rob:np.ndarray, matrix_cam:np.ndarray, point:list):
    """
    Takes in two matrices, robot and camera, and the dataframe containing the position vectors
    """
    point= np.append(point, 1)
    df_point= pd.DataFrame(np.array([point]))
    df_point= df_transformation(matrix_rob=matrix_rob, matrix_cam=matrix_cam, points=df_point)
    df_point = raw_to_xyz(df_point)
    return np.array(df_point)

def pixel_to_transformed_point(pixel:list, df:pd.DataFrame, matrix_rob:np.ndarray, matrix_cam:np.ndarray):
    point= pixel_to_point(pixel, df)
    return transform_point(matrix_rob=matrix_rob, matrix_cam=matrix_cam, point=point)[0]
    

def save_point(point:list, name:str= "template_point.csv"):
    df= pd.DataFrame(point)
    df.to_csv(name, header = None, index = None)
    return

        #### FUNCTIONS FOR SECOND PART OF THE TEMPLATE MATCHING ####
#TODO: 
"""
1: open the file with the template coordinate.
2: run a new template matching with the new image 
    - NOTE: this will not be saved, only the original template will be used to avoid graudal degregation of the result.
3: compare the points from the origina and new template matching.
    find the vector between them
4: use the vector to translate the old path (weld_path1.csv and weld_path2.csv) 
    to the new position given by the vector.
"""
def load_point(point_path:str= "template_point.csv") -> pd.DataFrame :
    point= pd.read_csv(point_path, header=None)
    return np.array(raw_to_xyz(point))[0]

def load_template(template_path:str="template.png") -> np.ndarray:
    return ski.io.imread(template_path)

# def match_template(template:np.array[np.array[int]], image:np.array[np.array[int]]):
#     result = ski.feature.match_template(image, template=template, pad_input=True)
#     ij = np.unravel_index(np.argmax(result), result.shape)
#     x, y = ij[::-1]
#     return [x,y]

def offset_vec_from_points(point1:list[int], point2:list[int]) -> list[int]:
    return [p2 - p1 for p2, p1 in zip(point2, point1)]

# def apply_template_matching_automation(skel_image:np.array, 
#                                        path_paths:list[str], 
#                                        df:pd.DataFrame, 
#                                        matrix_rob:np.ndarray, 
#                                        matrix_cam:np.ndarray) -> None : #This does currently not work due to what used to be one template point has been changed to 3.
#     """
#     path_paths is a list with each element being the path to a csv file, containing the welding path to be offset_compensated
#     """
#     saved_point= load_point("template_point.csv")
#     print(f"{saved_point = }")
#     template= load_template("template.png")
#     new_pixel= match_template(template=template, image=skel_image)
#     new_point= pixel_to_transformed_point(new_pixel, df=df, matrix_rob=matrix_rob, matrix_cam=matrix_cam)
#     print(f"{new_point = }")
#     offset_vec= offset_vec_from_points(point1=saved_point, point2=new_point)
#     print(f"{offset_vec = }")
#     for path_name in path_paths:
#         weld_path= pd.read_csv(path_name, header=None)
#         weld_path_new= df_translation(vector=offset_vec,points=weld_path)
#         weld_path_new.to_csv("offest_" + path_name, header=None, index=None)

#         # weld_path= translation


def apply_template_matching_automation(skel_image:np.array, 
                                       template_path_paths: list[str],
                                       weld_path_paths:list[str], 
                                       df:pd.DataFrame, 
                                       matrix_rob:np.ndarray, 
                                       matrix_cam:np.ndarray) -> None : #This does currently not work due to what used to be one template point has been changed to 3.
    """
    path_paths is a list with each element being the path to a csv file, containing the welding path to be offset_compensated
    """
    saved_points= load_point("template_point.csv")
    print(f"{saved_points = }")


    # template0= load_template("template0.png")
    # template1= load_template("template1.png")
    # template2= load_template("template2.png")

    templates = []
    new_points = []
    for i in range(len(template_path_paths)):
        template = load_template(template_path_paths[i])
        templates.append(template)
        new_pixel = match_template(template=template, image=skel_image)
        new_points.append(pixel_to_transformed_point(new_pixel, df=df, matrix_rob=matrix_rob, matrix_cam=matrix_cam))


    print(f"{new_points = }")

    offset_vec= offset_vec_from_points(point1=saved_points[1], point2=new_points[1]) #this is the offset for the middle point, might be useless
    print(f"{offset_vec = }")

    for path_name in path_paths:
        weld_path= pd.read_csv(path_name, header=None)
        weld_path_new= df_translation(vector=offset_vec,points=weld_path)
        weld_path_new.to_csv("offest_" + path_name, header=None, index=None)

        # weld_path= translation

#########  Template Matching end #########


def main() -> None:
    print(f"{load_point() = }")
    list1= [5, 4, 3]
    list2= [[1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6]]
    df=pd.DataFrame(list2)
    print(f"{df = }")
    # print()
    # print(offset_vec_from_points(list1, list2))
    # print(translation(list1, list2))

    trns= df_translation(list1, df)
    print(f"{trns = }")
    pass
    # df= pd.read_csv('weld_path.csv', sep = ',', header= None)
    # array = np.array(df)
    # array = df.to_numpy(dtype=float)


    # ski.io.imshow(comb)
    # ski.io.show()
    
    # print(arr)
    # min_val= min(min(df.iloc[0]), min(df.iloc[1]))
    # print(min(min(df.iloc[0]), min(df.iloc[1])))
    # df= df - min_val
    # print(min(min(df.iloc[0]), min(df.iloc[1])))
    # df= df + min_val
    # print(min(min(df.iloc[0]), min(df.iloc[1])))

    # image = ski.io.imread("/home/zivid/pytorch_env/LineDetection/images/results3.png")
    # print(image)
    # get_color_img_from_df(df, [1200, 1920])
    # image= get_color_img_from_df(df, [1200, 1920, 3], True, save_image_name="/home/zivid/pytorch_env/LineDetection/images/results4.png")

    # get_skeletonized_image_from_pointcloud(df, [1200, 1920], True, image_name_in= "/home/zivid/pytorch_env/LineDetection/images/results2.png")




    #loading the images
    # image_hed = img.imread('out.png')
    # image_result = img.imread('LineDetection/images/results3.png') # used to get the shapes we want

    # reading CSV file s
    # df_edges = pd.read_csv('hed3.csv', sep = ',', header = None)
    # df_pointcloud = pd.read_csv('results3.csv', sep = ',', header= None)
    #setting values:
    # gray_scaling_0_1_to_255 = 255
    # threshold = 170
    # img_shape = np.array(image_result.shape)
    # df_edges = df_edges * gray_scaling_0_1_to_255

    #function calls:
    # xyz= raw_to_xyz(df_pointcloud)
    # hed = thresholding_hed(df_edges, threshold)
    # color = raw_to_rgb(df_pointcloud)
    # xyz_clean = cleaning_by_hed(xyz, hed)

    # save_dataframe(xyz, "test_xyz.csv")
    # save_dataframe(color, "test_color.csv")
    # save_dataframe(xyz_clean, "test_xyz_ceaned.csv")

    # thresholded_hed_img = df_to_image(hed, img_shape)
    # plt.imshow(thresholded_hed_img, cmap= 'gray')
    # plt.show()


    # threshold_gray_low = 160

    # image = Image.open("/home/zivid/pytorch_env/out.png")
    # # image = image.filter(filter_thresholding_PIL_low)

    # image = image.point(lambda p: 0 if p <= threshold_gray_low else p)
    # image_bw = image.point(lambda p: 0 if p <= 10 else 1) 


    # image2 = ski.io.imread("/home/zivid/pytorch_env/out.png")
    # image2 = image2 > 10
    # thresh = ski.filters.threshold_otsu(image2)
    # skeleton = ski.morphology.skeletonize(image2> 160)
    # ski.io.imsave("skeleton.png", skeleton)
    # skeleton = ski.skeletonize(image_bw)

    # plt.bar(np.arange(256),hist)
    # plt.show()

# ###########
#     minLineLength = 100
#     maxLineGap = 10
#     lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap) 
#         for x1,y1,x2,y2 in lines[0]:
# 	        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
#     cv2.imwrite(‘houghlines5.jpg’,img)
# #############
    # image= skeletonizing_ED(save=False)



    # image = ski.io.imread("traversed.png", "gray")
    # image_to_csv(image, filename="traversed.csv")
    # traversed_df = pd.read_csv('traversed.csv', sep = ',', header = None)
    # df_pointcloud = pd.read_csv('results3.csv', sep = ',', header= None)
    
    # df = cleaning_by_hed(df_pointcloud, traversed_df)
    # save_dataframe(df, "cleaned_traversed.csv")



    # # tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 1080, endpoint=False)
    # lines = ski.transform.probabilistic_hough_line(skel, threshold=1, line_length=10,line_gap=5) #, theta = tested_angles )
    # for line in lines:
    #         p0, p1 = line
    #         plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
    # plt.show()
# Constructing test image
    # image = np.zeros((200, 200))
    # idx = np.arange(25, 175)
    # image[idx, idx] = 255
    # image[ski.draw.draw_line(45, 25, 25, 175)] = 255
    # image[ski.draw.draw_line(25, 135, 175, 155)] = 255

    # Classic straight-line Hough transform
    # Set a precision of 0.5 degree.


if __name__ == "__main__":
    main()
