from tkinter import *
from PIL import Image, ImageTk
import skimage as ski
import numpy as np
import pandas as pd
import time

from dataManipulation import *
from end_effector_pose_sorted_PC import subsample_path_and_estimate_poses
"""
last updated: 12th march, 2024
This is the main code to run for the project
When the window pops up (left)click on 3 (or more) points that you want to create a weld-line between. 
Right click on a point that is easy to uniquely identify to create a bounding box for template matching.
To exit the window, either press "q" or close it manually.
The prosess then saves the resulting weld paths to "weld_path1.csv" and "weld_path2.csv" which later 
can be subsampeled to extract a certain amount of points for the weld-path

By setting "runde2" to True, you activate the automation-part, which uses the previous template and weld path, 
and translates it based on the offset between the prevois part and the new one 
"""
runde2:bool = False #NOTE set this to true to only run the short version


#start of functions
def find_nearest_white(img, target):
    target = [target[1], target[0]]
    nonzero = np.argwhere(img == 255)
    distances = np.sqrt((nonzero[:,0] - target[0]) ** 2 + (nonzero[:,1] - target[1]) ** 2)
    nearest_index = np.argmin(distances)
    coordinate = nonzero[nearest_index]
    return [coordinate[1], coordinate[0]]


#####
def get_traversed_image(coords, image_path:str = 'skeleton.png', save:bool=True):
    image = ski.io.imread(image_path, "gray")
    # Define directions for traversing (8-connectivity)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1), (1, 0), (1, 1)]
    # Initialize a visited matrix
    visited = np.zeros_like(image, dtype=np.uint8)
    # Define a function to traverse along white pixels

    def traverse(start:list[int], end:list[int], visited): # this is bugged, currently only works correctly for points from top left towards bottom right, not the other way
        start = (start[0], start[1])
        end = (end[0], end[1])
        stack = [start]
        while stack:
            current = stack.pop()
            if current == end:
                return True
            visited[current[1], current[0]] = 1
            for direction in directions:
                next_pixel = (current[0] + direction[1], current[1] + direction[0])
                if (0 <= next_pixel[0] < image.shape[1] and
                        0 <= next_pixel[1] < image.shape[0] and
                        image[next_pixel[1], next_pixel[0]] == 255 and
                        not visited[next_pixel[1], next_pixel[0]]):
                    stack.append(next_pixel)
        return False
    
    for i in range(len(coords)-1):
        traverse(coords[i], coords[i+1], visited)
    result_image = np.zeros_like(image)
    result_image[np.where(visited == 1)] = 255
    # Save the resulting image
    if save:
        ski.io.imsave("traversed.png", result_image)
    return result_image
#####

def get_actual_cordinates(image,coordinates ):
    actual_coordinates = []
    actual_coordinates.append([find_nearest_white(image, coordinates[0])[0], find_nearest_white(image, coordinates[0])[1]])
    for i in range(len(coordinates)):
        if len(coordinates)-1 > i:
            actual_coordinates.append([find_nearest_white(image, coordinates[i+1])[0],find_nearest_white(image, coordinates[i+1])[1]]) 
    return actual_coordinates

def draw_lines(image, coordinates):
    for i in range(len(coordinates)-1):
        rr, cc = ski.draw.line(r0=coordinates[i][1], c0=coordinates[i][0], 
                               r1=coordinates[i+1][1], c1=coordinates[i+1][0])
        image[rr,cc] = 255
        # print(rr,cc)
    # ski.io.imsave("weld_lines.png", image, check_contrast=False)
    return image



def make_poses_transform_comapatible(poses_df):
    assert poses_df.shape[1] == 3, "The poses are in the wrong form, each pose is not of length 3"
    padding = pd.Series(np.zeros(poses_df.shape[0]))
    return pd.concat([poses_df, padding], axis=1)


def do_subsample_extract_transform(extracted_df, pointcloud, robot_matrix, camera_matrix, angle_offset=15, chosen_point_distance=10, pose_as_quaternion_xyzw:bool = True):

    points, poses = subsample_path_and_estimate_poses(extracted_df, pointcloud, angle_offset, chosen_point_distance)

    points_df = pd.DataFrame(points)
    transformed_points_df = df_transformation(matrix_rob=robot_matrix, matrix_cam= camera_matrix, points= points_df)

    poses_df = pd.DataFrame(poses)
    poses_df_comp = make_poses_transform_comapatible(poses_df) #important to add zero-padding to make the transformation act as a pure rotation on the vector, df_transformation adds ones as padding if the length of each input point is 3 instead of 4
    transformed_poses_df = df_transformation(matrix_rob=robot_matrix, matrix_cam= camera_matrix, points= poses_df_comp)

    transformed_points_df = raw_to_xyz(transformed_points_df)
    transformed_points = np.array(transformed_points_df)

    if not pose_as_quaternion_xyzw:
        transformed_poses_df = raw_to_xyz(transformed_poses_df)
    transformed_poses = np.array(transformed_poses_df)

    comb =[]
    
    for i in range(np.shape(transformed_points)[0]):
        placeholder = np.concatenate((transformed_points[i], transformed_poses[i]), axis=None)
        comb.append(placeholder)

    return np.array(comb)


#####
#########  MATH  ###########

def euclidian_distance(point):
    return np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)


########### Constants #############
    
camera_matrix= np.array([
    [-0.09208491, -0.9944386, -0.05110936, 175.9321],
    [0.9925474, -0.09578168, 0.07533592, -56.53379],
    [-0.07981229, -0.04379116, 0.9958475, -282.3045],
    [0,            0,          0,          1]])
    
pos01 = np.array([  
    [4.98318650e-01, -4.52953633e-01, 7.39264181e-01, 8.75285000e+02],
    [-1.89132288e-01, -8.88932145e-01, -4.17167375e-01, 3.15947000e+02],
    [8.46113173e-01, 6.80635573e-02, -5.28639623e-01, -6.92896000e+02],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

pos02 = np.array([
    [4.20848843e-01, -5.75051039e-01, 7.01571489e-01, 8.90010000e+02],
    [-3.13090458e-01, -8.17950301e-01, -4.82629951e-01, 3.31501000e+02],
    [8.51387466e-01, -1.65410825e-02, -5.24276431e-01, -6.95828000e+02],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

pos03 = np.array([
    [4.22271284e-01, -4.05924079e-01, 8.10501453e-01, 8.11877000e+02],
    [-1.98467283e-01, -9.13837467e-01, -3.54276476e-01, 3.06629000e+02],
    [8.84475947e-01, -1.12572386e-02, -4.66449969e-01, -6.86442000e+02],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


pos04= np.array([
    [4.98318650e-01, -4.39982707e-01, 7.47056719e-01, 8.75285000e+02],
    [-1.89132288e-01, -8.96077331e-01, -4.01589834e-01, 3.15947000e+02],
    [8.46113173e-01, 5.88271573e-02, -5.29746982e-01, -6.92896000e+02],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

robot_matrix = pos04  #robot arm position




# event2canvas = lambda e, c: (c.canvasx(e.x), c.canvasy(e.y))

def main():
    start_time =time.time()

    # df= pd.read_csv('/home/zivid/Zivid/undistorted_results_sample.csv', sep = ',', header= None)
    df= pd.read_csv('Front2.csv', sep = ',', header= None)
    df_read_time_end= time.time()

    get_skeletonized_image_from_pointcloud(df, 
                                           [1944, 1200], 
                                           image_name_in= "/home/zivid/pytorch_env/LineDetection/images/results3.png",
                                           threshold=120, 
                                           save=True)
    skeleton_time_end = time.time()
    ####################################
    img_path = "/home/zivid/pytorch_env/skeleton.png"
    skeleton= ski.io.imread(img_path, as_gray=True)
####### template round 2  #####################
    round2_start_time = time.time()
    if runde2: #change this later to set only this part to activate when re-running on a new image
        paths= ["weld_path1.csv", "weld_path2.csv"]
        apply_template_matching_automation(skel_image= skeleton, 
                                           path_paths=paths, 
                                           df=df, 
                                           matrix_rob=robot_matrix, 
                                           matrix_cam= camera_matrix)
        
        round2_end_time = time.time()

        print()
        print("Template matching:")
        df_read_time = df_read_time_end-start_time
        print(f"{df_read_time = }")

        skeleton_time= skeleton_time_end-start_time
        print(f"{skeleton_time = }")

        round2_time = round2_end_time - round2_start_time
        print(f"{round2_time = }")

        round2_total_time = round2_end_time - start_time
        print(f"{round2_total_time = }")

        exit()
    
    click_start_time= time.time()
    root = Tk()

    # setting up a tkinter canvas
    canvas = Canvas(root)

    # adding the image
    img_path = "/home/zivid/pytorch_env/skeleton.png"  
    img = Image.open(img_path)
    # width, height = img.width, img.height 
    img_tk = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, image=img_tk, anchor="nw")

    # Resize canvas to fit image
    canvas.config(width=img.width, height=img.height)

    # function to be called when mouse is clicked
    _coordinate_holder = []
    _bounding_box_center = []

    def left_click(event):
        # outputting x and y coords to console
        # cx, cy = event2canvas(event, canvas)
        # print ("(%d, %d) / (%d, %d)" % (event.x,event.y,cx,cy))
        _coordinate_holder.append([event.x, event.y])
    

    def right_click(event):
        _bounding_box_center.append([event.x, event.y])
        # _bounding_box_center = [event.x, event.y]

        # print(_bounding_box_center)
    
    def quit(event):
        print("Q has been pressed, exiting window")
        print("WARNING: This will cause the program to crash if insuficcient points are selected")
        root.destroy()
        
    def next(event):
        if len(_coordinate_holder) >=3 and len(_bounding_box_center)>=1:
            print()
            print(f"Sucsess! {len(_coordinate_holder)} points and {len(_bounding_box_center)} box centers extracted")
            print("going to next step")
            root.destroy()
        else:
            print()
            print(f"only {len(_coordinate_holder)} points {len(_bounding_box_center)} box centers chosen")
            print("3 points (left click), and 1 bounding box center (right click) are needed")
            print(f"Please choose the remaining ({max(3-len(_coordinate_holder),0)}) points and ({max(1-len(_bounding_box_center),0)}) box centers ")


    # mouseclick event
    canvas.bind("<ButtonPress-1>", left_click)
    canvas.bind("<ButtonPress-3>", right_click)
    root.bind("<Return>", next)
    root.bind("q", quit)
    # root.bind("<Key>", quit)
### now to create the size of the bounding box and implement it

    canvas.pack()

    root.mainloop()

    #### EXITING THE TKinter LOOP ####
    second_start_time= time.time()
    skeleton = ski.io.imread(img_path, as_gray=True)
    
    coordinate_temp = _coordinate_holder.copy()
    # print(_bounding_box_center)S
    center= _bounding_box_center[0]
    # template, top_left, bot_right= create_template(image=skeleton, center=center, size= 100)
    xy= template_matching_extended(skeleton, center=center, size=100, show= True)
    point = pixel_to_point(xy,df)
    transformed_point = transform_point(robot_matrix, camera_matrix, point)
    save_point(transformed_point)
    #### manipulate the data from button clicks ####

    actual_coordinates= get_actual_cordinates(skeleton, coordinate_temp)
    

    # get_traversed_image(actual_coordinates)



#####################################################THE DOUBLE IMPLEMENTATION should be streamlined, but works for now
    placeholder_image = np.zeros_like(skeleton)
    placeholder_image1 = np.zeros_like(skeleton)
    placeholder_image2 = np.zeros_like(skeleton)
    # print(actual_coordinates[0:2])
    # print(actual_coordinates[1:3])
    coords1 = actual_coordinates[0:2]
    coords2 = actual_coordinates[1:3]
    lines_image = draw_lines(placeholder_image, actual_coordinates)
    lines_image1 = draw_lines(placeholder_image1, coords1)
    lines_image2= draw_lines(placeholder_image2, coords2)
    
    # rr, cc = ski.draw.line(r0=actual_coordinates[1][1], c0=actual_coordinates[1][0], 
    #                            r1=actual_coordinates[2][1], c1=actual_coordinates[2][0])
    # print(placeholder_image2)
    # placeholder_image2[rr,cc] = 255
    # lines_image2 = placeholder_image2
    # print(lines_image2)
    ski.io.imsave("weld_lines.png", placeholder_image, check_contrast=False)
    ski.io.imsave("weld_line1.png", placeholder_image1, check_contrast=False)
    ski.io.imsave("weld_line2.png", placeholder_image2, check_contrast=False)

    # lines_image_sorted= get_traversed_image(actual_coordinates, "weld_lines.png")
    lines_df = pd.DataFrame(lines_image1.flatten())
    lines_df2 = pd.DataFrame(lines_image2.flatten())
    df2 = df.copy()


        ####filter and subsample and pose##########
    df_filtered = filter_df_by_df(df,lines_df)
    df_filtered2 = filter_df_by_df(df2,lines_df2)



    ##  this wil be used!!
    # screw= np.array([])
    # position= screw_to_homogeneus(screw)
    ##



    sorted_lines= sort_linesegments(df_filtered, df_filtered2)
    df_sorted_lines1= pd.DataFrame(sorted_lines[0])
    df_sorted_lines2= pd.DataFrame(sorted_lines[1])

    # df_sorted_lines1= df_transformation(matrix_rob=robot_matrix, matrix_cam= camera_matrix, points= df_sorted_lines1) #pos01, camera_matrix, df_sorted_lines1)
    # df_sorted_lines2= df_transformation(robot_matrix, camera_matrix, df_sorted_lines2)
    df_sorted_lines_comb = pd.concat([df_sorted_lines1, df_sorted_lines2], axis=0, ignore_index=True)




    # output = do_subsample_extract_transform(df_sorted_lines_comb, df, robot_matrix, camera_matrix, angle_offset=15, chosen_point_distance=10, pose_as_quaternion_xyzw= True)
    output = do_subsample_extract_transform(df_sorted_lines_comb, df, robot_matrix, camera_matrix, angle_offset=15, chosen_point_distance=10, pose_as_quaternion_xyzw= True)

    out_df = pd.DataFrame(output)
    out_df.to_csv("OUTPUT.csv", header = None, index = None)


    """
    TODO: make the poses into quaternions (add 0 in front or after?!!)
    output it!!
    
    """
    # print("points:")
    # print(transformed_points)

    # print()

    # print("poses:")
    # print(transformed_poses)
    print("points, poses:")
    print(output)







        #### end of transformation ####
    
    
    # print(df_sorted_lines_comb)
    # df_sorted_lines_comb= df_transformation(pos04, camera_matrix, df_sorted_lines_comb)


    ##################################################### BUG
    # df1= df_sorted_lines1.copy()
    # # coefficients = np.polyfit()
    # poly = np.polynomial.polynomial.polyval3d(df1.iloc[0,:], df1.iloc[1, :], df1.iloc[2, :], df_sorted_lines1.shape[0]) #THIS IS BUGGED!!!

    # # polyval1 = df_sorted_lines1.apply( lambda row: np.polynomial.polynomial.polyval3d(row[0], row[1], row[0], df_sorted_lines1.shape[0]))
    # print(poly)

    ######################################################E ND OF BUG

    df_sorted_lines1.to_csv("weld_path1.csv", header = None, index = None)
    df_sorted_lines2.to_csv("weld_path2.csv", header = None, index = None)
    df_sorted_lines_comb.to_csv("weld_path.csv", header = None, index = None)

    




        #### TRANSFORMATION ####

    df_sorted_lines1= df_transformation(matrix_rob=robot_matrix, matrix_cam= camera_matrix, points= df_sorted_lines1) #pos01, camera_matrix, df_sorted_lines1)
    df_sorted_lines2= df_transformation(robot_matrix, camera_matrix, df_sorted_lines2)
    df_sorted_lines_comb = pd.concat([df_sorted_lines1, df_sorted_lines2], axis=0, ignore_index=True)

    df_sorted_lines1.to_csv("weld_path1_transformed.csv", header = None, index = None)
    df_sorted_lines2.to_csv("weld_path2_transformed.csv", header = None, index = None)
    df_sorted_lines_comb.to_csv("weld_path_transformed.csv", header = None, index = None)

        #### end of transformation ####
    ####### time measurements
    end_time = time.time()
    print()
    df_read_time = df_read_time_end-start_time
    print(f"{df_read_time = }")

    skeleton_time= skeleton_time_end-start_time
    print(f"{skeleton_time = }")

    click_time = second_start_time - click_start_time
    print(f"{click_time = }")

    total_second_part_time = end_time - second_start_time
    print(f"{total_second_part_time = }")

    total_time_with_click = end_time - start_time
    print(f"{total_time_with_click = }")

    total_time_without_click= df_read_time + skeleton_time + total_second_part_time
    print(f"{total_time_without_click = }")


    


    # df_filtered.to_csv("weld_path.csv", header = None, index = None) #["x", "y", "z"]
    # df_filtered2.to_csv("weld_path2.csv", header = None, index = None)


if __name__ == "__main__":
    main()