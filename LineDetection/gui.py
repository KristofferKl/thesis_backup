from tkinter import *
from PIL import Image, ImageTk
import skimage as ski
import numpy as np
import pandas as pd
from dataManipulation import get_skeletonized_image_from_pointcloud, filter_df_by_df, sort_linesegments, screw_to_homogeneus
#starting to add changes 
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


#####
#########  MATH  ###########

def euclidian_distance(point):
    return np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)

def df_transformation(matrix_rob:np.ndarray, matrix_cam:np.ndarray, points:pd.DataFrame):
    """
    Takes in two matrixe, robot and camera, and the dataframe containing the position vectors
    """
    def multiply(matrix, row):
        return pd.Series(matrix @ row)
    
    padding = pd.Series(np.ones(points.shape[0]))
    points = pd.concat([points, padding], axis=1)

    comb_matrix = matrix_rob @ matrix_cam

    points = points.apply(lambda row: multiply(comb_matrix, row), axis= 1)

    points.drop(index= 4, axis = 1) #drops the "1" padding
    return points
    

def df_translation(vector:list, points:pd.DataFrame):
    """
    takes in a vector containing the offset and a dataframe the vector should be applied to
    returnes the dataframe with the offset compensated for
    """
    return points.apply(lambda row: row*vector, axis= 0)
    




event2canvas = lambda e, c: (c.canvasx(e.x), c.canvasy(e.y))

def main():

    # df= pd.read_csv('/home/zivid/Zivid/undistorted_results_sample.csv', sep = ',', header= None)
    df= pd.read_csv('Front2.csv', sep = ',', header= None)

    get_skeletonized_image_from_pointcloud(df, 
                                           [1944, 1200], 
                                           image_name_in= "/home/zivid/pytorch_env/LineDetection/images/results3.png",
                                           threshold=120, 
                                           save=True)

    ####################################
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

    def printcoords(event):
        # outputting x and y coords to console
        cx, cy = event2canvas(event, canvas)
        # print ("(%d, %d) / (%d, %d)" % (event.x,event.y,cx,cy))
        _coordinate_holder.append([event.x, event.y])

    # mouseclick event
    canvas.bind("<ButtonPress-1>", printcoords)

    canvas.pack()

    root.mainloop()
    skeleton = ski.io.imread(img_path, as_gray=True)
    
    coordinate_temp = _coordinate_holder.copy()
    # print(coordinate_temp)

    # rr, cc = ski.draw.line(r0=coordinate_holder[-2][1], c0=coordinate_holder[-2][0], #use this if only one line is wanted
                    #    r1=coordinate_holder[-1][1], c1=coordinate_holder[-1][0])
    
    # actual_coordinates = []
    # actual_coordinates.append([find_nearest_white(skeleton, coordinate_temp[0])[0], find_nearest_white(skeleton, coordinate_temp[0])[1]])
    # for i in range(len(coordinate_temp)):
    #     if len(coordinate_temp)-1 > i:
    #         actual_coordinates.append([find_nearest_white(skeleton, coordinate_temp[i+1])[0],find_nearest_white(skeleton, coordinate_temp[i+1])[1]]) 
    #         # print(actual_coordinates[i], coordinate_temp[i])
    actual_coordinates= get_actual_cordinates(skeleton, coordinate_temp)

    # print(f"temp coordinates: {coordinate_temp}")
    # print(f"actual coordinates: {actual_coordinates}")

    # for i in range(len(coordinate_temp)):
    #     if len(coordinate_temp)-1 > i:

    #         rr, cc = ski.draw.line(r0=actual_coordinates[i][1], c0=actual_coordinates[i][0],
    #                    r1=actual_coordinates[i+1][1], c1=actual_coordinates[i+1][0])
    #         skeleton[rr, cc] = 255


    # ski.io.imshow(skeleton, cmap="gray")
    # ski.io.show()

    # get_traversed_image(actual_coordinates)
#####################################################THE DOUBLE IMPLEMENTATION DOSENT WORK YET
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


        #### TRANSFORMATION ####
    
        #use this for input from robot#
    """
    screw= np.array([])  #fill array with robot coordinates when taking image
    position= screw_to_homogeneus(screw)
    """
    df_filtered = filter_df_by_df(df,lines_df)
    df_filtered2 = filter_df_by_df(df2,lines_df2)

    #conastant offset-matrix between the camera and the end-effector
    camera_matrix= np.array([
        [-0.09208491, -0.9944386, -0.05110936, 175.9321],
        [0.9925474, -0.09578168, 0.07533592, -56.53379],
        [-0.07981229, -0.04379116, 0.9958475, -282.3045],
        [0,            0,          0,          1]])
    

    #used temporary for testing until automation can be done
    pos04= np.array([
    [4.98318650e-01, -4.39982707e-01, 7.47056719e-01, 8.75285000e+02],
    [-1.89132288e-01, -8.96077331e-01, -4.01589834e-01, 3.15947000e+02],
    [8.46113173e-01, 5.88271573e-02, -5.29746982e-01, -6.92896000e+02],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


    position = pos04  #robot arm position, used temporary until conection to robot can automate it

    sorted_lines= sort_linesegments(df_filtered, df_filtered2)
    df_sorted_lines1= pd.DataFrame(sorted_lines[0])
    df_sorted_lines2= pd.DataFrame(sorted_lines[1])
    df_sorted_lines_comb = pd.concat([df_sorted_lines1, df_sorted_lines2], axis=0, ignore_index=True)
    #there might be a faster way of doing the matrix-multipications, look into c++ implementation or builtin functions
    df_sorted_lines1= df_transformation(matrix_rob=position, matrix_cam= camera_matrix, points= df_sorted_lines1)
    df_sorted_lines2= df_transformation(position, camera_matrix, df_sorted_lines2)

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

    


    # df_filtered.to_csv("weld_path.csv", header = None, index = None) #["x", "y", "z"]
    # df_filtered2.to_csv("weld_path2.csv", header = None, index = None)


if __name__ == "__main__":
    main()