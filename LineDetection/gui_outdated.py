from tkinter import *
from PIL import Image, ImageTk
import skimage as ski
import numpy as np
import pandas as pd
from dataManipulation import get_skeletonized_image_from_pointcloud
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



event2canvas = lambda e, c: (c.canvasx(e.x), c.canvasy(e.y))

if __name__ == "__main__":

    df= pd.read_csv('Side4v2.csv', sep = ',', header= None)
    get_skeletonized_image_from_pointcloud(df, [1200, 1944], True, image_name_in= "/home/zivid/pytorch_env/LineDetection/images/results2.png")

    ####################################
    root = Tk()

    # setting up a tkinter canvas
    canvas = Canvas(root)

    # adding the image
    img_path = "/home/zivid/pytorch_env/skeleton.png"  # assuming the image is located one folder above
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
    actual_coordinates = []
    actual_coordinates.append([find_nearest_white(skeleton, coordinate_temp[0])[0], find_nearest_white(skeleton, coordinate_temp[0])[1]])
    for i in range(len(coordinate_temp)):
        if len(coordinate_temp)-1 > i:
            actual_coordinates.append([find_nearest_white(skeleton, coordinate_temp[i+1])[0],find_nearest_white(skeleton, coordinate_temp[i+1])[1]]) 
            # print(actual_coordinates[i], coordinate_temp[i])

    # print(f"temp coordinates: {coordinate_temp}")
    print(f"actual coordinates: {actual_coordinates}")

    for i in range(len(coordinate_temp)):
        if len(coordinate_temp)-1 > i:

            rr, cc = ski.draw.line(r0=actual_coordinates[i][1], c0=actual_coordinates[i][0],
                       r1=actual_coordinates[i+1][1], c1=actual_coordinates[i+1][0])
            skeleton[rr, cc] = 255


    ski.io.imshow(skeleton, cmap="gray")
    ski.io.show()

    get_traversed_image(actual_coordinates)