
import skimage as ski
import numpy as np
import pandas as pd


class Node:
    def __init__(self, point:list) -> None:
        self.x= point[0]
        self.y= point[1]
        self.z= point[2]
        # self.d = float("inf")
        self.parent = None
        self.child = None
        self.finished = False

    def distance(self, Node):
        return np.sqrt((self.x - Node.x)**2 + (self.y - Node.y)**2 + (self.z - Node.z)**2)

# def euclidian_distance(point):
#     if len(point)==3:
#         return np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
#     if len(point)==2:
#         return np.sqrt(point[0]**2 + point[1]**2)
#     else:
#         print("not correct input, should be array containing two or three values (x,y) or (x,y,z)")
#         return None
def df_to_nodes(df:pd.DataFrame):
    array = df.to_numpy()
    


    
def traverse_graph(graph, start, end):
    nodes = {}










df= pd.read_csv('weld_path.csv', sep = ',', header= None)

offset= min(min(df.iloc[0]), min(df.iloc[1]))

df_filtered_offset= df-offset

def sort_linesegments(lines:np.array):
    """
    lines is supposed to be a list containing each lines as separate lists
    """
    #first line:
    sorted_lines= []
    first= lines[0]
    second = lines[1]
    first_first= first[0]
    last_first = first[-1]

    if first[0]== second[0] or first[0] ==second[-1]:
        sorted_lines.append(first[::-1])
    else:
        sorted_lines.append(first)
    for i in range(len(lines)-1):
        if sorted_lines[i][-1] == lines[i+1][0]:
            sorted_lines.append(lines[i+1])
        else: #reverse the order of the next line (this is done blindly), should add check if they corrolate, and maybe a "closest" if none are the same
            sorted_lines.append(lines[i+1][::-1])
    return sorted_lines


        



# df_sorted_filtered_offset = df_filtered_offset.sort_values(by="x")
# df_sorted= df_sorted_filtered_offset + offset
# df_sorted.to_csv("weld_path.csv", header = None, index = None)