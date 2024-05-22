import numpy as np

# x,y,z = 1,1,1 #placeholder

NPOINTS_IN = 20 #number of points to choose on a consistent basis, this excludes the start and end points as well as the detected weld points
threshold_weld_IN = 0.342 #corresponds to 20 degrees change, might be too big, 0.1763 corresponds to 10 degrees offset, might be better
threshold_noise_IN = 0.966 #corresponds to 75 degrees, anything within this area is most likely some sort of noise, not an actual point
points_IN = [[x,y,z],[x,y,z],[x,y,z], ...] # used to show how ther points are
subsampl_IN = []

def mean(a,b):
    return [(a[0]+ b[0])/2, (a[1] +b[1])/2, (a[2], b[2])/2]

def subsample_points(points:list[list], NPOINTS, threshold_weld, threshold_noise):
    subsample = []
    vectors = []
    weld_index = []
    point_index_stack = (len(points)//NPOINTS) * range(NPOINTS) # finish this
    if point_index_stack[0] == 0:
        point_index_stack.pop(0)
    subsample.append(points[0]) # adding first point to the subsample as the first three points are not eglible to be chosen
    for i in range(len(points)):
        if i + 3 > len(points):
            break

        current = points[i]
        next_mean = mean(points[i+1], points[i+2]) #sums the value of each coordinate before dividing it by 2, returns the mean of the input points
        current_vector = next_mean - current #could create a "current_mean" to make the results more accurate
        observed = points[i+3]
        observed_vector = observed - next_mean 
        current_vector_normalized = current_vector / np.abs(current_vector)
        observed_vector_normalized = observed_vector / np.abs(observed_vector)
        delta = np.abs(np.cross(current_vector_normalized, observed_vector_normalized)) #only interested in the value, could potentially look at the direction but that is TODO
        if i+3 >= point_index_stack[0]:
            subsample.append(points[i+3])
            vectors.append(observed_vector_normalized)
            point_index_stack.pop(0) #removes the first ndex
            
            continue
        if delta >= threshold_weld and delta < threshold_noise:
            subsample.append(observed)
            vectors.append(observed_vector_normalized)
            weld_index.append(i) # can potentially be used to turn on/off the welding torch, not fully implemented yet
            if np.shape(vectors)[0]< np.shape(subsample)[0]:
                vectors.append(observed_vector_normalized) # shoul only be used in the first iteration, but makes sure there are a corresponding vector to each point
                
    #aditionally this can be included to make shure the last point of the line is included:
    last = points[-1]
    if last not in subsample: #evt: if last != subsample[-1]:
        subsample.append(last)
        vectors.append(vectors[-1]) #adding the last vector once more to have a corresponding vector to the last point
    #
    assert np.shape(subsample)[0]==np.shape(vectors)[0], f"Error: the output subsample {np.shape(subsample)} and vectors {np.shape(vectors)} does not have the same amount of datapoints"
    return subsample, vectors, weld_index

#NOTE: to look "forward" in the path, one might want to remove the first vector and instead add a copy of the last vector, but this miught cause problems


