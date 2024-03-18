import numpy as np
import skimage as ski

# Load the skeletonized PNG image
def traverse(coords, image_path:str = 'skeleton.png'):
    image = ski.io.imread(image_path, "gray")
    coords = [[1385, 402], [417, 388], [266, 376], [272, 291], [861, 332]]
    # Define directions for traversing (8-connectivity)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1), (1, 0), (1, 1)]
    # Initialize a visited matrix
    visited = np.zeros_like(image, dtype=np.uint8)
    # Define a function to traverse along white pixels

    def traverse(start:list[int], end:list[int], visited):
        # start = (start[0], start[1])
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
    ski.io.imsave("traversed.png", result_image)
    return None