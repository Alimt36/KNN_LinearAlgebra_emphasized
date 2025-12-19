
from __future__ import annotations
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import math

#---------------------------------------------------------------------------------------------------------------------------
points = []   
# points_by_distance = []
#---------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------
class vector_ : 

#---------------------------------------------------------------------------------------------------------------------------
    def __init__(self , x  , y , z=0 , classification=None ):
        self.x = x 
        self.y = y
        self.z = z
        self.classification = classification

        global points 
        points.append(self)
#---------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------
    def __str__(self) -> str : 
        return f"({self.x} , {self.y} , {self.z})"
#---------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def calculate_distance(mode=0 , p0:vector_=None , p1:vector_=None ) -> float :
        # Euclidean distance 
        if (mode == 0): 
            return math.sqrt( (abs(p0.x - p1.x)**2) + (abs(p0.y - p1.y)**2) + (abs(p0.z - p1.z)**2) )
        # Manhattan Distance (L1 norm)
        elif mode == 1:
            return abs(p0.x - p1.x) + abs(p0.y - p1.y) + abs(p0.z - p1.z)
        
        # Chebyshev Distance (L∞ norm)
        elif mode == 2:
            return max(abs(p0.x - p1.x), abs(p0.y - p1.y), abs(p0.z - p1.z))
        
        # Minkowski Distance (generalized)
        elif mode == 3:
            p = 3
            return (abs(p0.x - p1.x)**p + abs(p0.y - p1.y)**p + abs(p0.z - p1.z)**p)**(1/p)
        
        # Mode 4: Cosine Distance
        elif mode == 4:
            # Inner product <p0, p1>
            inner_product = p0.x * p1.x + p0.y * p1.y + p0.z * p1.z

            norm_p0 = math.sqrt(p0.x**2 + p0.y**2 + p0.z**2)
            norm_p1 = math.sqrt(p1.x**2 + p1.y**2 + p1.z**2)
            
            if norm_p0 == 0 or norm_p1 == 0:
                return 1.0
            
            cosine_similarity = inner_product / (norm_p0 * norm_p1)
            
            return 1 - cosine_similarity
        
        # Mode 5: Squared Euclidean Distance
        elif mode == 5:
            diff_x = p0.x - p1.x
            diff_y = p0.y - p1.y
            diff_z = p0.z - p1.z
            return diff_x**2 + diff_y**2 + diff_z**2
        
        # Mode 6: Canberra Distance
        elif mode == 6:
            sum_dist = 0
            coords = [(p0.x, p1.x), (p0.y, p1.y), (p0.z, p1.z)]
            for c0, c1 in coords:
                if abs(c0) + abs(c1) != 0:
                    sum_dist += abs(c0 - c1) / (abs(c0) + abs(c1))
            return sum_dist
        
        else:
            return None
#---------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def print_by_distance(points_by_distance) -> None : 
        # print(points_by_distance)
        for point, distance in points_by_distance:
            print("--------------------------------------")
            print(f"Point: {point},\nDistance: {distance:.2f}")
            print("--------------------------------------\n")
#---------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def KNN( k=3 , v_to_classify:vector_=None , points=None ) :

        # global points_by_distance
        points_by_distance = []

        for i in range (0 , len(points)) :
            distance = vector_.calculate_distance( 3 , points[i] , v_to_classify)
            if (distance != 0):
                points_by_distance.append((points[i] , distance))
            
        points_by_distance.sort(key=lambda x: x[1])

        vector_.print_by_distance(points_by_distance)

        temp = []
        count_ = 0
        for i in range (0 , len(points)):
            if (points[i].classification not in temp and points[i].classification != None) :
                temp.append(points[i].classification)
                count_ += 1
        
        # print(count_)
        print(temp)

        temp = [0] * count_
        for i in range (0 , k):
            p , d = points_by_distance[i]
            temp[p.classification] += 1

        return temp.index(max(temp)) , points_by_distance
#---------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------
def plot_points(points_by_distance, v_to_classify, k):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Separate points by classification
    classifications = {}
    for point, distance in points_by_distance:
        if point.classification not in classifications:
            classifications[point.classification] = {'x': [], 'y': [], 'z': []}
        classifications[point.classification]['x'].append(point.x)
        classifications[point.classification]['y'].append(point.y)
        classifications[point.classification]['z'].append(point.z)
    
    # Plot each classification with different color
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (cls, coords) in enumerate(classifications.items()):
        ax.scatter(coords['x'], coords['y'], coords['z'], 
                  c=colors[i % len(colors)], marker='o', s=100, 
                  label=f'Class {cls}', alpha=0.6)
    
    # Highlight k-nearest neighbors
    k_nearest = points_by_distance[:k]
    knn_x = [p.x for p, d in k_nearest]
    knn_y = [p.y for p, d in k_nearest]
    knn_z = [p.z for p, d in k_nearest]
    ax.scatter(knn_x, knn_y, knn_z, c='black', marker='x', s=200, 
              linewidths=3, label=f'{k} Nearest Neighbors')
    
    # Plot the point to classify
    ax.scatter(v_to_classify.x, v_to_classify.y, v_to_classify.z, 
              c='yellow', marker='*', s=500, edgecolors='black', 
              linewidths=2, label='Point to Classify')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('K-NN Classification Visualization')
    plt.show()
#---------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------
def __main__() -> None :
    # print("!!!!")

    v1 = vector_(1,1,1 , 0)
    v2 = vector_(2,2,2 , 0)
    v3 = vector_(3,3,3 , 0)

    v4 = vector_(10,10,10 , 1)
    v5 = vector_(11,11,11 , 1)
    v6 = vector_(13,13,13 , 1)

    v7 = vector_(1,0,0 , 0)
    v8 = vector_(0,0,0 , 0)
    v9 = vector_(0,1,1 , 0)

    v10 = vector_(6,6,6 , 2)
    v11 = vector_(7,6,6 , 2)
    v12 = vector_(8,7,6 , 2)

    # v_in = vector_(0.1,0.1,0.1)
    v_in = vector_(9,9,9)
    # v_in = vector_(6,7,7)

    k_input = 3

    v_in.classification , points_by_distance = vector_.KNN(k=k_input , v_to_classify=v_in , points=points)
    
    print(v_in.classification)

    plot_points(points_by_distance, v_in , k_input)


__main__()
#---------------------------------------------------------------------------------------------------------------------------
