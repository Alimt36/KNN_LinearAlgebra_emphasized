
from __future__ import annotations
import matplotlib.pyplot as plt 
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#---------------------------------------------------------------------------------------------------------------------------
points = [] # every element of it is a point with 4 atributes
#---------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------
class Linear_Algebra : 

#---------------------------------------------------------------------------------------------------------------------------
# the objects get made by 4 atributes : (x, y, z, classification) ---> basicaly a datastructure!
# and then the objects get added to a global list (points) for easier access
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
# a static function that gets two objects of type class (Linear_Algebra) and calculate the distance between them
#   ---> it has 7 diffrent modes that by the theory of linear algebra they have seperate meanings and usages
#       ---> 0 : Euclidean distance
#            1 : Manhattan Distance 
#            2 : Chebyshev Distance
#            3 : Minkowski Distance
#            4 : Cosine Distance
#            5 : Squared Euclidean Distance
#            6 : Canberra Distance
#---------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def calculate_distance(mode=0 , p0:Linear_Algebra=None , p1:Linear_Algebra=None ) -> float :
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
        
        # Cosine Distance
        elif mode == 4:
            inner_product = p0.x * p1.x + p0.y * p1.y + p0.z * p1.z

            norm_p0 = math.sqrt(p0.x**2 + p0.y**2 + p0.z**2)
            norm_p1 = math.sqrt(p1.x**2 + p1.y**2 + p1.z**2)
            
            if norm_p0 == 0 or norm_p1 == 0:
                return 1.0
            
            cosine_similarity = inner_product / (norm_p0 * norm_p1)
            
            return 1 - cosine_similarity
        
        # Squared Euclidean Distance
        elif mode == 5:
            diff_x = p0.x - p1.x
            diff_y = p0.y - p1.y
            diff_z = p0.z - p1.z
            return diff_x**2 + diff_y**2 + diff_z**2
        
        # Canberra Distance
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
# a function capable of printing the list (points_by_distance) in a neat way
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
# K-NN : 
#   ---> inputs : 
#       k : number of near neighbors to calculate the classification
#       v_to_classify :‌ is the obeject of the point that we want to clssify
#       points : is the list that holds the points of the dataset 
#       distance_mode : is the control varible of the function (calculate_distance()) 
#
#   ---> outputs : 
#       it returns two values : 
#           temp.index(max(temp)) : is the class number that the KNN has calculated.
#           points_by_distance : is a list that gets made by the function and is like : ( (a points list object) , (distance of the object form the new point) )
#
#   ---> how it works : 
#           calculates distance of all the points from the new point 
#           make the list (points_by_distance) 
#           sort the list (points_by_distance) by distance 
#           calculate all the distinct classes in the dataset
#           check the class of the k nearest neighbors and classify
#---------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def KNN( k=3 , v_to_classify:Linear_Algebra=None , points=None , distance_mode=0 ) :

        # global points_by_distance
        points_by_distance = []

        for i in range (0 , len(points)) :
            distance = Linear_Algebra.calculate_distance( distance_mode , points[i] , v_to_classify)
            if (distance != 0):
                points_by_distance.append((points[i] , distance))
            
        points_by_distance.sort(key=lambda x: x[1])

        Linear_Algebra.print_by_distance(points_by_distance)

        temp = []
        count_ = 0
        for i in range (0 , len(points)):
            if (points[i].classification not in temp and points[i].classification != None) :
                temp.append(points[i].classification)
                count_ += 1
        
        # print(count_)
        # print(temp)

        temp = [0] * count_
        for i in range (0 , k):
            p , d = points_by_distance[i]
            temp[p.classification] += 1

        return temp.index(max(temp)) , points_by_distance
#---------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------
def plot_points(points_by_distance, v_to_classify, k):#
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    classifications = {}
    for point, distance in points_by_distance:
        if point.classification not in classifications:
            classifications[point.classification] = {'x': [], 'y': [], 'z': []}
        classifications[point.classification]['x'].append(point.x)
        classifications[point.classification]['y'].append(point.y)
        classifications[point.classification]['z'].append(point.z)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (cls, coords) in enumerate(classifications.items()):
        ax.scatter(coords['x'], coords['y'], coords['z'], 
                  c=colors[i % len(colors)], marker='o', s=100, 
                  label=f'Class {cls}', alpha=0.6)
    
    k_nearest = points_by_distance[:k]
    knn_x = [p.x for p, d in k_nearest]
    knn_y = [p.y for p, d in k_nearest]
    knn_z = [p.z for p, d in k_nearest]
    ax.scatter(knn_x, knn_y, knn_z, c='black', marker='x', s=200, 
              linewidths=3, label=f'{k} Nearest Neighbors')
    
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
# loading the Iris dataset and tuning it so that it is compatible with this code 
#---------------------------------------------------------------------------------------------------------------------------
def load_iris_dataset():
    iris = load_iris()
    X = iris.data
    y = iris.target  # Classes: 0, 1, 2 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def create_vectors_from_dataset(X, y):

    global points
    points = []
    for i in range(len(X)):
        x, y_coord, z = X[i][0], X[i][1], X[i][2]
        classification = int(y[i])
        Linear_Algebra(x, y_coord, z, classification)
    return points

def test_knn_accuracy(X_test, y_test, k=3, distance_mode=0):

    correct = 0
    total = len(X_test)
    for i in range(total):
        test_point = Linear_Algebra.__new__(Linear_Algebra)
        test_point.x = X_test[i][0]
        test_point.y = X_test[i][1]
        test_point.z = X_test[i][2]
        test_point.classification = None
        
        predicted_class, _ = Linear_Algebra.KNN(k=k, v_to_classify=test_point, points=points, distance_mode=distance_mode)
        
        if predicted_class == int(y_test[i]):
            correct += 1
    
    accuracy = (correct / total) * 100
    return accuracy
#---------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------
def __main__() -> None :
    # print("!!!!")

    # v1 = Linear_Algebra(1,1,1 , 0)
    # v2 = Linear_Algebra(2,2,2 , 0)
    # v3 = Linear_Algebra(3,3,3 , 0)

    # v4 = Linear_Algebra(10,10,10 , 1)
    # v5 = Linear_Algebra(11,11,11 , 1)
    # v6 = Linear_Algebra(13,13,13 , 1)

    # v7 = Linear_Algebra(1,0,0 , 0)
    # v8 = Linear_Algebra(0,0,0 , 0)
    # v9 = Linear_Algebra(0,1,1 , 0)

    # v10 = Linear_Algebra(6,6,6 , 2)
    # v11 = Linear_Algebra(7,6,6 , 2)
    # v12 = Linear_Algebra(8,7,6 , 2)

    # # v_in = Linear_Algebra(0.1,0.1,0.1)
    # v_in = Linear_Algebra(9,9,9)
    # # v_in = Linear_Algebra(6,7,7)

    # k_input = 3

    # v_in.classification , points_by_distance = Linear_Algebra.KNN(k=k_input , v_to_classify=v_in , points=points)
    
    # print(v_in.classification)

    # plot_points(points_by_distance, v_in , k_input)
    # Load real dataset

    k_input = 4

    distance_mode = 0

    X_train, X_test, y_train, y_test = load_iris_dataset()
    create_vectors_from_dataset(X_train, y_train)
    
    test_point = Linear_Algebra(X_test[0][0], X_test[0][1], X_test[0][2])
    predicted, points_by_distance = Linear_Algebra.KNN(k=k_input, v_to_classify=test_point, points=points , distance_mode=distance_mode)
    
    print(f"\nPredicted class: {predicted}")
    print(f"Actual class: {int(y_test[0])}")
    
    plot_points(points_by_distance, test_point, k=k_input)

__main__()
#---------------------------------------------------------------------------------------------------------------------------
