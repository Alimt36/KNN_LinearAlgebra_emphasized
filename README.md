# K-Nearest Neighbors (KNN) Classifier with emphasis on Linear Algebra

A Python implementation of K-NN classification emphasizing linear algebra concepts: inner products, vector representations, and distance metrics.

![KNN Visualization](images/img.png)

---

## Class: `Linear_Algebra`

### `__init__(x, y, z=0, classification=None)`
Creates a point object with 4 attributes: (x, y, z, classification)
- Automatically adds the object to global `points` list

### `calculate_distance(mode=0, p0, p1) -> float`
**Static method** that calculates distance between two points using different metrics.

**Modes:**
- `0`: Euclidean distance
- `1`: Manhattan Distance
- `2`: Chebyshev Distance
- `3`: Minkowski Distance
- `4`: Cosine Distance
- `5`: Squared Euclidean Distance
- `6`: Canberra Distance

### `KNN(k=3, v_to_classify, points, distance_mode=0)`
**K-Nearest Neighbors classifier**

**Inputs:**
- `k`: Number of nearest neighbors
- `v_to_classify`: Point object to classify
- `points`: List of dataset points
- `distance_mode`: Controls which distance metric to use (0-6)

**Outputs:**
- `classification`: Predicted class number
- `points_by_distance`: List of (point, distance) tuples sorted by distance

**How it works:**
1. Calculates distance from new point to all dataset points
2. Sorts points by distance
3. Finds k nearest neighbors
4. Returns most common classification among k neighbors

---

## Helper Functions

### `load_iris_dataset()`
Loads Iris dataset and splits into train/test sets (80/20)

### `create_vectors_from_dataset(X, y)`
Converts dataset arrays into `Linear_Algebra` objects

### `test_knn_accuracy(X_test, y_test, k=3, distance_mode=0)`
Tests KNN accuracy on test dataset

### `plot_points(points_by_distance, v_to_classify, k)`
Creates 3D visualization of classification results

---

## Linear Algebra Connection

**Vector Representation:** Each data point is a vector in ℝ³ space

### Mode 0: Euclidean Distance (L2 norm)
```
d(u,v) = √(<u-v, u-v>) = √(||u-v||²)
```
Uses inner product of difference vector with itself

### Mode 1: Manhattan Distance (L1 norm)
```
d(u,v) = |u₁-v₁| + |u₂-v₂| + |u₃-v₃|
```
Sum of absolute coordinate differences

### Mode 2: Chebyshev Distance (L∞ norm)
```
d(u,v) = max(|u₁-v₁|, |u₂-v₂|, |u₃-v₃|)
```
Maximum absolute difference across all dimensions

### Mode 3: Minkowski Distance
```
d(u,v) = (|u₁-v₁|ᵖ + |u₂-v₂|ᵖ + |u₃-v₃|ᵖ)^(1/p)
```
Generalized distance metric (p=3 in implementation)

### Mode 4: Cosine Distance
```
d(u,v) = 1 - (<u,v> / (||u|| × ||v||))
where ||u|| = √(<u,u>)
```
Uses inner product directly to measure angle between vectors

### Mode 5: Squared Euclidean Distance
```
d(u,v) = <u-v, u-v> = ||u||² - 2<u,v> + ||v||²
```
Pure inner product without square root (faster computation)

### Mode 6: Canberra Distance
```
d(u,v) = Σ |uᵢ-vᵢ| / (|uᵢ| + |vᵢ|)
```
Weighted Manhattan distance, sensitive to small changes near zero

---

by [@Alimt36](https://github.com/Alimt36)