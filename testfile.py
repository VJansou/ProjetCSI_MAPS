import MapsModel
from obja import *




model = MapsModel.MapsModel(L=4)
model.parse_file('./example/test1.obj')
print(model.vertices)
print('\n')
print(model.faces)
print('\n')
print(model.facesToList())
print('\n')
print(model.createEdgesList())
print('\n')
print(model.createNeighborsDict())
print('\n')
print(model.getEdgesWithVertex(3))
print('\n')
print(model.getFacesWithVertex(3))
print('\n')
print(model.get1RingExternalEdges(3))
print('\n')
print(model.getExternalVerticesInCyclicOrder(3)[0])


import numpy as np
import matplotlib.pyplot as plt

def is_point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using the ray-casting algorithm."""
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def calculate_internal_angles(polygon):
    angles = []
    num_vertices = len(polygon)

    for i in range(num_vertices):
        # Get the previous, current, and next vertex
        p1 = np.array(polygon[i - 1])  # Previous vertex
        p2 = np.array(polygon[i])      # Current vertex
        p3 = np.array(polygon[(i + 1) % num_vertices])  # Next vertex

        # Create vectors from the vertices
        vector_a = p1 - p2  # Vector from current to previous
        vector_b = p3 - p2  # Vector from current to next

        # Calculate the angle using the dot product
        dot_product = np.dot(vector_a, vector_b)
        magnitude_a = np.linalg.norm(vector_a)
        magnitude_b = np.linalg.norm(vector_b)

        # Calculate the angle in radians
        angle_radians = np.arccos(dot_product / (magnitude_a * magnitude_b))

        # Convert angle to degrees
        angle_degrees = np.degrees(angle_radians)

        # Calculate the barycenter of the triangle formed by p1, p2, p3
        barycenter = (p1 + p2 + p3) / 3

        # Check if the barycenter is inside the polygon
        if is_point_in_polygon(barycenter, polygon):
            angles.append(angle_degrees)  # Angle is internal
        else:
            angles.append(360 - angle_degrees)  # Angle is external

    return angles

def plot_polygon_with_angles(polygon, angles):
    # Convert the polygon to a numpy array for easy plotting
    polygon = np.array(polygon)
    
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the polygon
    ax.plot(*np.append(polygon, [polygon[0]], axis=0).T, 'b-', label='Polygon')  # Close the polygon by adding the first point again

    # Annotate angles at each vertex
    for i, angle in enumerate(angles):
        # Get the vertex position
        vertex = polygon[i]
        
        # Position for the angle text
        ax.text(vertex[0], vertex[1], f'{angle:.1f}°', fontsize=12, ha='center', va='bottom')

    # Set equal aspect ratio
    ax.set_aspect('equal')
    ax.set_title('Polygon with Internal Angles')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.grid()
    ax.legend()
    
    plt.show()

# Example usage:
polygon = [(2,2),(3,5),(6,6),(9,5),(12,4),(10,2),(8,4),(8,3),(7,3),(8,1),(5,0),(5,3)]  # Define your polygon vertices here
internal_angles = calculate_internal_angles(polygon)
print("Internal angles of the polygon:", internal_angles)

# Plot the polygon and its internal angles
plot_polygon_with_angles(polygon, internal_angles)

verticesInMesh = [1] * len([1,2,3,4,5])
print(verticesInMesh)

