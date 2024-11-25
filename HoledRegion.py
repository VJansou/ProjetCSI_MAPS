from Mesh import Mesh
from typing import List,Tuple
from scipy.spatial import Delaunay,ConvexHull

import matplotlib.pyplot as plt

import numpy as np

class HoledRegion:
    def __init__(self,vertexToRemove:int,mesh:Mesh):
        
        self.vertexToRemove = vertexToRemove
        self.mesh = mesh

        self.isBoundary = False

        #if self.vertexToRemove == 222 or self.vertexToRemove == 129:
        #    self.mesh.plot('',zoomPoint=self.mesh.points[self.vertexToRemove])

        self.starEdges = self.mesh.get1RingExternalEdges(self.vertexToRemove).copy()

        res = self.mesh.getExternalVerticesInCyclicOrder(vertexId=self.vertexToRemove)
        self.starVertices = res[0].copy()
        self.isBoundary = res[1]

        # if vertexToRemove == 132:
            # print("star of 132 : ",self.starVertices)

            # print(vertexToRemove," neighbors : ",mesh.neighbors[vertexToRemove])
            # for neighbor in mesh.neighbors[vertexToRemove]:
            #     print("           ",neighbor," neighbors : ",mesh.neighbors[neighbor])

        # print('star edges = ',self.starEdges)

        #if vertexToRemove==222:
        #    print('starEdges = ',self.starEdges)

        # print('AFTER getExternalEdgesInCyclicOrder')


        #if vertexToRemove==222:
        #    print('star vertices : ',self.starVertices)
    
    """
        Étant donné trois points 3D p0, p1 et p2, retourne la valeur de l'angle (p0p1,p0p2)

        Args: p0:np.ndarray, les coordonnées du premier point, c'est en celui-ci qu'est calculé l'angle
              p1:np.ndarray, les coordonnées du second point
              p2:np.ndarray, les coordonnées du second point
              Chacun de ces array a pour dimension (3,)

        Return: float, la valeur de l'angle exprimé en radian, cette valeur est comprise entre 0 et pi
    """
    def getFaceCentralAngle(self,p0:np.ndarray,p1:np.ndarray,p2:np.ndarray) -> float:
        # Pour le calcul d'un angle d'une face, on utilise la loi d'Al-Kashi
        p0p1 = np.sqrt(np.sum((p0-p1)**2))
        p1p2 = np.sqrt(np.sum((p1-p2)**2))
        p0p2 = np.sqrt(np.sum((p0-p2)**2))
        return np.arccos((p0p1**2 + p0p2**2 - p1p2**2)/(2*p0p1*p0p2))
    
    def getFacesCentralAngles(self) -> np.ndarray:
        # Calculer les angles centraux de chaque face
        angles = np.zeros((len(self.starEdges),1))
        for k in range(len(self.starEdges)):
            p_j_lm1 = self.mesh.points[self.starEdges[k][0]]
            p_j_l = self.mesh.points[self.starEdges[k][1]]
            angles[k,0] = self.getFaceCentralAngle(p0=self.mesh.points[self.vertexToRemove],p1=p_j_l,p2=p_j_lm1)
        return angles
    
    """
        Retourne, depuis un ensemble de coordonnées polaires 2D, ces mêmes coordonnées dans un repère cartésien 2D.

        Args: polarCoordonates:np.ndarray, la matrice coordonnées polaires en question, chaque colonne correspond aux coordonnées d'un point.
                La première ligne correspond aux modules et la deuxième aux arguments.

        Return: np.ndarray, la matrice des coordonnées cartésiennes, chaque colonne correspond aux coordonnées d'un point.
                La première ligne correspond aux abscisses et la deuxième aux ordonnées.
    """
    def getConformalMap(self) -> np.ndarray:

        angles:np.ndarray = self.getFacesCentralAngles()

        angles = angles[::-1,:]

        polarCoordonates = np.zeros((2,len(self.starVertices)))

        p_i = self.mesh.points[self.vertexToRemove]
        
        # Pour chaque sommet de N(i) en partant de j_k = j_1
        for k in range(0,len(self.starVertices)): #len(N_i)):
            
            p_j_k = self.mesh.points[self.starVertices[k]]
            # Calculer et stocker r_k
            polarCoordonates[0,k] = np.sqrt(np.sum((p_i-p_j_k)**2))
            # Calculer et stocker theta_k
            polarCoordonates[1,k] = np.sum(angles[:k])

        cartesianCoordonates = np.zeros(polarCoordonates.shape)

        cartesianCoordonates[0,:] = polarCoordonates[0,:] * np.cos(polarCoordonates[1,:])
        cartesianCoordonates[1,:] = polarCoordonates[0,:] * np.sin(polarCoordonates[1,:])

        return cartesianCoordonates
    
    def getForbiddenEdges(self) -> List[List[int]]:

        forbiddenEdges:List[List[int]] = []

        if not self.isBoundary:
            points = self.getConformalMap()
            hull = ConvexHull(points=points.T)

            for edge in hull.simplices:
                edge = [self.starVertices[edge[0]],self.starVertices[edge[1]]]
                if edge not in self.mesh.simplicies['edges']:
                    edge.reverse()
                    if edge not in self.mesh.simplicies['edges']:
                        edge.reverse()
                        forbiddenEdges.append(edge)

        return forbiddenEdges
    
    def isPointInPolygon(self, point, polygon):
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
    
    def calculateInternalAngle(self, i, polygon):

        num_vertices = len(polygon)
            
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
        # print('valeure bizarre ', dot_product / (magnitude_a * magnitude_b))
        # > 1 
        angle_radians = np.arccos(np.min([dot_product / (magnitude_a * magnitude_b),1]))

        # Convert angle to degrees
        angle_degrees = np.degrees(angle_radians)

        # Calculate the barycenter of the triangle formed by p1, p2, p3
        barycenter = (p1 + p2 + p3) / 3

        # Check if the barycenter is inside the polygon
        if self.isPointInPolygon(barycenter, polygon):
            return angle_degrees  # Angle is internal
        else:
            return 360 - angle_degrees  # Angle is external

    def calculateInternalAngles(self, polygon):
        angles = []
        num_vertices = len(polygon)

        for i in range(num_vertices):
            angles.append(self.calculateInternalAngle(i, polygon))

        return angles
    
    def plotPolygonWithAngles(self, polygon, angles):
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
    
    def getNewSimplices(self) -> Tuple[List[List[int]],List[List[int]]]:

        newEdges = []
        newFaces = []

        if len(self.starEdges) > 1:
            if self.isBoundary:
                boundaryEdge = tuple(sorted([self.starVertices[0], self.starVertices[-1]]))
                if boundaryEdge not in self.starEdges: # a priori inutile
                    self.starEdges.append(boundaryEdge)
                    newEdges.append(boundaryEdge)
            # We now have a cyclic star whether the vertex to remove is on the border or not
            points = self.getConformalMap().T.tolist()
            internalAngles = self.calculateInternalAngles(points)
            #self.plotPolygonWithAngles(points, internalAngles)
            
            while(len(points) > 3):
                angles = np.array(internalAngles)

                sortedAnglesIdx = np.argsort(angles)
                
                indexToModify = sortedAnglesIdx[0]

                nbVertices = len(self.starVertices)
                newEdges.append(tuple(sorted([self.starVertices[indexToModify-1], self.starVertices[(indexToModify+1)%nbVertices]])))
                newFaces.append(tuple(sorted([self.starVertices[indexToModify-1], self.starVertices[(indexToModify%nbVertices)], self.starVertices[(indexToModify+1)%nbVertices]])))

                if self.vertexToRemove == 132:
                    print("new face add to newFaces :",newFaces[-1])

                del points[indexToModify]
                del internalAngles[indexToModify]
                del self.starVertices[indexToModify]

                nbPointsRestant = len(points)
                internalAngles[indexToModify%nbPointsRestant] = self.calculateInternalAngle(indexToModify%nbPointsRestant, points)
                internalAngles[indexToModify-1] = self.calculateInternalAngle(indexToModify-1, points)

                # if self.vertexToRemove==74:
                #     self.plotPolygonWithAngles(points, internalAngles)
            # print(self.starVertices)
            newFaces.append(tuple(sorted([self.starVertices[0], self.starVertices[1], self.starVertices[2]])))

        return newEdges, newFaces
                


            

        
#   def getNewSimplices(self) -> Tuple[List[List[int]],List[List[int]]]:
#
#       newEdges = []
#       newFaces = []
#
#       if len(self.starEdges) > 1:
#
#           forbiddenEdges:List[List[int]] = self.getForbiddenEdges()
#
#           if self.vertexToRemove == 222:
#               print('forbidden edges : ',forbiddenEdges)
#
#           points = self.getConformalMap()
#           points = points.T
#           tri = Delaunay(points=points)
#
#           if self.vertexToRemove==222:
#               plt.triplot(points[:,0], points[:,1], tri.simplices)
#
#               # Tracer les points
#               plt.plot(points[:,0], points[:,1], 'o')
#
#               # Ajouter les numéros des sommets
#               for i, (x, y) in enumerate(points):
#                   if x != -np.inf:
#                       plt.text(x, y, str(i), fontsize=12, color='red')  # Position et numéro des sommets # a_supprimer[i] -> i
#
#               # Afficher la figure
#               plt.show()
#
#           for _face in tri.simplices:
#
#               face = [self.starVertices[int(_face[0])],self.starVertices[int(_face[1])],self.starVertices[int(_face[2])]]
#
#               if self.vertexToRemove == 222:
#                   print("new faces : ", face)
#
#               # On ajoute les nouvelles arrêtes
#               newEdgesInFace = [[face[0],face[1]],[face[0],face[2]],[face[1],face[2]]]
#
#               unforbiddenEdges = 0
#
#               for edge in newEdgesInFace:
#                   if edge not in forbiddenEdges:
#                       edge.reverse()
#                       if edge not in forbiddenEdges:
#                           edge.reverse()
#                           unforbiddenEdges = unforbiddenEdges + 1
#
#               if unforbiddenEdges == 3:
#                   newFaces.append(face)
#
#               for edge in newEdgesInFace:
#                   if edge not in forbiddenEdges and edge not in self.mesh.simplicies['edges']:
#                       edge.reverse()
#                       if edge not in forbiddenEdges and edge not in self.mesh.simplicies['edges']:
#                           edge.reverse()
#                           newEdges.append(edge)
#
#       return newEdges,newFaces
    
# hr = HoledRegion(vertexToRemove=129,mesh=self.mesh.copy())
#                 pointsbis = hr.getConformalMap().T
#                 tribis = Delaunay(pointsbis)

#                 plt.triplot(pointsbis[:,0], pointsbis[:,1], tribis.simplices)

#                 # Tracer les points
#                 plt.plot(pointsbis[:,0], pointsbis[:,1], 'o')

#                 hr_l = hr.starVertices
                
#                 # Ajouter les numéros des sommets
#                 for i, (x, y) in enumerate(pointsbis):
#                     if x != -np.inf:
#                         plt.text(x, y, str(hr_l[i]), fontsize=12, color='red')  # Position et numéro des sommets

#                 plt.title("129")

#                 # Afficher la figure
#                 plt.show()

#                 plt.triplot(points[:,0], points[:,1], tri.simplices)

#                 # Tracer les points
#                 plt.plot(points[:,0], points[:,1], 'o')

#                 # Ajouter les numéros des sommets
#                 for i, (x, y) in enumerate(points):
#                     if x != -np.inf:
#                         plt.text(x, y, str(''), fontsize=12, color='red')  # Position et numéro des sommets

#                 plt.title("test")

#                 # Afficher la figure
#                 plt.show()

#                 pp = np.array([np.array([0,0]),points[5,:],points[1,:],points[2,:],points[3,:],points[4,:],points[0,:]])
               
#                 plt.triplot(pp[:,0], pp[:,1], np.array([[0,1,6],[0,1,2],[0,2,3],[0,3,4],[0,4,5]]))

#                 # Tracer les points
#                 plt.plot(pp[:,0], pp[:,1], 'o')

#                 l = [222,129,225,220,133,131,71]

#                 # Ajouter les numéros des sommets
#                 for i, (x, y) in enumerate(pp):
#                     if x != -np.inf:
#                         plt.text(x, y, str(l[i]), fontsize=12, color='red')  # Position et numéro des sommets

#                 plt.title("test")

#                 # Afficher la figure
#                 plt.show()