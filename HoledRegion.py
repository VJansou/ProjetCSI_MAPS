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

        if self.vertexToRemove == 222 or self.vertexToRemove == 129:
            self.mesh.plot('',zoomPoint=self.mesh.points[self.vertexToRemove])

        self.starEdges:List[int] = self.mesh.getExternalEdgesInCyclicOrder(vertexId=self.vertexToRemove)
        # print('star edges = ',self.starEdges)

        if vertexToRemove==222:
            print('starEdges = ',self.starEdges)

        # print('AFTER getExternalEdgesInCyclicOrder')

        # Si centralVertex est un sommet de la frontière
        if self.starEdges[0][0] != self.starEdges[-1][1] and len(self.starEdges) > 1:
            self.starEdges.append([self.starEdges[-1][1],self.starEdges[0][0]])
            self.isBoundary = True

        self.starVertices = [edge[0] for edge in self.starEdges]

        if vertexToRemove==222:
            print('star vertices : ',self.starVertices)
    
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
        return np.acos((p0p1**2 + p0p2**2 - p1p2**2)/(2*p0p1*p0p2))
    
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

        polarCoordonates = np.zeros((2,len(self.starEdges)))

        p_i = self.mesh.points[self.vertexToRemove]
        if self.vertexToRemove==222:print('ordre de parcours :')
        # Pour chaque sommet de N(i) en partant de j_k = j_1
        for k in range(0,len(self.starEdges)): #len(N_i)):
            if self.vertexToRemove==222:print("sommet ",k," ",self.starEdges[k][0])
            p_j_k = self.mesh.points[self.starEdges[k][0]]
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

    def getNewSimplices(self) -> Tuple[List[List[int]],List[List[int]]]:

        newEdges = []
        newFaces = []

        if len(self.starEdges) > 1:

            forbiddenEdges:List[List[int]] = self.getForbiddenEdges()

            if self.vertexToRemove == 222:
                print('forbidden edges : ',forbiddenEdges)

            points = self.getConformalMap()
            points = points.T
            tri = Delaunay(points=points)

            if self.vertexToRemove==222:
                plt.triplot(points[:,0], points[:,1], tri.simplices)

                # Tracer les points
                plt.plot(points[:,0], points[:,1], 'o')

                # Ajouter les numéros des sommets
                for i, (x, y) in enumerate(points):
                    if x != -np.inf:
                        plt.text(x, y, str(i), fontsize=12, color='red')  # Position et numéro des sommets # a_supprimer[i] -> i

                # Afficher la figure
                plt.show()

            for _face in tri.simplices:

                face = [self.starVertices[int(_face[0])],self.starVertices[int(_face[1])],self.starVertices[int(_face[2])]]

                if self.vertexToRemove == 222:
                    print("new faces : ", face)

                # On ajoute les nouvelles arrêtes
                newEdgesInFace = [[face[0],face[1]],[face[0],face[2]],[face[1],face[2]]]

                unforbiddenEdges = 0

                for edge in newEdgesInFace:
                    if edge not in forbiddenEdges:
                        edge.reverse()
                        if edge not in forbiddenEdges:
                            edge.reverse()
                            unforbiddenEdges = unforbiddenEdges + 1

                if unforbiddenEdges == 3:
                    newFaces.append(face)

                for edge in newEdgesInFace:
                    if edge not in forbiddenEdges and edge not in self.mesh.simplicies['edges']:
                        edge.reverse()
                        if edge not in forbiddenEdges and edge not in self.mesh.simplicies['edges']:
                            edge.reverse()
                            newEdges.append(edge)

        return newEdges,newFaces
    
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