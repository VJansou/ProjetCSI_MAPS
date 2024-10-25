from typing import List,Dict
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pandas as pd

class Mesh:
    def __init__(self,stepNum:int):
        self.stepNum = stepNum
        self.points:List[np.ndarray] = []
        self.simplicies:Dict[List[int]] = {'vertices':[],'edges':[],'faces':[]}
    
    def copy(self):
        # Créer une copie de l'objet Mesh
        return Mesh(self.stepNum)  # Remplace par une copie réelle des attributs de Mesh

    def removeDoubleEdges(self) -> None:
        edges:List[List[int]] = self.simplicies['edges'].copy()

        for edge in edges:
            try:
                reversedEdge = edge.copy()
                reversedEdge.reverse()
                edges.remove(reversedEdge)
            except ValueError:
                pass

        self.simplicies['edges'] = edges.copy()

    """
        Retourne le nombre de voisin d'un sommet du maillage ( = card(1-Ring) ).

        Args : vertexId:int, l'indice associé au sommet en question.
        
        Returns : int, le nombre de voisins.
    """
    def getNumberOfNeighbors(self,vertexId:int) -> int:
        return np.ravel(np.array(self.simplicies['edges'])).tolist().count(vertexId)
    
    """
        Retourne la liste des arrêtes du maillage contenant un certain sommet.

        Args : vertexId:int, l'indice associé au sommet en question.
        
        Returns : List[List[int]], la liste des arrêtes, une arrête étant une liste de deux entiers : 
                    les indices associés aux deux sommets de l'arrête.
    """
    def getEdgesWithVertex(self,vertexId:int) -> List[List[int]]:
        
        flattenEdges = np.ravel(np.array(self.simplicies['edges']))
        edgesList = flattenEdges.tolist()

        indices = []
        start = 0
        allIndicesFound = False

        while not allIndicesFound:
            try:
                index = edgesList.index(vertexId,start)
                indices.append(index)
                start = index+1
            except ValueError:
                allIndicesFound = True

        indices = np.ravel(np.array(indices))
        indices = (indices//2).tolist()

        edges = [self.simplicies['edges'][i] for i in indices] # np.array(self.simplicies['edges'])[indicesInNumpy.tolist()]
        return edges
    
    """
        Retourne la liste des sommets voisins d'un certain sommet ( = sommet du 1-ring d'un sommet).

        Args : vertexId:int, l'indice associé au sommet en question.
        
        Returns : List[int], la liste des indices associés sommets.
    """
    def getNeighbors(self,vertexId:int) -> List[int]:
        verticesInEdgesWithVertex = np.ravel(np.array(self.getEdgesWithVertex(vertexId=vertexId))).tolist()
        selectedVertexInList = True

        while selectedVertexInList:
            try:
                verticesInEdgesWithVertex.remove(vertexId)
            except ValueError:
                selectedVertexInList = False

        neighbors = verticesInEdgesWithVertex
        return neighbors

    """
        Retourne la liste des faces du maillage contenant un certain sommet.

        Args : vertexId:int, l'indice associé au sommet en question.
        
        Returns : List[List[int]], la liste des faces, une face étant une liste de trois entiers : 
                    les indices associés trois aux sommets de la face.
    """
    def getFacesWithVertex(self,vertexId:int) -> List[List[int]]:

        # print('BEGIN getFacesWithVertex')
        
        # for face in self.simplicies['faces']:
        #     print(str(face[0])+' '+str(face[1])+' '+str(face[2])+'\n')

        flattenFaces = np.ravel(np.array(self.simplicies['faces']))

        # print(flattenFaces)

        facesList:List[int] = flattenFaces.tolist()

        indices = []
        start = 0
        allIndicesFound = False

        while not allIndicesFound:
            try:
                index = facesList.index(vertexId,start)
                indices.append(index)
                start = index+1
            except ValueError:
                allIndicesFound = True

        indicesInNumpy = np.ravel(np.array(indices))
        indicesInNumpy = indicesInNumpy//3

        faces = np.array(self.simplicies['faces'])[indicesInNumpy.tolist()]

        # print('faces = ',faces)

        # print('END getFacesWithVertex')

        if vertexId==129:
            print(faces.tolist())
        
        return faces.tolist()
    
    """
        Retourne la liste des arrêtes extérieures du 1-ring d'un sommet.

        Args : centralVertex:int, l'indice associé au sommet en question.
        
        Returns : List[List[int]], la liste des arrêtes, une arrête étant une liste de deux entiers : 
                    les indices associés aux deux sommets de l'arrête.
    """
    def get1RingExternalEdges(self,centralVertex:int) -> List[List[int]]:

        facesWithCentralVertex = self.getFacesWithVertex(vertexId=centralVertex)
        externalEdges = []

        for face in facesWithCentralVertex:
            externalEdge = []
            
            if centralVertex == face[0]:
                externalEdge.append(face[1])
                externalEdge.append(face[2])
            elif centralVertex == face[1]:
                externalEdge.append(face[0])
                externalEdge.append(face[2])
            elif centralVertex == face[2]:
                externalEdge.append(face[0])
                externalEdge.append(face[1])

            externalEdges.append(externalEdge)

        # Suppression des potentiels arrêtes présentes en double
        for edge in externalEdges:
            try:
                reversedEdge = edge.copy()
                reversedEdge.reverse()
                externalEdges.remove(reversedEdge)
            except ValueError:
                pass

        return externalEdges

    """
        Retourne la liste des voisins d'un sommet dans.

        Args : centralVertex:int, l'indice associé au sommet en question.
        
        Returns : List[List[int]], la liste des arrêtes, une arrête étant une liste de deux entiers : 
                    les indices associés aux deux sommets de l'arrête.
    """
    # ATTENTION : Cette fonction ne fonctionne pas pour les sommets centraux sur la frontiere du maillage
    def getExternalEdgesInCyclicOrder(self,vertexId:int) -> List[int]:

        # print('BEGIN getExternalEdgesInCyclicOrder')
        
        externalEdges = self.get1RingExternalEdges(centralVertex=vertexId)

        for edge in externalEdges:
            try:
                reversedEdge = edge.copy()
                reversedEdge.reverse()
                externalEdges.remove(reversedEdge)
            except ValueError:
                pass

        # print('externalEdges = ',externalEdges)

        rearangedExternalEdges:List[List[int]] = [externalEdges[0]]

        i = 0

        # print("BEFORE FIRST WHILE LOOP")

        while i+1 < len(externalEdges) :

            # print('i = ',i,' len(rearangedExternalEdges) = ',len(rearangedExternalEdges))

            complementaryEdgeFound = False
            j = 0
            
            while j < len(externalEdges) and not complementaryEdgeFound:
                
                edge = externalEdges[j]

                if edge not in rearangedExternalEdges:
                    
                    if edge[0] == rearangedExternalEdges[i][1]:
                        
                        rearangedExternalEdges.append(edge)
                        complementaryEdgeFound = True
                    
                    elif edge[1] == rearangedExternalEdges[i][1]:
                        
                        edge.reverse()
                        rearangedExternalEdges.append(edge)
                        complementaryEdgeFound = True
                
                j = j + 1

            if complementaryEdgeFound:
                i = i+1
                # print("i = i + 1")
            else:
                rearangedExternalEdges.reverse() # rearangedExternalEdges[i-1].reverse()
                for edge in rearangedExternalEdges: edge.reverse()

            # print("rearangedExternalEdges = ",rearangedExternalEdges)

        return rearangedExternalEdges
    
    """
        Affiche dans une fenêtre de matplotlib le maillage. La figure affiché est une figure 2D

        Args : title:str, le titre de la figure.
        
        Returns : None
    """
#     def plot(self,title:str,zoomPoint:np.ndarray=None) -> None:

#         points = self.points

#         bool = True

#         while bool:
#             try:
#                 points.remove(np.array([-np.inf,-np.inf,-np.inf]))
#             except ValueError:
#                 bool = False

#         points = np.array(points)

#         plt.triplot(points[:,0], points[:,1], self.simplicies['faces'])

#         # Tracer les points
#         # plt.plot(points[:,0], points[:,1], 'o')
#         fig = go.Figure(go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='markers', marker=dict(size=5)))
#         fig.show()

#         # if points.shape[0]==8:
#         #     a_supprimer = [222,225,227,220,133,131,129,71]
#         # elif points.shape[0]==7:
#         #     a_supprimer = [225,227,220,133,131,129,71]

#         # # Ajouter les numéros des sommets
#         # for i, (x, y, z) in enumerate(self.points):
#         #     if x != -np.inf:
#         #         plt.text(x, y, str(i), fontsize=12, color='red')  # Position et numéro des sommets # a_supprimer[i] -> i

#         # plt.title(title)

#         # if zoomPoint is not None:
#         #         plt.axis([zoomPoint[0]-0.15, zoomPoint[0]+0.15, zoomPoint[1]-0.15, zoomPoint[1]+0.15])

#         # # Afficher la figure
#         # plt.savefig(title+'.png')
#         # plt.show()

#     def copy(self):
#         meshCopy:Mesh = Mesh(stepNum=self.stepNum)
#         meshCopy.points = self.points.copy()
#         meshCopy.simplicies['vertices'] = self.simplicies['vertices'].copy()
#         meshCopy.simplicies['edges'] = self.simplicies['edges'].copy()
#         meshCopy.simplicies['faces'] = self.simplicies['faces'].copy()
#         return meshCopy

# # def getNeighborsInCyclicOrder(self,externalEdges:List[List[int]]) -> List[int]:
        
# #         externalEdges = 

# #         rearangedExternalEdges:List[List[int]] = [externalEdges[0]]

# #         i = 1

# #         while i < len(externalEdges):
# #             # print('ITERATOIN ',i)
# #             # print(rearangedExternalEdges)
# #             complementaryEdgeFound = False
# #             j = 0
# #             # print('len(externalEdges) = ',len(externalEdges))
# #             while j < len(externalEdges) and not complementaryEdgeFound:
# #                 # print('j = ',j)
# #                 edge = externalEdges[j]
# #                 if edge not in rearangedExternalEdges:
# #                     # print('a')
# #                     # print('edge = ',edge)
# #                     if edge[0] == rearangedExternalEdges[i-1][1]:
# #                         # print('b')
# #                         rearangedExternalEdges.append(edge)
# #                         complementaryEdgeFound = True
# #                     elif edge[1] == rearangedExternalEdges[i-1][1]:
# #                         # print('c')
# #                         edge.reverse()
# #                         rearangedExternalEdges.append(edge)
# #                         complementaryEdgeFound = True
# #                 j = j + 1

# #             if complementaryEdgeFound:
# #                 i = i+1
# #             else:
# #                 rearangedExternalEdges[i-1].reverse()

# #         vertices = [edge[0] for edge in rearangedExternalEdges]

# #         return vertices

def plot(self, title: str, zoomPoint: np.ndarray = None) -> None:
    points = self.points

    # Nettoyer les points infiniment grands
    while True:
        try:
            points.remove(np.array([-np.inf, -np.inf, -np.inf]))
        except ValueError:
            break

    points = np.array(points)

    # Si 'simplicies' contient des faces, on les utilise pour tracer les arêtes
    faces = self.simplicies['faces']

    # Tracer les arêtes (lignes entre les sommets)
    edges_x = []
    edges_y = []
    edges_z = []
    for face in faces:
        for i in range(3):
            for j in range(i + 1, 3):  # Assurez-vous de relier chaque paire de sommets
                edges_x += [points[face[i], 0], points[face[j], 0], None]
                edges_y += [points[face[i], 1], points[face[j], 1], None]
                edges_z += [points[face[i], 2], points[face[j], 2], None]

    # Tracer les points
    point_trace = go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=5),
        name='Points'
    )

    # Tracer les arêtes
    edge_trace = go.Scatter3d(
        x=edges_x, y=edges_y, z=edges_z,
        mode='lines',
        line=dict(color='black', width=2),
        name='Arêtes'
    )

    # Créer la figure
    fig = go.Figure(data=[point_trace, edge_trace])
    
    # Ajout d'un titre et affichage
    fig.update_layout(title=title)
    fig.show()

    