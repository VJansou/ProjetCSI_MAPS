from typing import List,Dict
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import copy

class Mesh:
    def __init__(self,stepNum:int):
        self.stepNum = stepNum
        self.currentStep = stepNum
        self.points:List[np.ndarray] = []
        self.simplicies:Dict[List[int]] = {'vertices':[],'edges':[],'faces':[]}
        self.neighbors = {}
    
    def copy(self):
        # Créer une copie de l'objet Mesh
        return copy.deepcopy(self)  # Remplace par une copie réelle des attributs de Mesh
    
    def createNeighborsDict(self):
        self.neighbors = {}
        for vertex in self.simplicies['vertices']:
            self.neighbors[vertex] = []
        for edge in self.simplicies['edges']:
            self.neighbors[edge[0]].append(edge[1])
            self.neighbors[edge[1]].append(edge[0])
        return self.neighbors


    """
        Retourne le nombre de voisin d'un sommet du maillage ( = card(1-Ring) ).

        Args : vertexId:int, l'indice associé au sommet en question.
        
        Returns : int, le nombre de voisins.
    """
    def getNumberOfNeighbors(self,vertexId:int) -> int:
        #print(vertexId)
        #print("neigh = ")
        #print(self.neighbors[vertexId])
        #print('\n')
        return len(self.neighbors[vertexId])
    
    """
        Retourne la liste des arrêtes du maillage contenant un certain sommet.

        Args : vertexId:int, l'indice associé au sommet en question.
        
        Returns : List[List[int]], la liste des arrêtes, une arrête étant une liste de deux entiers : 
                    les indices associés aux deux sommets de l'arrête.
    """
    def getEdgesWithVertex(self,vertexId):
        edges = []
        #print(self.neighbors)
        for vertex in self.neighbors[vertexId]:
            if vertex < vertexId:
                edges.append((vertex, vertexId))
            else:
                edges.append((vertexId, vertex))

        return edges
    
    """
        Retourne la liste des sommets voisins d'un certain sommet ( = sommet du 1-ring d'un sommet).

        Args : vertexId:int, l'indice associé au sommet en question.
        
        Returns : List[int], la liste des indices associés sommets.
    """
    def getNeighbors(self,vertexId:int) -> List[int]:
        return self.neighbors[vertexId]

    """
        Retourne la liste des faces du maillage contenant un certain sommet.

        Args : vertexId:int, l'indice associé au sommet en question.
        
        Returns : List[List[int]], la liste des faces, une face étant une liste de trois entiers : 
                    les indices associés trois aux sommets de la face.
    """
    def getFacesWithVertex(self, vertexId):
        faces = set()
        neighbors = self.neighbors[vertexId]
        for neighbor in neighbors:
            subNeighbors = self.neighbors[neighbor]
            for subNeighbor in subNeighbors:
                if subNeighbor in neighbors:
                    face = sorted([vertexId, neighbor, subNeighbor])
                    faces.add(tuple(face))
        return list(faces)
    
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
            externalEdges.append(tuple(vertex for vertex in face if vertex != centralVertex))
        
        return externalEdges

    """
        Retourne la liste des voisins d'un sommet dans.

        Args : centralVertex:int, l'indice associé au sommet en question.
        
        Returns : List[List[int]], la liste des arrêtes, une arrête étant une liste de deux entiers : 
                    les indices associés aux deux sommets de l'arrête.
    """
    def getExternalVerticesInCyclicOrder(self,vertexId:int):

        isBorderVertex = False
        edgesInOrder = []
        edges = self.get1RingExternalEdges(vertexId)
        neighbors = {}
        for edge in edges:
            neighbors.setdefault(edge[0], []).append(edge[1])
            neighbors.setdefault(edge[1], []).append(edge[0])

        currentVertex = next((vertex for vertex, neighborList in neighbors.items() if len(neighborList) == 1), None)
        if currentVertex is None:
            if len(edges) > 0:
                currentVertex = edges[0][0]
        else:
            isBorderVertex = True
        
        edgesInOrder.append(currentVertex)
        while True:
            neighborList = neighbors[currentVertex]
            nextVertex = next((v for v in neighborList if v not in edgesInOrder), None)
            if nextVertex is None:  
                break
            
            edgesInOrder.append(nextVertex)
            currentVertex = nextVertex

        return edgesInOrder, isBorderVertex
                


    
    """
        Affiche dans une fenêtre de matplotlib le maillage. La figure affiché est une figure 2D

        Args : title:str, le titre de la figure.
        
        Returns : None
    """

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

        #print("POINTS")
        #print(points)
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

    