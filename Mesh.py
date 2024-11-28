from typing import List,Dict
import numpy as np
import plotly.graph_objects as go
import copy
import obja
import decimate

class Mesh:
    """
    Classe représentant un maillage 3D pour une étape.
    """
    def __init__(self,stepNum:int):
        """
        Intialise un objet Mesh.
        """
        self.stepNum = stepNum
        self.currentStep = stepNum
        self.points:List[np.ndarray] = []
        self.simplicies:Dict[List[int]] = {'vertices':[],'edges':[],'faces':[]}
        self.neighbors = {}
    
    def copy(self):
        """
        Retourne une copie de l'objet Mesh.
        """
        return copy.deepcopy(self)  # Remplace par une copie réelle des attributs de Mesh
    
    def createNeighborsDict(self):
        """
            Crée un dictionnaire des voisins de chaque sommet du maillage.

            Returns : Dict[int,List[int]], le dictionnaire des voisins, un voisin étant une liste d'entiers : 
                        les indices associés aux sommets voisins.
        """
        self.neighbors = {}
        for vertex in self.simplicies['vertices']:
            self.neighbors[vertex] = []
        for edge in self.simplicies['edges']:
            self.neighbors[edge[0]].append(edge[1])
            self.neighbors[edge[1]].append(edge[0])
        return self.neighbors


    def getNumberOfNeighbors(self,vertexId:int) -> int:
        """
        Retourne le nombre de voisin d'un sommet du maillage ( = card(1-Ring) ).

        Args : vertexId:int, l'indice associé au sommet en question.
        
        Returns : int, le nombre de voisins.
        """
        return len(self.neighbors[vertexId])
    

    def getEdgesWithVertex(self,vertexId):
        """
        Retourne la liste des arrêtes du maillage contenant un certain sommet.

        Args : vertexId:int, l'indice associé au sommet en question.
        
        Returns : List[List[int]], la liste des arrêtes, une arrête étant une liste de deux entiers : 
                    les indices associés aux deux sommets de l'arrête.
        """
        edges = []
        for vertex in self.neighbors[vertexId]:
            if vertex < vertexId:
                edges.append((vertex, vertexId))
            else:
                edges.append((vertexId, vertex))

        return edges
    

    def getNeighbors(self,vertexId:int) -> List[int]:
        """
        Retourne la liste des sommets voisins d'un certain sommet ( = sommet du 1-ring d'un sommet).

        Args : vertexId:int, l'indice associé au sommet en question.
        
        Returns : List[int], la liste des indices associés sommets.
        """
        return self.neighbors[vertexId]

    
    def getFacesWithVertex(self, vertexId):
        """
        Retourne la liste des faces du maillage contenant un certain sommet.

        Args : vertexId:int, l'indice associé au sommet en question.
        
        Returns : List[List[int]], la liste des faces, une face étant une liste de trois entiers : 
                    les indices associés trois aux sommets de la face.
        """
        faces = set()
        for face in self.simplicies['faces']:
            if face[0] == vertexId or face[1] == vertexId or face[2] == vertexId:
                face = sorted(face)
                faces.add(tuple(face))
        return list(faces)
    

    def get1RingExternalEdges(self,centralVertex:int) -> List[List[int]]:
        """
        Retourne la liste des arrêtes extérieures du 1-ring d'un sommet.

        Args : centralVertex:int, l'indice associé au sommet en question.
        
        Returns : List[List[int]], la liste des arrêtes, une arrête étant une liste de deux entiers : 
                    les indices associés aux deux sommets de l'arrête.
        """

        facesWithCentralVertex = self.getFacesWithVertex(vertexId=centralVertex)
        externalEdges = []

        for face in facesWithCentralVertex:
            externalEdges.append(tuple(vertex for vertex in face if vertex != centralVertex))
        
        return externalEdges


    def getExternalVerticesInCyclicOrder(self,vertexId:int):
        """
        Retourne la liste des voisins d'un sommet dans l'ordre cyclique.

        Args : centralVertex:int, l'indice associé au sommet en question.
        
        Returns : List[List[int]], la liste des arrêtes, une arrête étant une liste de deux entiers : 
                    les indices associés aux deux sommets de l'arrête.
        """

        isBorderVertex = False
        edgesInOrder = []
        # On récupère les arrêtes extérieures du 1-ring du sommet
        edges = self.get1RingExternalEdges(vertexId)

        # On crée un dictionnaire des voisins de chaque sommet
        neighbors = {}
        for edge in edges:
            neighbors.setdefault(edge[0], []).append(edge[1])
            neighbors.setdefault(edge[1], []).append(edge[0])

        # On récupère un bord de l'étoile s'il existe : 
        # cela permet de partir d'un sommet sur le bord s'il y en a un et donc de s'assurer de parcourir toute l'étoile
        currentVertex = next((vertex for vertex, neighborList in neighbors.items() if len(neighborList) == 1), None)
        if currentVertex is None: # S'il n'y a pas de bord, on prend un sommet quelconque
            if len(edges) > 0:
                currentVertex = edges[0][0]
        else:
            isBorderVertex = True
        # On ajoute le sommet de départ à la liste des sommets
        edgesInOrder.append(currentVertex)

        # On parcourt les sommets voisins dans l'ordre cyclique
        while True:
            # On récupère les voisins du sommet courant
            neighborList = neighbors[currentVertex]
            # On récupère le prochain sommet à parcourir
            nextVertex = next((v for v in neighborList if v not in edgesInOrder), None)
            # Si on a parcouru tous les sommets, on arrête
            if nextVertex is None:  
                break
            
            # On ajoute le sommet à la liste des sommets
            edgesInOrder.append(nextVertex)

            # On passe au sommet suivant
            currentVertex = nextVertex

        return edgesInOrder, isBorderVertex
                   
    

    def plot(self, title: str, zoomPoint: np.ndarray = None) -> None:
        """
        Affiche dans une fenêtre de matplotlib le maillage. La figure affiché est une figure 2D

        Args : title:str, le titre de la figure.
        """
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

    def mesh2model(self) -> decimate.Decimater:
        """
        Convertit un maillage en un modèle.
        """
        # création d'un nouveau modèle
        model = decimate.Decimater()
        
        # ajouter les sommets du maillage au modèle
        model.vertices = [point for point in self.points]

        # ajouter les faces du maillage au modèle
        model.faces = []
        for face in self.simplicies['faces']:
            model.faces.append(obja.Face(a=face[0], b=face[1], c=face[2]))
        
        # définir le nombre de lignes du modèle
        model.line = len(model.vertices) + len(model.faces)

        return model