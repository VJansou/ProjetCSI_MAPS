import obja
import numpy as np
from typing import List,Dict,Tuple
from scipy.spatial import Delaunay,ConvexHull
from math import pi,sqrt
from numpy import inf
import matplotlib.pyplot as plt
from Mesh import Mesh
from HoledRegion import HoledRegion

#tests purpose
import sys


class MapsModel(obja.Model):
    
    def __init__(self,L):
        super().__init__()
        self.L = L
        self.liste_faces = self.facesToList()
        self.list_faces = self.facesToList()
        self.edges = self.createEdgesList()
        self.neighbours = self.createNeighborsDict()
        self.status_vertices = {i: 1 for i in range(len(self.vertices))} # 1 = sommet removable, 0 = sommet unremovable
        self.status_edges = {i: 1 for i in range(len(self.edges))} # 1 = normal edge, 0 = feature edge
        self.taille_completeL = self.vertices

    def parse_file(self, path):
        super().parse_file(path)
        self.facesToList()
        
        self.createEdgesList()
        self.createNeighborsDict()


    """
        Retourne le maillage associé à un modèle déduit d'un fichier .obj

        Args: None

        Return: Mesh
    """
    def model2Mesh(self) -> Mesh:
        finestMesh = Mesh(stepNum=self.L)

        # Parcours des sommets du modèle, on récupère l'indice d'un sommet dans self.vertices et son indice grâce à index
        for index,vertex in enumerate(self.vertices):
            coordonates = vertex
            finestMesh.points.append(coordonates)
            finestMesh.simplicies['vertices'].append(index)

        finestMesh.simplicies['faces'] = self.list_faces
        finestMesh.simplicies['edges'] = self.edges
        finestMesh.neighbors = self.neighbors

        return finestMesh
    
    def facesToList(self):
        listFaces = []
        for face in self.faces:
            listFace = sorted([face.a, face.b, face.c])
            listFaces.append(tuple(listFace))
        self.list_faces = listFaces
        return self.list_faces
    
    def createEdgesList(self):
        self.edges = set()
        for face in self.list_faces:
            self.edges.add((face[0], face[1]))
            self.edges.add((face[0], face[2]))
            self.edges.add((face[1], face[2]))
        self.edges = list(self.edges)
        return self.edges
    
    def createNeighborsDict(self):
        self.neighbors = {}
        for index,_ in enumerate(self.vertices):
            self.neighbors[index] = []
        for edge in self.edges:
            self.neighbors[edge[0]].append(edge[1])
            self.neighbors[edge[1]].append(edge[0])
        return self.neighbors
    
    def get1RingExternalEdges(self,centralVertex:int) -> List[List[int]]:

        facesWithCentralVertex = self.getFacesWithVertex(vertexId=centralVertex)
        externalEdges = []

        for face in facesWithCentralVertex:
            externalEdges.append(tuple(vertex for vertex in face if vertex != centralVertex))
        
        return externalEdges
    
    def getExternalVerticesInCyclicOrder(self,vertexId:int) -> List[int]:

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
    

    def getEdgesWithVertex(self,vertexId):
        edges = []
        for vertex in self.neighbors[vertexId]:
            if vertex < vertexId:
                edges.append((vertex, vertexId))
            else:
                edges.append((vertexId, vertex))

        return edges
    
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
    

    def computeArea(self,p0:np.ndarray,p1:np.ndarray,p2:np.ndarray) -> np.ndarray:
        base = p1 - p0
        milieu_segment = (p0 + p1)/2
        
        hauteur = p2 - milieu_segment
        #print(base)
        #print(hauteur)
        return np.linalg.norm(np.cross(base,hauteur)) / 2
    
    def computeCurvature(self,p0:np.ndarray,p1:np.ndarray,p2:np.ndarray) -> float:
        # Pour le calcul d'un angle d'une face, on utilise la loi d'Al-Kashi
        p0p1 = np.sqrt(np.sum((p0-p1)**2))
        p1p2 = np.sqrt(np.sum((p1-p2)**2))
        p0p2 = np.sqrt(np.sum((p0-p2)**2))
        return np.arccos((p0p1**2 + p0p2**2 - p1p2**2)/(2*p0p1*p0p2))

    def getAreasAndCurvatures(self,mesh:Mesh,selectedVertices:List[int]) -> Tuple[np.ndarray,float,np.ndarray,float]:

        nbVertices = len(selectedVertices)

        areas = np.zeros((nbVertices,1))
        curvatures = np.zeros((nbVertices,1))

        maxArea = -inf
        maxCurvature = -inf

        for indx,vertex in enumerate(selectedVertices):
            facesWithVertex = mesh.getFacesWithVertex(vertexId=vertex)

            starArea = 0
            starCurvature = 0

            for face in facesWithVertex:
                starArea = starArea + self.computeArea(p0=mesh.points[face[0]],p1=mesh.points[face[1]],p2=mesh.points[face[2]])
                face_copy = list(face)
                face_copy.remove(vertex)
                starCurvature = starCurvature + self.computeCurvature(p0=mesh.points[vertex],p1=mesh.points[face_copy[0]],p2=mesh.points[face_copy[1]])

            areas[indx,0] = starArea
            curvatures[indx,0] = starCurvature

            if starArea > maxArea:
                maxArea = starArea

            if starCurvature > maxCurvature:
                maxCurvature = starCurvature

        return areas,maxArea,curvatures,maxCurvature

    def computevertexfeatureedge(self,mesh:Mesh):

        for vertex in mesh.simplicies["vertices"]:
            edges = mesh.getEdgesWithVertex(vertex)
            

    def calculate_normal(self,v1, v2, v3):
        """
        Calcule la normale d'une face triangulaire définie par 3 sommets.
        """

        u = np.array(v2) - np.array(v1)
        v = np.array(v3) - np.array(v1)
        
        # Produit vectoriel pour obtenir la normale
        normal = np.cross(u, v)
        
        # Normalisation
        norm = np.linalg.norm(normal)
        if norm == 0:
            return np.array([0, 0, 0])  # Cas dégénéré (points colinéaires)
        
        return normal / norm

    def dihedral_angle(self,normal1, normal2, shared_edge_vector):
        """
        Calcule l'angle dièdre entre deux normales et le vecteur de l'arête partagée.
        """

        shared_edge_vector = shared_edge_vector / np.linalg.norm(shared_edge_vector)
        

        cos_theta = np.dot(normal1, normal2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Éviter les erreurs numériques hors [-1, 1]
        

        angle = np.arccos(cos_theta)
        
        # On calcule un vecteur "binormal" pour déterminer si l'angle est obtus ou aigu
        cross_product = np.cross(normal1, normal2)
        orientation = np.dot(cross_product, shared_edge_vector)
        
        if orientation < 0:
            
            angle = -angle
        
        return angle

    def get_face_from_edge(self,edge):
        faces = []
        compteur = 0
        for face in self.liste_faces:
            if edge[0] in face and edge[1] in face and compteur < 2:
                faces.append(face)
                compteur += 1

        return faces
    # obtenir l'angle des normales des faces d'une edge
    def calcul_diedre(self,edge):

        
        faces = self.get_face_from_edge(edge)
        diedre = -inf
        if len(faces) == 2:
            normal1 = self.calculate_normal(self.vertices[faces[0][0]], self.vertices[faces[0][1]], self.vertices[faces[0][2]])
            normal2 = self.calculate_normal(self.vertices[faces[1][0]], self.vertices[faces[1][1]], self.vertices[faces[1][2]])
            shared_edge_vector = self.vertices[edge[1]] - self.vertices[edge[0]]
            diedre = self.dihedral_angle(normal1, normal2, shared_edge_vector)

        return diedre
    # selopn l'oangle considere comme unremovable
    def feature_ou_pas(self,ind_edge,edge, threshold_angle = np.pi/6):
        diedre = self.calcul_diedre(edge)
        if diedre > threshold_angle:
            self.status_edges[ind_edge] = 0
            


    def getVerticesToRemove(self,mesh:Mesh,maxNeighborsNum:int=12,_lambda:float= 1/2, threshold_curv = np.pi/6) -> List[int]:

        # print("Vertices to remove computation")
        
        selectedVertices = []
        
        for vertex in mesh.simplicies['vertices']:
            if vertex is not None:
                nbNeighbors:int = mesh.getNumberOfNeighbors(vertex)
                if nbNeighbors <= maxNeighborsNum and nbNeighbors > 0:
                    selectedVertices.append(vertex)
        ##gerer les feature edges

        for ind_edge,edge in enumerate(mesh.simplicies['edges']):
            self.feature_ou_pas(ind_edge,edge)
            if ind_edge not in self.status_edges:
                print(f"Key {ind_edge} not found in status_edges")
                continue
            # 1 = removable, 0 = unremovable
            if self.status_edges[ind_edge] == 0:
                if edge[0] in selectedVertices:
                    selectedVertices.remove(edge[0])
                if edge[1] in selectedVertices:
                    selectedVertices.remove(edge[1])


        #print(selectedVertices)

        areas,maxArea,curvatures,maxCurvature = self.getAreasAndCurvatures(mesh=mesh,selectedVertices=selectedVertices)

        supressionOrder = []
        
        for indx,vertex in enumerate(selectedVertices):
            if curvatures[indx,0] > threshold_curv:
                weight = _lambda * areas[indx,0]/maxArea + (1-_lambda) * curvatures[indx,0]/maxCurvature
                supressionOrder.append([vertex,weight])

        supressionOrder.sort(key=lambda x: x[1],reverse=True)

        verticesToRemove = [e[0] for e in supressionOrder]

        return verticesToRemove
    
    """
        Étant donné un maillage de départ, retourne la liste des maillages jusqu'au "Domain Base".
    """
    def getMeshHierarchy(self,initialMesh:Mesh,maxNeighborsNum = 12) -> List[Mesh]:
        
        numStep:int = initialMesh.stepNum

        # Liste des maillages obtenues après les simplifications successives
        meshHierarchy:List[Mesh] = [initialMesh.copy()]
        currentMesh:Mesh = initialMesh

        #testVerticesList = [[3],[15,0]] #[[12,14],[9,3]]
        #testVerticesIndex = 0
        
        #currentMesh.plot(title='level '+str(numStep-1))
        # Pour chaque étape
        for l in range(numStep-1,-1,-1):
            currentMesh.currentStep = l
            verticesToRemove = self.getVerticesToRemove(mesh=currentMesh,maxNeighborsNum = maxNeighborsNum )
            
            # Filter keys with a list length greater than N
            #filtered_keys = {key: value for key, value in currentMesh.neighbors.items() if len(value) > maxNeighborsNum}
            #print("vertices with more than max neighbors")
            #print(filtered_keys)
            #print('\n')
#
            #print('##############################################################')
            #print('Mesh oooo', len(currentMesh.simplicies['vertices']))
            #print("coucoucouc",verticesToRemove)
            #print("coouc",len(verticesToRemove))
            #print('##############################################################')
            #print('##############################################################')
            #print('STEP ',l)
#
            #print("VERTICES TO REMOVE NUMBER = \n",len(verticesToRemove))

            i = 0
            Liste_Point_A_Supp = []
            # testVertices:List[int] = testVerticesList[testVerticesIndex]

            # Tant que le maillage courant contient des sommets "supprimables"
            while len(verticesToRemove) != 0:
                
                printer=True

                if printer:print('1')

                vertexToRemove = verticesToRemove[0]
                print("VERTEX")
                print(vertexToRemove)


                holedRegion:HoledRegion = HoledRegion(vertexToRemove=vertexToRemove,mesh=currentMesh)

                if printer:print('1.5')

                # Avant de supprimer quoi que ce soit dans les listes de faces, d'arrêtes et de sommets du maillage courant, on calcule les
                # nouvelles arrêtes et faces qu'il faudra ajouter au maillage courant après la suppression du sommet sélectionné
                newEdges,newFaces = holedRegion.getNewSimplices()

                if printer:print('2')

                # Supprimer v de la liste des sommets du maillage courant
                # ## Supprimer v de la liste des points du maillage
                vertexIndex = currentMesh.simplicies['vertices'].index(vertexToRemove)
                currentMesh.points[vertexIndex] = np.array([-np.inf,-np.inf,-np.inf])
                # ## Supprimer de la liste des sommets du maillage


                currentMesh.simplicies['vertices'][vertexToRemove] = None

                # Récupérer la liste des voisins de v

                selectedVertexNeighbors = currentMesh.getNeighbors(vertexId=vertexToRemove).copy()

                # Marquer les sommets voisins comme "non-supprimable" et les supprimer de la liste des sommets "supprimables"
                # for vertexId in selectedVertexNeighbors:
                #     try:
                #         marked_vertices['removable'].remove(vertexId)
                #     except ValueError:
                #         pass
                #     if vertexId not in marked_vertices['unremovable']:
                #         marked_vertices['unremovable'].append(vertexId)

                ##################
                # print('vertexID',vertexID)
################ probleme ici putain les selected vertex neighbors devrait apparaitre dans to be reomved au moins a la prmeiere iteration
                # boucle un peu long mais il y'a t'il mieux ?

                #la disparition est bien ici
                selectedVertexNeighbors.append(vertexToRemove)

                # Supprimer les arrêtes auxquelles appartient v
                verticesToRemove = [v for v in verticesToRemove if v not in selectedVertexNeighbors]

                edgesWithSelectedVertex = currentMesh.getEdgesWithVertex(vertexId=vertexToRemove)
                print("edges removed", edgesWithSelectedVertex)
                #print("current mesh", currentMesh.simplicies['edges'])
                # soucis de nonapparition des sommets supprimés

                for edge in edgesWithSelectedVertex:
                    print(edge)
                    currentMesh.simplicies['edges'].remove(edge)

                # Supprimer les faces auxquelles appartient v
                facesWithSelectedVertex = currentMesh.getFacesWithVertex(vertexId=vertexToRemove)
                for face in facesWithSelectedVertex:
                    if face in currentMesh.simplicies['faces']:
                        currentMesh.simplicies['faces'].remove(face)


                # Ajouter les nouvelles arrêtes et faces au maillage courrant
                for edge in newEdges:
                    currentMesh.simplicies['edges'].append(edge)
                    currentMesh.neighbors[edge[0]].append(edge[1])
                    currentMesh.neighbors[edge[1]].append(edge[0])


                for face in newFaces:
                    currentMesh.simplicies['faces'].append(face)

                currentMesh.neighbors[vertexToRemove] = []
                for vertex in selectedVertexNeighbors:
                    if vertex != vertexToRemove:
                        currentMesh.neighbors[vertex].remove(vertexToRemove)

                
            #currentMesh.plot(title='level '+str(l))

                # TEST : on regarde les points suppr.
                #currentMesh.plot(f"loupe n° {l} nb sommets  {len(verticestoremove)} ")
                
                #i = i + 1
            

            # on ajoute le maillage obtenu à la hierarchie des maillages
            meshHierarchy.append(currentMesh.copy())

            #testVerticesIndex = testVerticesIndex + 1
        print("stratus_pck bizatrre", self.status_edges)
        print("param du mesh", self.list_faces,self.edges)
        print("neighbours", self.neighbors)
        return meshHierarchy
    

    
