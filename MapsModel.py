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

        finestMesh.simplicies['faces'] = list(set(sorted(self.list_faces)))
        # print("Initial faces : ")
        # for (index,face) in enumerate(finestMesh.simplicies['faces']):
        #     print(index," ",face)
        finestMesh.simplicies['edges'] = self.edges
        finestMesh.neighbors = self.neighbors

        return finestMesh
    
    def facesToList(self):
        listFaces = []
        for face in self.faces:
            listFace = sorted([face.a, face.b, face.c])
            listFaces.append(tuple(listFace))
        self.list_faces = list(set(sorted(listFaces)))
        # return list(set(sorted(self.list_faces)))
    
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
        # print("p0 = ",p0)
        # print("p1 = ",p1)
        # print("p2 = ",p2)
        base = p1 - p0
        milieu_segment = (p0 + p1)/2
        
        hauteur = p2 - milieu_segment
        # print(base)
        # print(hauteur)
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
                if -np.inf in mesh.points[face[0]] or -np.inf in mesh.points[face[1]] or -np.inf in mesh.points[face[2]]:
                    print(face)
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

    def getVerticesToRemove(self, verticesInMesh, mesh:Mesh,maxNeighborsNum:int=12,_lambda:float= 1/2) -> List[int]:

        # print("Vertices to remove computation")
        
        selectedVertices = []
        
        for idx, vertex in enumerate(mesh.simplicies['vertices']):
            if verticesInMesh[idx] == 1:
                nbNeighbors:int = mesh.getNumberOfNeighbors(vertex)
                if nbNeighbors <= maxNeighborsNum and nbNeighbors > 0:
                    selectedVertices.append(vertex)
        #print(selectedVertices)
        areas,maxArea,curvatures,maxCurvature = self.getAreasAndCurvatures(mesh=mesh,selectedVertices=selectedVertices)

        supressionOrder = []
        
        for indx,vertex in enumerate(selectedVertices):

            weight = _lambda * areas[indx,0]/maxArea + (1-_lambda) * curvatures[indx,0]/maxCurvature
            supressionOrder.append([vertex,weight])

        supressionOrder.sort(key=lambda x: x[1],reverse=True)

        verticesToRemove = [e[0] for e in supressionOrder]

        return verticesToRemove
    
    """
        Étant donné un maillage de départ, retourne la liste des maillages jusqu'au "Domain Base".
    """
    def getMeshHierarchy(self,initialMesh:Mesh,maxNeighborsNum = 12) -> List[Mesh]:

        # initialMesh.simplicies['faces'] = sorted(initialMesh.simplicies['faces'])
        # for face in initialMesh.simplicies['faces']:
        #     print(face)
        # input()
        
        numStep:int = initialMesh.stepNum

        # Liste des maillages obtenues après les simplifications successives
        meshHierarchy:List[Mesh] = [initialMesh.copy()]

        currentMesh:Mesh = initialMesh

        verticesInMesh = [1] * len(currentMesh.simplicies['vertices'])

        operations = []

        # Pour chaque étape
        for l in range(numStep-1,-1,-1):

            # if l == 0:
            #     print("At l == 0, faces with 474",currentMesh.getFacesWithVertex(474))

            currentMesh.currentStep = l
            verticesToRemove = self.getVerticesToRemove(verticesInMesh, mesh=currentMesh,maxNeighborsNum = maxNeighborsNum)

            i = 0

            operations_l = []

            # Tant que le maillage courant contient des sommets "supprimables"
            while len(verticesToRemove) != 0:
                
                printer=True

                # if printer:print('1')

                vertexToRemove = verticesToRemove[0]
                verticesInMesh[vertexToRemove] = 0
                # print("VERTEX")
                # print(vertexToRemove)
                # if vertexToRemove==454:
                #     print('VERTEX 454 DETECTED at l=',l)

                holedRegion:HoledRegion = HoledRegion(vertexToRemove=vertexToRemove,mesh=currentMesh)

                # if printer:print('1.5')

                # Avant de supprimer quoi que ce soit dans les listes de faces, d'arrêtes et de sommets du maillage courant, on calcule les
                # nouvelles arrêtes et faces qu'il faudra ajouter au maillage courant après la suppression du sommet sélectionné
                newEdges,newFaces = holedRegion.getNewSimplices()

                # if printer:print('2')

                # Supprimer v de la liste des sommets du maillage courant
                # ## Supprimer v de la liste des points du maillage
                #currentMesh.points[vertexToRemove] = np.array([-np.inf,-np.inf,-np.inf])
                # ## Supprimer de la liste des sommets du maillage
                #currentMesh.simplicies['vertices'][vertexToRemove] = None

                # Récupérer la liste des voisins de v
                selectedVertexNeighbors = currentMesh.getNeighbors(vertexId=vertexToRemove).copy()

                #la disparition est bien ici
                selectedVertexNeighbors.append(vertexToRemove)

                # Supprimer les arrêtes auxquelles appartient v
                verticesToRemove = [v for v in verticesToRemove if v not in selectedVertexNeighbors]           

                # Ajouter les nouvelles arrêtes et faces au maillage courrant
                for edge in newEdges:
                    currentMesh.simplicies['edges'].append(edge)
                    currentMesh.neighbors[edge[0]].append(edge[1])
                    currentMesh.neighbors[edge[1]].append(edge[0])

                for face in newFaces:
                    if face not in currentMesh.simplicies['faces']: 
                        operations_l.append(('new_face', 0, obja.Face(face[0],face[1],face[2])))
                    currentMesh.simplicies['faces'].append(face)

                edgesWithSelectedVertex = currentMesh.getEdgesWithVertex(vertexId=vertexToRemove)
                # print("edges removed", edgesWithSelectedVertex)

                for edge in edgesWithSelectedVertex:
                    # print(edge)
                    currentMesh.simplicies['edges'].remove(edge)

                # Supprimer les faces auxquelles appartient v
                facesWithSelectedVertex = currentMesh.getFacesWithVertex(vertexId=vertexToRemove)

                # if vertexToRemove==107:
                #     for face in facesWithSelectedVertex:
                #         print("At l=0, face ",face," is in facesWithSelectedVertex")

                # if vertexToRemove==107:

                #     print("CURRENT MESH AT STAGE 0")
                #     print(print('\n',"points = \n",currentMesh.points,'\n'))
                #     print(print('\n',"vertices = \n",currentMesh.simplicies['vertices'],'\n'))
                #     print(print('\n',"edges = \n",currentMesh.simplicies['edges'],'\n'))
                #     print(print('\n',"faces = \n",currentMesh.simplicies['faces'],'\n'))
                #     print(print('\n',"neighbors = \n",currentMesh.neighbors,'\n'))

                #     for face in currentMesh.simplicies['faces']:
                #         if face[0] == 107 or face[1]==107 or face[2]==107:
                #             print("At l=0, face, ",face," is in current mesh")

                # if vertexToRemove == 454:
                #     print("faces with 454 before removal :",currentMesh.getFacesWithVertex(454))

                for face in facesWithSelectedVertex:
                    if face in currentMesh.simplicies['faces']: 

                        # On ajoute la face supprimée à la liste des opérations du l-ième maillage
                        operations_l.append(('face', currentMesh.simplicies['faces'].index(face), obja.Face(face[0],face[1],face[2])))
                        
                        currentMesh.simplicies['faces'].remove(face)
                        # if vertexToRemove==454:
                        #     print(currentMesh.simplicies['faces'].index(face))

                # if vertexToRemove == 454:
                #     print("faces with 454 after removal :",currentMesh.getFacesWithVertex(454))

                currentMesh.neighbors[vertexToRemove] = []
                for vertex in selectedVertexNeighbors:
                    if vertex != vertexToRemove:
                        currentMesh.neighbors[vertex].remove(vertexToRemove)

                ######### PHASE DE TEST #########
                # facesWithSelectedVertex = currentMesh.getFacesWithVertex(vertexId=vertexToRemove)

                # testList = []
                # for face in currentMesh.simplicies['faces']:
                #     if face[0] == vertexToRemove or face[1]==vertexToRemove or face[2]==vertexToRemove:
                #         testList.append(face)

                # if sorted(testList) != sorted(currentMesh.getFacesWithVertex(vertexToRemove)):

                #     print(vertexToRemove," neighbors : ",currentMesh.neighbors[vertexToRemove])
                #     for neighbor in currentMesh.neighbors[vertexToRemove]:
                #         print("           ",neighbor," neighbors : ",currentMesh.neighbors[neighbor])

                #     print("vertexToRemove = ",vertexToRemove,"   stage = ",l)
                #     print(sorted(testList))
                #     print(sorted(currentMesh.getFacesWithVertex(vertexToRemove)))
                #     input()

                ######### PHASE DE TEST #########   

                # ATTENTION il faudra surement convertir currentMesh.points[vertexToRemove] en numpy,
                # actuellement c'est juste une liste de float
                operations_l.append(('vertex', vertexToRemove, currentMesh.points[vertexToRemove])) 

                # if vertexToRemove==454:
                #     print("At l == ",l," vertexToRemove == 454, faces with 474",currentMesh.getFacesWithVertex(474))
                #     print("At l == ",l," vertexToRemove == 454, faces with 454",currentMesh.getFacesWithVertex(454))

            # On ajoute le maillage obtenu à la hierarchie des maillages
            meshHierarchy.append(currentMesh.copy())

            # La ligne suivante ne devrait pas être utile, mais elle l'est pourtant...
            currentMesh.simplicies['faces'] = list(set(currentMesh.simplicies['faces']))

            # On ajoute la liste des opérations du l-ième malliage à la liste globale
            operations.append(operations_l)

        # On ajoute dans la liste des opérations globale, la liste des opérations nécessaires pour
        # retrouver le maillage de base : "Base Domain" 
        operationBaseDomain = []
        
        baseDomain = currentMesh

        for face in baseDomain.simplicies['faces']:
            # if face[0] == 103 or face[1]==103 or face[2]==103:
            #     print("face, ",face[0]," ",face[1]," ",face[2]," is in base domain")
            operationBaseDomain.append(('face', baseDomain.simplicies['faces'].index(face), obja.Face(face[0],face[1],face[2])))

        for vertex in baseDomain.simplicies['vertices']:
            if vertex is not None:
                
                operationBaseDomain.append(('vertex', vertex, baseDomain.points[vertex]))

        operations.append(operationBaseDomain)

        # for operation in operations:
        #     print(operation)

        for operation in operations:
            operation.reverse()

        return meshHierarchy,operations
    

    
