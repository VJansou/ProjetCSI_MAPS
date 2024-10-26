import obja
import numpy as np
from typing import List,Dict,Tuple
from scipy.spatial import Delaunay,ConvexHull
from math import pi,sqrt
from numpy import inf
import matplotlib.pyplot as plt
from Mesh import Mesh
from HoledRegion import HoledRegion


class MapsModel(obja.Model):
    
    def __init__(self,L):
        super().__init__()
        self.L = L
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
        for edge in self.edges:
            self.neighbors.setdefault(edge[0], []).append(edge[1])
            self.neighbors.setdefault(edge[1], []).append(edge[0])
        return self.neighbors
    
    def get1RingExternalEdges(self,centralVertex:int) -> List[List[int]]:

        facesWithCentralVertex = self.getFacesWithVertex(vertexId=centralVertex)
        externalEdges = []

        for face in facesWithCentralVertex:
            externalEdges.append(tuple(vertex for vertex in face if vertex != centralVertex))
        
        return externalEdges
    
    def getExternalEdgesInCyclicOrder(self,vertexId:int) -> List[int]:

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
                face_copy = face.copy()
                face_copy.remove(vertex)
                starCurvature = starCurvature + self.computeCurvature(p0=mesh.points[vertex],p1=mesh.points[face_copy[0]],p2=mesh.points[face_copy[1]])

            areas[indx,0] = starArea
            curvatures[indx,0] = starCurvature

            if starArea > maxArea:
                maxArea = starArea

            if starCurvature > maxCurvature:
                maxCurvature = starCurvature

        return areas,maxArea,curvatures,maxCurvature

    def getVerticesToRemove(self,mesh:Mesh,maxNeighborsNum:int=12,_lambda:float= 1/2) -> List[int]:

        # print("Vertices to remove computation")
        
        selectedVertices = []

        for vertex in mesh.simplicies['vertices']:
            nbNeighbors:int = mesh.getNumberOfNeighbors(vertex)
            if nbNeighbors <= maxNeighborsNum and nbNeighbors > 0:
                selectedVertices.append(vertex)

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

        numStep:int = initialMesh.stepNum

        # Liste des maillages obtenues après les simplifications successives
        meshHierarchy:List[Mesh] = [initialMesh.copy()]
        currentMesh:Mesh = initialMesh

        testVerticesList = [[3],[15,0]] #[[12,14],[9,3]]
        testVerticesIndex = 0

        # Pour chaque étape
        for l in range(numStep-1,-1,-1):

            verticesToRemove = self.getVerticesToRemove(mesh=currentMesh,maxNeighborsNum = maxNeighborsNum )

            print('##############################################################')
            print('Mesh oooo', len(currentMesh.simplicies['vertices']))
            print("coucoucouc",verticesToRemove)
            print("coouc",len(verticesToRemove))
            print('##############################################################')
            print('##############################################################')
            print('STEP ',l)

            print("VERTICES TO REMOVE NUMBER = \n",len(verticesToRemove))

            i = 0
            Liste_Point_A_Supp = []
            # testVertices:List[int] = testVerticesList[testVerticesIndex]

            # Tant que le maillage courant contient des sommets "supprimables"
            while len(verticesToRemove) != 0:

                # print('##############################################################')
                print('VERTICES N',i)

                printer=True
                # if i==60:
                #     printer=True

                # Selectionner aléatoirement un sommet v non-marqué dont le degré de sorti est inférieur à 12
                # while(True):
                #     selectedVertex = choice(marked_vertices['removable'])

                #     # Tester si le sommet sélectionné a au moins de 12 voisins
                #     if currentMesh.getNumberOfNeighbors(vertexId=selectedVertex) <= 4:
                #         break

                if printer:print('1')

                vertexToRemove = verticesToRemove[0]


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

                selectedVertexNeighbors = currentMesh.getNeighbors(vertexId=vertexToRemove)

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
                apres_nettoyage = [v for v in verticesToRemove if v not in selectedVertexNeighbors]
                print("longueur", len(verticesToRemove))


                print("ttooo ", [v for v in verticesToRemove if v in selectedVertexNeighbors])
                Liste_Point_A_Supp = Liste_Point_A_Supp + [v for v in verticesToRemove if v in selectedVertexNeighbors]
                print("Liste_Point_A_Supp",len(Liste_Point_A_Supp))
                print("Liste_Point_A_Supp en set",len(set(Liste_Point_A_Supp)))

                # Supprimer les arrêtes auxquelles appartient v

                #okou
                verticesToRemove = apres_nettoyage
                ##

                edgesWithSelectedVertex = currentMesh.getEdgesWithVertex(vertexId=vertexToRemove)
                print("edges with selected vertex nu 0", edgesWithSelectedVertex)
                #print("current mesh", currentMesh.simplicies['edges'])
                # soucis de nonapparition des sommets supprimés
                try:
                    verticesToRemove.pop(0)
                except Exception as e:
                    print("message",e)
                # potentiel soucis dans edge removal
                print("yo le sang je capte r", len(verticesToRemove))



                print("edges with selected vertex nu 1", edgesWithSelectedVertex)
                for edge in edgesWithSelectedVertex:
                    currentMesh.simplicies['edges'].remove(edge)


                print("edges with selected vertex nu 2", edgesWithSelectedVertex)
                # Supprimer les faces auxquelles appartient v
                facesWithSelectedVertex = currentMesh.getFacesWithVertex(vertexId=vertexToRemove)
                for face in facesWithSelectedVertex:
                    currentMesh.simplicies['faces'].remove(face)


                # Ajouter les nouvelles arrêtes et faces au maillage courrant
                for edge in newEdges:
                    currentMesh.simplicies['edges'].append(edge)


                for face in newFaces:
                    currentMesh.simplicies['faces'].append(face)

                # TEST : on regarde les points suppr.
                currentMesh.plot(f"loupe n° {l} nb sommets  {len(verticesToRemove)} ")
                #i = i + 1


            # On ajoute le maillage obtenu à la hierarchie des maillages
            meshHierarchy.append(currentMesh.copy())

            testVerticesIndex = testVerticesIndex + 1

        return meshHierarchy