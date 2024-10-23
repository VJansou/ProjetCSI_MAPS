import obja
import numpy as np
from typing import List,Dict,Tuple
from scipy.spatial import Delaunay,ConvexHull
from math import pi,acos,sqrt
from numpy import inf
import matplotlib.pyplot as plt
from Mesh import Mesh
from HoledRegion import HoledRegion


class MapsModel(obja.Model):
    
    def __init__(self,L):
        super().__init__()
        self.L = L

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

        # Liste qui contiendra les arrêtes du maillage sous la forme d'une liste [ [id du premier sommet , id du seconde sommet] ... ]
        edges = []
        # Pour remplir cette liste on effectue un parcours sur les faces du modèle
        for face in self.faces:

            if (not [face.a,face.b,face.c] in finestMesh.simplicies['faces'] and 
                not [face.a,face.c,face.b] in finestMesh.simplicies['faces'] and
                not [face.b,face.a,face.c] in finestMesh.simplicies['faces'] and
                not [face.b,face.c,face.a] in finestMesh.simplicies['faces'] and
                not [face.c,face.a,face.b] in finestMesh.simplicies['faces'] and
                not [face.c,face.b,face.a] in finestMesh.simplicies['faces']):

                finestMesh.simplicies['faces'].append([face.a,face.b,face.c])
                edge1 = [face.a,face.b]
                edge2 = [face.b,face.c]
                edge3 = [face.a,face.c]
                edges.append(edge1)
                edges.append(edge2)
                edges.append(edge3)

        # On supprime les doublon en passant par un np.ndarray
        edges = np.unique(np.array(edges),axis=0).tolist()
        edges = self.removeDoubleEdges(edges=edges)

        finestMesh.simplicies['edges'] = edges

        return finestMesh
    
    """
        Étant donné une liste d'arrête, retourne la même liste sans les arrêtes présentent deux fois dans des sens différents.
        Exemple:
            input = [[1,2],[3,4],[2,1],[5,4]]
            output = [[3,4],[2,1],[5,4]]
        
        Args: edges:List[List[int]], la liste des arrêtes de départ

        Return: List[List[int]], la liste des arrêtes finales sans les doublons

    """
    def removeDoubleEdges(self,edges:List[List[int]]) -> List[List[int]]:
        for edge in edges:
            try:
                reversedEdge = edge.copy()
                reversedEdge.reverse()
                edges.remove(reversedEdge)
            except ValueError:
                pass
        return edges

    def getEdgesWithVertex(self,vertexId,edgesList):
        flattenEdges = np.ravel(np.array(edgesList))
        flattenEdges = flattenEdges.tolist()

        indices = []
        start = 0
        allIndicesFound = False

        while not allIndicesFound:
            try:
                index = flattenEdges.index(vertexId,start)
                indices.append(index)
                start = index+1
            except ValueError:
                allIndicesFound = True

        indices = np.array(indices)
        indices = (indices//2).tolist()

        edgesInCyclicOrder = [edgesList[i] for i in indices]

        return edgesInCyclicOrder

    def getFacesFromEdges(self,edgesList):

        vertices = np.unique(np.ravel(np.array(edgesList))).tolist()

        faces = []

        for vertex in vertices:

            # print('vertex : ',vertex)
            
            edgesCollection = self.getEdgesWithVertex(vertex,edgesList)
            
            for edge in edgesCollection:
                # print(edge)
                if edge[0] != vertex: edge.reverse()

            # print('--- edges collection : ',edgesCollection)

            for edge in edgesCollection:
                
                # print('--- --- edge in edgesCollection : ',edge)

                nextEdgesCollection = self.getEdgesWithVertex(edge[1],edgesList)
                nextEdgesCollection.remove(edge)

                # print('--- --- next edges collection : ',nextEdgesCollection)

                coupleEdgesCollection = []

                for nextEdge in nextEdgesCollection:
                    if nextEdge[0] != edge[1]:
                        nextEdge.reverse()
                    coupleEdgesCollection.append([edge,nextEdge])

                # print('--- --- couple edges collection : ',coupleEdgesCollection)

                for edgesCouple in coupleEdgesCollection:

                    # print('--- --- --- edges couple : ',edgesCouple)

                    missingEdge = [edgesCouple[0][0],edgesCouple[1][1]]

                    # print('--- --- --- missing edges : ',missingEdge)

                    isPresent = False

                    if missingEdge not in edgesList:
                        missingEdge.reverse()
                        if missingEdge in edgesList:
                            isPresent = True
                    else:
                        isPresent = True

                    # print('--- --- --- isPresent : ',isPresent)

                    if isPresent:
                        newFace = list(set([edgesCouple[0][0],edgesCouple[0][1],edgesCouple[1][1]])) 
                        newFace.sort()
                        if newFace not in faces:
                            faces.append(newFace)
                            # print(faces)

                    isPresent = False
            # break
        return faces

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
        return np.acos((p0p1**2 + p0p2**2 - p1p2**2)/(2*p0p1*p0p2))

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
    def getMeshHierarchy(self,initialMesh:Mesh) -> List[Mesh]:

        numStep:int = initialMesh.stepNum

        # Liste des maillages obtenues après les simplifications successives
        meshHierarchy:List[Mesh] = [initialMesh.copy()]
        currentMesh:Mesh = initialMesh

        testVerticesList = [[3],[15,0]] #[[12,14],[9,3]]
        testVerticesIndex = 0

        # Pour chaque étape
        for l in range(numStep-1,-1,-1):

            # if l==3:
                # facesWith129 = currentMesh.getFacesWithVertex(129)
                # for face in facesWith129:
                    # print(face)
                # print('/////////////////')
                # facesWith220 = currentMesh.getFacesWithVertex(220)
                # for face in facesWith220:
                #     print(face)
                # print('222 : ',currentMesh.points[222])
                # print('225 : ',currentMesh.points[225])
                # print('227 : ',currentMesh.points[227])
                # print('220 : ',currentMesh.points[220])
                # print('133 : ',currentMesh.points[133])
                # print('131 : ',currentMesh.points[131])
                # print('129 : ',currentMesh.points[129])
                # print('71 : ',currentMesh.points[71])
                # print('/////////////////')
                # facesWith71 = currentMesh.getFacesWithVertex(71)
                # for face in facesWith71:
                #     print(face)
            
            # Liste des sommets marqués : un sommet est soit supprimable soit non-supprimable
            verticesToRemove = self.getVerticesToRemove(mesh=currentMesh,maxNeighborsNum=6)
            # verticesToRemove = [0]

            print('##############################################################')
            print('##############################################################')
            print('STEP ',l)

            print('VERTICES TO REMOVE')
            for vertex in verticesToRemove:
                print(vertex)

            i = 0

            # testVertices:List[int] = testVerticesList[testVerticesIndex]

            # Tant que le maillage courant contient des sommets "supprimables"
            while i < len(verticesToRemove):

                # print('##############################################################')
                print('VERTICES N',i)
                    
                # Selectionner aléatoirement un sommet v non-marqué dont le degré de sorti est inférieur à 12
                # while(True):
                #     selectedVertex = choice(marked_vertices['removable'])

                #     # Tester si le sommet sélectionné a au moins de 12 voisins
                #     if currentMesh.getNumberOfNeighbors(vertexId=selectedVertex) <= 4:
                #         break

                printer=False

                # if l==2 and i==152:
                #     printer=True
                
                vertexToRemove = verticesToRemove[i]

                # if i==152:
                #     print('vertex to remove ',vertexToRemove)
                #     print(vertexToRemove is not None)

                if i==62:
                    currentMesh.plot('i',currentMesh.points[verticesToRemove[62]])

                if i==63:
                    currentMesh.plot('i',currentMesh.points[129])

                if vertexToRemove is not None:

                    if len(currentMesh.getEdgesWithVertex(vertexToRemove)) > 0:

                        if printer:print('1')
                        holedRegion:HoledRegion = HoledRegion(vertexToRemove=vertexToRemove,mesh=currentMesh)
                        if printer:print('2')
                        # Avant de supprimer quoi que ce soit dans les listes de faces, d'arrêtes et de sommets du maillage courant, on calcule les 
                        # nouvelles arrêtes et faces qu'il faudra ajouter au maillage courant après la suppression du sommet sélectionné
                        newEdges,newFaces = holedRegion.getNewSimplices()

                        # if vertexToRemove == 222:
                            # print('VERTEX TO REMOVE = ',vertexToRemove)
                        if vertexToRemove==222:
                            # a_supprimer = [222,225,227,220,133,131,129,71]
                            print('NEW FACES :\n',newFaces)
                            # for face in newFaces:
                            #     print(a_supprimer[face[0]],' ',a_supprimer[face[1]],' ',a_supprimer[face[2]])
                            print('NEW EDGES :\n',newEdges)
                            # for edge in newEdges:
                            #     print(a_supprimer[edge[0]],' ',a_supprimer[edge[1]])

                        if printer:print('3')
                        # Supprimer v de la liste des sommets du maillage courant
                        # ## Supprimer v de la liste des points du maillage
                        vertexIndex = currentMesh.simplicies['vertices'].index(vertexToRemove)
                        currentMesh.points[vertexIndex] = np.array([-np.inf,-np.inf,-np.inf])
                        # ## Supprimer de la liste des sommets du maillage
                        currentMesh.simplicies['vertices'][vertexToRemove] = None

                        # Récupérer la liste des voisins de v
                        selectedVertexNeighbors = currentMesh.getNeighbors(vertexId=vertexToRemove)

                        # Marquer les voisins du sommet à supprimer comme non-supprimable
                        for neighbor in selectedVertexNeighbors:
                            try:
                                neighborIndex = verticesToRemove.index(neighbor)
                                verticesToRemove[neighborIndex] = None
                            except ValueError:
                                pass

                        # Marquer les sommets voisins comme "non-supprimable" et les supprimer de la liste des sommets "supprimables"
                        # for vertexId in selectedVertexNeighbors:
                        #     try:
                        #         marked_vertices['removable'].remove(vertexId)
                        #     except ValueError:
                        #         pass
                        #     if vertexId not in marked_vertices['unremovable']:
                        #         marked_vertices['unremovable'].append(vertexId)

                        # Supprimer les arrêtes auxquelles appartient v
                        edgesWithSelectedVertex = currentMesh.getEdgesWithVertex(vertexId=vertexToRemove)

                        for edge in edgesWithSelectedVertex:
                            currentMesh.simplicies['edges'].remove(edge)

                        # Supprimer les faces auxquelles appartient v
                        facesWithSelectedVertex = currentMesh.getFacesWithVertex(vertexId=vertexToRemove)
                        for face in facesWithSelectedVertex:
                            currentMesh.simplicies['faces'].remove(face)

                        # Ajouter les nouvelles arrêtes et faces au maillage courrant
                        for edge in newEdges:
                            currentMesh.simplicies['edges'].append(edge)

                        for face in newFaces:
                            currentMesh.simplicies['faces'].append(face)

                    else:
                        currentMesh.simplicies['vertices'][vertexToRemove] = None


                i = i + 1

                # Supprimer les arrêtes présentent deux fois, 
                currentMesh.removeDoubleEdges()

            # On ajoute le maillage obtenu à la hierarchie des maillages
            meshHierarchy.append(currentMesh.copy())

            testVerticesIndex = testVerticesIndex + 1
        
        return meshHierarchy
        

def main():
    """
    Runs the program on the model given as parameter.
    """
    np.seterr(invalid = 'raise')
    model = MapsModel(L=3)
    model.parse_file('example/test1.obj')

    finestMesh = model.model2Mesh()
    finestMesh.plot(' ')

    meshHierarchy:List[Mesh] = model.getMeshHierarchy(initialMesh=finestMesh)

    for mesh in meshHierarchy:
        mesh.plot(title='level '+str(mesh.stepNum))

if __name__ == '__main__':
    main()


# def constraintRetriangulation(self,mesh:Mesh,vertexToRemove:int) -> Tuple[List[List[int]],List[List[int]]]:

#         newEdges = []
#         newFaces = []

#         externalEdges = mesh.getExternalEdgesInCyclicOrder(vertexId=vertexToRemove)

#         # Calculer les coordonnees polaires dans le plan de projection du 1-ring du sommet à supprimer
#         polarCoordonates = self.getConformalMap(mesh=mesh,centralVertex=vertexToRemove)

#         # Convertir coordonnees polaires en coordonnees cartesiennes
#         cartesianCoordonates = self.convertPolar2CartesianCoordonates(polarCoordonates)

#         # Calculer les arrêtes interdites
#         hull = ConvexHull(points=cartesianCoordonates.T)

#         externalEdges = mesh.getExternalEdgesInCyclicOrder(vertexId=vertexToRemove)

#         # Si centralVertex est un sommet de la frontière
#         if externalEdges[0][0] != externalEdges[-1][1]:
#             # On ajoute une dernière arrête : 
#             # Exemple : si on a extEdges = [[1,2] ; [2,3] ; [3,4]], alors on obtient [[1,2] ; [2,3] ; [3,4] ; [4,1]]
#             externalEdges.append([externalEdges[-1][1],externalEdges[0][0]])

#         externalVertices = [edge[0] for edge in externalEdges]

#         print("external vertices : ",externalVertices)

#         forbiddenEdges:List[List[int]] = []

#         ############ PRINT ##############
#         print("Convexe Hull")
#         for edge in hull.simplices:
#             print('edge[0] = ',edge[0])
#             print('edge[1] = ',edge[1])
#             print([externalVertices[edge[0]],externalVertices[edge[1]]])
#         ############ PRINT ##############
        
#         for edge in hull.simplices:
#             edge = [externalVertices[edge[0]],externalVertices[edge[1]]]
#             if edge not in mesh.simplicies['edges']:
#                 edge.reverse()
#                 if edge not in mesh.simplicies['edges']:
#                     edge.reverse()
#                     forbiddenEdges.append(edge)
#         ###################

#         # Si le sommet à supprimer est un sommet de la frontière
#         if externalEdges[0][0] != externalEdges[-1][1]:
#             externalEdges.append([externalEdges[-1][1],externalEdges[0][0]])

#         externalVertices:List[int] = [edge[0] for edge in externalEdges]

#         while len(externalVertices) > 3:

#             nb_angles = len(externalVertices)

#             angles:np.ndarray = np.zeros((nb_angles,1))

#             # Calculer tout les angles aux sommets restant dans externalVertices
#             for i in range(nb_angles):
#                 p0 = externalVertices[i]
#                 p1 = externalVertices[i-1]
#                 p2 = externalVertices[(i+1)%nb_angles]

#                 angle = self.getFaceCentralAngle(mesh.points[p0],mesh.points[p1],mesh.points[p2])

#                 angles.append(angle)

#             argmin = np.argmin(angles)

#             newEdge = [externalVertices[argmin-1],externalVertices[(argmin+1)%nb_angles]]
#             newFace = [externalVertices[argmin-1],externalVertices[argmin],externalVertices[(argmin+1)%nb_angles]]


#             newEdges.append(newEdge)
#             newFaces.append(newFace)

#         return newEdges,newFaces
