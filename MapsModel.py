import obja
import numpy as np
from typing import List,Tuple
from numpy import inf
from Mesh import Mesh
from HoledRegion import HoledRegion

#tests purpose
import sys

class MapsModel(obja.Model):
    """
    Sous-classe de obja.Model, permettant de manipuler un modèle 3D à partir d'un fichier .obj.
    """
    
    def __init__(self,path):
        """
        intialise un objet MapsModel.
        """
        super().__init__()
        self.L = 0
        super().parse_file(path)
        self.facesToList()
        self.createEdgesList()
        self.createNeighborsDict()

    
    def model2Mesh(self) -> Mesh:
        """
        Retourne le maillage associé à un modèle déduit d'un fichier .obj
        Return: Mesh
    """
        finestMesh = Mesh()

        # Parcours des sommets du modèle, on récupère l'indice d'un sommet dans self.vertices et son indice
        for index,vertex in enumerate(self.vertices):
            coordonates = vertex
            finestMesh.points.append(coordonates)
            finestMesh.simplicies['vertices'].append(index)

        # Parcours des faces du modèle
        finestMesh.simplicies['faces'] = list(set(sorted(self.list_faces)))

        # On ajoute les arrêtes du modèle
        finestMesh.simplicies['edges'] = self.edges

        # On ajoute les voisins de chaque sommet du maillage
        finestMesh.neighbors = self.neighbors

        return finestMesh
    
    def facesToList(self):
        """
        Crée la liste qui contient les faces sous forme de liste de tuples.
        Les faces sont triées par ordre croissant de leurs sommets.
        """
        listFaces = []
        for face in self.faces:
            listFace = sorted([face.a, face.b, face.c])
            listFaces.append(tuple(listFace))
        self.list_faces = list(set(sorted(listFaces)))
    
    def createEdgesList(self):
        """
        Crée la liste qui contient les arrêtes du modèle.
        """
        self.edges = set()
        for face in self.list_faces:
            self.edges.add((face[0], face[1]))
            self.edges.add((face[0], face[2]))
            self.edges.add((face[1], face[2]))
        self.edges = list(self.edges)
        return self.edges
    
    def createNeighborsDict(self):
        """
        Crée un dictionnaire qui contient les voisins de chaque sommet du modèle.
        """
        self.neighbors = {}
        for index,_ in enumerate(self.vertices):
            self.neighbors[index] = []
        for edge in self.edges:
            self.neighbors[edge[0]].append(edge[1])
            self.neighbors[edge[1]].append(edge[0])
        return self.neighbors
    
    
    def computeArea(self,p0:np.ndarray,p1:np.ndarray,p2:np.ndarray) -> np.ndarray:
        """
        Calcule l'aire d'un triangle à partir de ses trois sommets.
        """
        base = p1 - p0
        milieu_segment = (p0 + p1)/2
        hauteur = p2 - milieu_segment
        return np.linalg.norm(np.cross(base,hauteur)) / 2
    
    def computeCurvature(self,p0:np.ndarray,p1:np.ndarray,p2:np.ndarray) -> float:
        """
        Calcule l'angle d'une face à partir de ses trois sommets.
        """
        # Pour le calcul d'un angle d'une face, on utilise la loi d'Al-Kashi
        p0p1 = np.sqrt(np.sum((p0-p1)**2))
        p1p2 = np.sqrt(np.sum((p1-p2)**2))
        p0p2 = np.sqrt(np.sum((p0-p2)**2))
        return np.arccos((p0p1**2 + p0p2**2 - p1p2**2)/(2*p0p1*p0p2))

    def getAreasAndCurvatures(self,mesh:Mesh,selectedVertices:List[int]) -> Tuple[np.ndarray,float,np.ndarray,float]:
        """
        Calcule les aires et les courbures des étoiles des sommets sélectionnés.
        """

        nbVertices = len(selectedVertices)

        areas = np.zeros((nbVertices,1))
        curvatures = np.zeros((nbVertices,1))

        maxArea = -inf
        maxCurvature = -inf
        # Pour chaque sommet sélectionné :
        for indx,vertex in enumerate(selectedVertices):
            # On récupère les faces qui contiennent le sommet
            facesWithVertex = mesh.getFacesWithVertex(vertexId=vertex)

            # On calcule l'aire et la courbure de l'étoile du sommet
            starArea = 0
            starCurvature = 0
            for face in facesWithVertex:
                starArea = starArea + self.computeArea(p0=mesh.points[face[0]],p1=mesh.points[face[1]],p2=mesh.points[face[2]])
                face_copy = list(face)
                face_copy.remove(vertex)
                starCurvature = starCurvature + self.computeCurvature(p0=mesh.points[vertex],p1=mesh.points[face_copy[0]],p2=mesh.points[face_copy[1]])
            areas[indx,0] = starArea
            starCurvature = 2*np.pi - starCurvature
            curvatures[indx,0] = starCurvature

            # On met à jour les aires et courbures maximales si nécessaire
            if starArea > maxArea:
                maxArea = starArea
            if starCurvature > maxCurvature:
                maxCurvature = starCurvature

        return areas,maxArea,curvatures,maxCurvature
    
    def get_faces_from_edge(self, mesh:Mesh, edge):
        """
        Renvoie les faces qui contiennent une arête.
        """
        faces = []
        compteur = 0 # Il ne peut y avoir que 2 faces qui contiennent une arête
        for face in mesh.simplicies['faces']:
            if edge[0] in face and edge[1] in face:
                faces.append(face)
                compteur += 1
            if compteur == 2:
                break
        return faces
    
    def compute_angle_diedre(self, mesh:Mesh, v0, v1):
        """
        Calcule l'angle diedre entre deux sommets.
        """
        faces = self.get_faces_from_edge(mesh, (v0, v1))
        if len(faces) != 2:
            return np.pi
        normal_0 = np.cross(mesh.points[faces[0][1]] - mesh.points[faces[0][0]], mesh.points[faces[0][2]] - mesh.points[faces[0][0]])
        normal_1 = np.cross(mesh.points[faces[1][1]] - mesh.points[faces[1][0]], mesh.points[faces[1][2]] - mesh.points[faces[1][0]])
        return np.arccos(np.clip(np.dot(normal_0, normal_1) / (np.linalg.norm(normal_0) * np.linalg.norm(normal_1)), -1, 1))
        

    def getVerticesToRemove(self, status_vertices, mesh:Mesh,maxNeighborsNum:int=12,_lambda:float= 1/2,threshold_curv=np.pi, threshold_dihedral_angle=np.pi/24) -> List[int]:
        """
        Renvoie la liste des sommets à supprimer pour un niveau de compression dans l'ordre de priorité.
        """
        selectedVertices = []
        supressionOrder = []

        # On calcule les aires et les courbures des étoiles des sommets sélectionnés
        for idx, vertex in enumerate(mesh.simplicies['vertices']):
            if status_vertices[idx] == 1:
                nbNeighbors:int = mesh.getNumberOfNeighbors(vertex)
                if nbNeighbors <= maxNeighborsNum and nbNeighbors > 0:
                    selectedVertices.append(vertex)
        areas,maxArea,curvatures,maxCurvature = self.getAreasAndCurvatures(mesh=mesh,selectedVertices=selectedVertices)
        
        # Pour chaque sommet sélectionné :
        for indx,vertex in enumerate(selectedVertices):
            # On récupère les sommets voisins du sommet
            star_vertices, _ = mesh.getExternalVerticesInCyclicOrder(vertex)
            # On calcule le nombre d'arrêtes caractéristiques
            nb_feature_edges =  sum(1 for neigh_vertex in star_vertices if np.abs(self.compute_angle_diedre(mesh, neigh_vertex, vertex)) < threshold_dihedral_angle)

            # Si le sommet a plus de 2 arrêtes caractéristiques ou si sa courbure est supérieure à un seuil, il n'est pas supprimable
            if nb_feature_edges >= 2 or curvatures[indx,0] >= threshold_curv:
                ind_abs = mesh.simplicies['vertices'].index(vertex) if vertex in mesh.simplicies['vertices'] else -1
                if ind_abs != -1:
                    status_vertices[ind_abs] = 0
            # Sinon, on calcule le poids du sommet
            else:
                weight = _lambda * areas[indx,0]/maxArea + (1-_lambda) * curvatures[indx,0]/maxCurvature
                supressionOrder.append([vertex,weight])

        # On trie les sommets à supprimer par ordre de priorité
        supressionOrder.sort(key=lambda x: x[1])
        verticesToRemove = [e[0] for e in supressionOrder]

        return verticesToRemove
    

    def getMeshHierarchy(self,initialMesh:Mesh,maxNeighborsNum = 12):
        """
        Étant donné un maillage de départ, retourne la liste des maillages jusqu'au "Domain Base",
        la liste des opérations nécessaires pour retrouver chaque maillage de la liste et le nombre d'étapes.
        """

        # Liste des maillages obtenues après les simplifications successives
        meshHierarchy:List[Mesh] = [initialMesh.copy()]

        currentMesh:Mesh = initialMesh

        status_vertices = ([1] * (len(currentMesh.simplicies['vertices'])))

        vertices_removed = []

        operations = []
        operations_old = []

        compteur = 0

        # Pour chaque étape
        while 1 in status_vertices:

            currentMesh.currentStep = compteur
            verticesToRemove = self.getVerticesToRemove(status_vertices, mesh=currentMesh,maxNeighborsNum = maxNeighborsNum)
            if verticesToRemove == []:
                break
            i = 0

            operations_l = []
            operations_l_old = []

            # Tant que le maillage courant contient des sommets "supprimables"
            while len(verticesToRemove) != 0:
                
                printer=True


                vertexToRemove = verticesToRemove[0]
                status_vertices[vertexToRemove] = 0

                holedRegion:HoledRegion = HoledRegion(vertexToRemove=vertexToRemove,mesh=currentMesh)


                # Avant de supprimer quoi que ce soit dans les listes de faces, d'arrêtes et de sommets du maillage courant, on calcule les
                # nouvelles arrêtes et faces qu'il faudra ajouter au maillage courant après la suppression du sommet sélectionné
                newEdges,newFaces = holedRegion.getNewSimplices()

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
                        operations_l.append(('new_face',0, obja.Face(face[0],face[1],face[2])))
                        operations_l_old.append(('new_face',0, obja.Face(face[0],face[1],face[2])))
                        currentMesh.simplicies['faces'].append(face)

                edgesWithSelectedVertex = currentMesh.getEdgesWithVertex(vertexId=vertexToRemove)

                for edge in edgesWithSelectedVertex:
                    currentMesh.simplicies['edges'].remove(edge)

                # Supprimer les faces auxquelles appartient v
                facesWithSelectedVertex = currentMesh.getFacesWithVertex(vertexId=vertexToRemove)

                for face in facesWithSelectedVertex:
                    if face in currentMesh.simplicies['faces']: 

                        # On ajoute la face supprimée à la liste des opérations du l-ième maillage
                        operations_l.append(('face',0, obja.Face(face[0],face[1],face[2])))
                        operations_l_old.append(('face',0, obja.Face(face[0],face[1],face[2])))
                        
                        currentMesh.simplicies['faces'].remove(face)

                currentMesh.neighbors[vertexToRemove] = []
                for vertex in selectedVertexNeighbors:
                    if vertex != vertexToRemove:
                        currentMesh.neighbors[vertex].remove(vertexToRemove)

                
                operations_l.append(('vertex',vertexToRemove, currentMesh.points[vertexToRemove]))
                vertices_removed.append(vertexToRemove)

            # On ajoute le maillage obtenu à la hierarchie des maillages
            meshHierarchy.append(currentMesh.copy())

            # La ligne suivante ne devrait pas être utile, mais elle l'est pourtant...
            currentMesh.simplicies['faces'] = list(set(currentMesh.simplicies['faces']))

            # On ajoute la liste des opérations du l-ième malliage à la liste globale
            operations.append(operations_l)
            operations_old.append(operations_l_old)

            compteur += 1

        # On ajoute dans la liste des opérations globale, la liste des opérations nécessaires pour
        # retrouver le maillage de base : "Base Domain" 
        operationBaseDomain = []
        operationBaseDomain_old = []
        
        baseDomain = currentMesh

        for face in baseDomain.simplicies['faces']:
            operationBaseDomain.append(('face',0, obja.Face(face[0],face[1],face[2])))
            operationBaseDomain_old.append(('face',0, obja.Face(face[0],face[1],face[2])))

        for vertex in baseDomain.simplicies['vertices']:
            if vertex not in vertices_removed:
                
                operationBaseDomain.append(('vertex', vertex, baseDomain.points[vertex]))
            
            operationBaseDomain_old.append(('vertex', vertex, baseDomain.points[vertex]))

        operations.append(operationBaseDomain)
        operations_old.append(operationBaseDomain_old)

        for operation in operations:
            operation.reverse()
        
        for operation_old in operations_old:
            operation_old.reverse()

        return meshHierarchy,operations, compteur, operations_old
    

    
