from Mesh import Mesh
from typing import List,Tuple
import matplotlib.pyplot as plt
import numpy as np

class HoledRegion:
    """
    Classe permettant de gérer une région trouée d'un maillage.
    Elle permet de calculer les nouvelles arêtes et faces après suppression d'un sommet.
    """

    def __init__(self,vertexToRemove:int,mesh:Mesh):
        """
        Initialise un objet HoledRegion.
        """
        self.vertexToRemove = vertexToRemove
        self.mesh = mesh
        self.isBoundary = False
        self.starEdges = self.mesh.get1RingExternalEdges(self.vertexToRemove).copy()
        res = self.mesh.getExternalVerticesInCyclicOrder(vertexId=self.vertexToRemove)
        self.starVertices = res[0].copy()
        self.isBoundary = res[1]
    

    def getFaceCentralAngle(self,p0:np.ndarray,p1:np.ndarray,p2:np.ndarray) -> float:
        """
        Étant donné trois points 3D p0, p1 et p2, retourne la valeur de l'angle (p0p1,p0p2)

        Args: p0:np.ndarray, les coordonnées du premier point, c'est en celui-ci qu'est calculé l'angle
              p1:np.ndarray, les coordonnées du second point
              p2:np.ndarray, les coordonnées du second point
              Chacun de ces array a pour dimension (3,)

        Return: float, la valeur de l'angle exprimé en radian, cette valeur est comprise entre 0 et pi
    """
        # Pour le calcul d'un angle d'une face, on utilise la loi d'Al-Kashi
        p0p1 = np.sqrt(np.sum((p0-p1)**2))
        p1p2 = np.sqrt(np.sum((p1-p2)**2))
        p0p2 = np.sqrt(np.sum((p0-p2)**2))
        return np.arccos((p0p1**2 + p0p2**2 - p1p2**2)/(2*p0p1*p0p2))
    
    def getFacesCentralAngles(self) -> np.ndarray:
        """
        Calcule les angles liés au sommet à retirer pour chaque face du polygone.
        Return: np.ndarray, un vecteur colonne contenant les angles internes de chaque face du polygone.
        """
        angles = np.zeros((len(self.starEdges),1))
        for k in range(len(self.starEdges)):
            p_j_lm1 = self.mesh.points[self.starEdges[k][0]]
            p_j_l = self.mesh.points[self.starEdges[k][1]]
            angles[k,0] = self.getFaceCentralAngle(p0=self.mesh.points[self.vertexToRemove],p1=p_j_l,p2=p_j_lm1)
        return angles
    
    
    def getConformalMap(self) -> np.ndarray:
        """
        Calcule les coordonnées cartésiennes des sommets de l'étoile en utilisant la carte conforme.

        Return: np.ndarray, la matrice des coordonnées cartésiennes, chaque colonne correspond aux coordonnées d'un point.
                La première ligne correspond aux abscisses et la deuxième aux ordonnées.
    """
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

        # Calculer les coordonnées cartésiennes
        cartesianCoordonates = np.zeros(polarCoordonates.shape)
        cartesianCoordonates[0,:] = polarCoordonates[0,:] * np.cos(polarCoordonates[1,:])
        cartesianCoordonates[1,:] = polarCoordonates[0,:] * np.sin(polarCoordonates[1,:])

        return cartesianCoordonates

    
    def isPointInPolygon(self, point, polygon):
        """
        Vérifie si un point est à l'intérieur d'un polygone donné en utilisant l'algorithme du raycasting.
        Return: bool, True si le point est à l'intérieur du polygone, False sinon.
        """
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
        """
        Calcule l'angle interne d'un polygone à un sommet donné.
        Return: float, la valeur de l'angle en degrés.
        """
        num_vertices = len(polygon)
            
        # On récupère les coordonnées des points
        p1 = np.array(polygon[i - 1]) # point précédent
        p2 = np.array(polygon[i]) # point courant
        p3 = np.array(polygon[(i + 1) % num_vertices]) # point suivant
        
        # On calcule les vecteurs
        vector_a = p1 - p2  # Vecteur du précédent au courant
        vector_b = p3 - p2  # Vecteur du suivant au courant

        # On calcule le produit scalaire et les normes des vecteurs
        dot_product = np.dot(vector_a, vector_b)
        magnitude_a = np.linalg.norm(vector_a)
        magnitude_b = np.linalg.norm(vector_b)

        # On calcule l'angle entre les deux vecteurs
        angle_radians = np.arccos(dot_product / (magnitude_a * magnitude_b))

        # On convertit l'angle en degrés
        angle_degrees = np.degrees(angle_radians)

        # On calcule le barycentre des trois points
        barycenter = (p1 + p2 + p3) / 3

        # On vérifie si l'angle est à l'intérieur du polygone
        if self.isPointInPolygon(barycenter, polygon):
            return angle_degrees  # l'angle est interne
        else:
            return 360 - angle_degrees  # l'angle est externe

    def calculateInternalAngles(self, polygon):
        """
        Calcule les angles internes d'un polygone.
        Return: np.ndarray, un vecteur contenant les angles internes de chaque sommet du polygone.
        """
        num_vertices = len(polygon)
        angles = np.zeros(num_vertices)

        for i in range(num_vertices):
            angles[i] = self.calculateInternalAngle(i, polygon)

        return angles


    def plotPolygonWithAngles(self, polygon, angles):
        """
        Affiche un polygone avec les angles internes annotés.
        """
        # On convertit le polygone en un tableau numpy pour faciliter l'affichage
        polygon = np.array(polygon)
        
        # On crée la figure et les axes
        fig, ax = plt.subplots()

        # on trace le polygone
        ax.plot(*np.append(polygon, [polygon[0]], axis=0).T, 'b-', label='Polygon')  # On relie le dernier point au premier pour fermer le polygone

        # On annote les angles pour chaque sommet
        for i, angle in enumerate(angles):
            # On récupère la position du sommet
            vertex = polygon[i]
            
            # On l'utilise pour annoter l'angle
            ax.text(vertex[0], vertex[1], f'{angle:.1f}°', fontsize=12, ha='center', va='bottom')

        ax.set_aspect('equal')
        ax.set_title('Polygon with Internal Angles')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.grid()
        ax.legend()

        plt.show()


    def getNewSimplices(self) -> Tuple[List[List[int]],List[List[int]]]:
        """
        Calcule les nouvelles arêtes et faces après suppression du sommet.
        Return: Tuple[List[List[int]],List[List[int]]], une paire de listes contenant les nouvelles arêtes et faces.
        """

        newEdges = []
        newFaces = []
        if len(self.starEdges) > 1 and len(self.starVertices) > 2:
            # Dna le cas où le sommet à retirer est sur le bord, on ajoute une arête entre les deux sommets du bord
            if self.isBoundary:
                boundaryEdge = tuple(sorted([self.starVertices[0], self.starVertices[-1]]))
                if boundaryEdge not in self.starEdges:
                    self.starEdges.append(boundaryEdge)
                    newEdges.append(boundaryEdge)
            # On a maintenant dans tout les cas une étoile qui fait un cycle

            # On calcule les coordonnées des sommets de l'étoile en utilisant la carte conforme
            points = self.getConformalMap().T.tolist()

            # On calcule les angles internes liés à chaque sommet de l'étoile
            internalAngles = self.calculateInternalAngles(points)
            
            # On boucle tant qu'il reste plus de 3 sommets
            while(len(points) > 3):
                # On cherche l'indice du sommet avec l'angle interne le plus petit
                indexToModify = np.argmin(internalAngles)

                nbVertices = len(self.starVertices)
                # On ajoute l'arête entre les sommets précédent et suivant
                newEdges.append(tuple(sorted([self.starVertices[indexToModify-1], self.starVertices[(indexToModify+1)%nbVertices]])))

                # On ajoute la face formé par les sommet précédent, courant et suivant
                newFaces.append(tuple(sorted([self.starVertices[indexToModify-1], self.starVertices[(indexToModify%nbVertices)], self.starVertices[(indexToModify+1)%nbVertices]])))

                # On supprime le sommet courant, son angle interne et ses coordonnées
                del points[indexToModify]
                internalAngles = np.delete(internalAngles, indexToModify)
                del self.starVertices[indexToModify]

                # On met à jour les angles internes des sommets précédent et suivant
                nbPointsRestant = len(points)
                internalAngles[indexToModify%nbPointsRestant] = self.calculateInternalAngle(indexToModify%nbPointsRestant, points)
                internalAngles[indexToModify-1] = self.calculateInternalAngle(indexToModify-1, points)

            # On ajoute la dernière face
            newFaces.append(tuple(sorted([self.starVertices[0], self.starVertices[1], self.starVertices[2]])))

        return newEdges, newFaces
                