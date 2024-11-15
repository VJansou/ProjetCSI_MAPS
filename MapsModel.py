import obja
import numpy as np

# Sous-classe de obja.Model afin de mieux gérer les faces
class MapsModel(obja.Model):

    def __init__(self, path):
        super().__init__()
        self.parse_file(path)
        self.liste_faces = self.faces_to_list()
        self.edges = self.create_edges_list()
        self.neighbours = self.create_neighbours_dict()
        self.status_vertices = {i: 1 for i in range(len(self.vertices))} # 1 = sommet removable, 0 = sommet unremovable
        self.status_edges = {i: 1 for i in range(len(self.edges))} # 1 = normal edge, 0 = feature edge
    
    def faces_to_list(self):
        """
        Renvoie la liste des faces sous forme de liste de tuples de 3 sommets.
        """
        liste_faces = []
        for face in self.faces:
            liste_faces.append(tuple(sorted([face.a, face.b, face.c])))
        return liste_faces

    def create_edges_list(self):
        """
        Crée la liste des arêtes du modèle.
        """
        edges = set()
        for face in self.liste_faces:
            self.edges.add((face[0], face[1]))
            self.edges.add((face[1], face[2]))
            self.edges.add((face[0], face[2]))
        return edges
    
    def create_neighbours_dict(self):
        """
        Crée un dictionnaire des voisins pour chaque sommet.
        """
        neighbours = {}
        for index,_ in enumerate(self.vertices):
            neighbours[index] = []
        for edge in self.edges:
            neighbours[edge[0]].append(edge[1])
            neighbours[edge[1]].append(edge[0])
        return neighbours

    def get_feature_edges(self):
        """
        Renvoie la liste des feature edges.
        """
        feature_edges = []
        for index, edge in enumerate(self.edges):
            if edge
    
    def get_neighbours(self, vertex):
        """
        Renvoie les voisins d'un sommet.
        """
        return self.neighbours[vertex]
    
    def get_star(self, vertex):
        """
        Renvoie l'étoile d'un sommet.
        """
        star = []
        for face in self.liste_faces:
            if vertex in face:
                star.append(face)
        return star

    def compute_area_face(self, face):
        """
        Calcule l'aire d'une face.
        """
        a = self.vertices[face[0]]
        b = self.vertices[face[1]]
        c = self.vertices[face[2]]
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a))
    
    def compute_area_star(self, vertex):
        """
        Calcule l'aire associée à l'étoile d'un sommet.
        """
        star = self.get_star(vertex)
        area = 0
        for face in star:
            area += self.compute_area_face(face)
        return area

    def compute_curvature_face(self, face):
        """
        Calcule la courbure d'une face.
        """
        a = self.vertices[face[0]]
        b = self.vertices[face[1]]
        c = self.vertices[face[2]]
        ab = np.linalg.norm(b - a)
        bc = np.linalg.norm(c - b)
        ac = np.linalg.norm(a - c)
        return np.arccos((ab**2 + ac**2 - bc**2) / (2 * ab * ac))
    
    def compute_curvature_star(self, vertex):
        """
        Calcule la courbure associée à l'étoile d'un sommet.
        """
        star = self.get_star(vertex)
        curvature = 0
        for face in star:
            curvature += self.compute_curvature_face(face)
        return curvature
    
