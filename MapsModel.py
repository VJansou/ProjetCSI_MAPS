import obja
import numpy as np
from typing import List,Dict
from HoledRegion import HoledRegion
import decimate

# Sous-classe de obja.Model afin de mieux gérer les faces
class MapsModel(obja.Model):

    def __init__(self, path, treshold_dihedral_angle=3*np.pi/4):
        super().__init__()
        super().parse_file(path)
        self.liste_faces = self.faces_to_list()
        self.edges = self.create_edges_list()
        self.neighbours = self.create_neighbours_dict()
        self.status_vertices = {i: 1 for i in range(len(self.vertices))} # 1 = sommet removable, 0 = sommet unremovable, -1 sommet déjà enlevé
        self.status_edges = {i: 1 for i in range(len(self.edges))} # 1 = normal edge, 0 = feature edge
        self.initialize_feature_edges(treshold_dihedral_angle)
        self.intitialize_status_vertices()
        self.L = 0 # nombre de niveaux
        self.liste_simplicies:Dict[List[int]] = [self.initialize_liste_simplicies()] # Le K du papier

    def faces_to_list(self):
        """
        Renvoie la liste des faces sous forme de liste de tuples de 3 sommets.
        """
        liste_faces = []
        for face in self.faces:
            #DEBUG on regarde si une face contient 0
            if 0 in [face.a, face.b, face.c]:
                print("face = ", face)
            liste_faces.append(tuple(sorted([face.a, face.b, face.c])))
            liste_faces = list(set(liste_faces))
        return liste_faces

    def create_edges_list(self):
        """
        Crée la liste des arêtes du modèle.
        """
        edges = set()
        for face in self.liste_faces:
            edges.add((face[0], face[1]))
            edges.add((face[1], face[2]))
            edges.add((face[0], face[2]))
        edges = list(edges)
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

    def initialize_feature_edges(self, treshold_dihedral_angle):
        """
        Initialise le dictionnaire des feature edges.
        """
        #DEBUG on va print tout les angles diedres dans un ableau, et aussi la moyenne
            # creation du tableau des angles diedres
        angles = []
        for index, edge in enumerate(self.edges):
            v0, v1 = edge
            angles.append(self.compute_angle_diedre(v0, v1))

            # On met les arêtes avec un angle diedre inférieur à treshold_angle comme feature edges
            # Remarque : on calcule la distance de l'angle diedre à pi pour éviter les problèmes de périodicité
            if np.abs(np.pi - self.compute_angle_diedre(v0, v1)) > treshold_dihedral_angle:
                self.status_edges[index] = 0 # Marque l'arête comme feature edge
        # print("angles = ", angles)
        # print("moyenne = ", np.mean(angles))
        # # Affiche un histogramme des angles diedres
        # import matplotlib.pyplot as plt
        # plt.hist(angles, bins=100)
        # plt.show()

    def intitialize_status_vertices(self, treshold_curvature=np.pi):
        """
        Initialise le dictionnaire des sommets removable.
        """
        for index, _ in enumerate(self.vertices):
            # # On met le premier et dernier sommet comme unremovable
            # if index == 0 or index == len(self.vertices) - 1:
            #     self.status_vertices[index] = 0
            # On met les sommets avec une courbure supérieure à treshold_curvature comme unremovable
            if self.compute_curvature_star(index) > treshold_curvature:
                self.status_vertices[index] = 0
            # # On met les sommets qui sont associées à au moins 2 feature edges comme unremovable
            # elif len([edge for edge in self.edges if index in edge and self.status_edges[self.edges.index(edge)] == 0]) >= 2:
            #     self.status_vertices[index] = 0
        print("status_vertices = ", self.status_vertices)

    def initialize_liste_simplicies(self):
        """
        Initialise les simplicies.
        """
        simplicies = {'vertices':[],'edges':[],'faces':[]}
        simplicies['vertices'] = [_ for _ in range(len(self.vertices))] # Contient les indices des sommets
        simplicies['edges'] = self.edges
        simplicies['faces'] = self.liste_faces
        print("vertices de simplicies = ", simplicies['vertices'])
        return simplicies
        

    def get_faces_from_edge(self, edge):
        """
        Renvoie les faces qui contiennent une arête.
        """
        faces = []
        compteur = 0 # Il ne peut y avoir que 2 faces qui contiennent une arête
        for face in self.liste_faces:
            if edge[0] in face and edge[1] in face and compteur < 2:
                faces.append(face)
                compteur += 1
        return faces

    def get_neighbours(self, vertex):
        """
        Renvoie les voisins d'un sommet.
        """
        return self.neighbours[vertex]
    
    def get_star_faces(self, central_vertex):
        """
        Renvoie la liste des faces de l'étoile d'un sommet.
        """
        star = []
        for face in self.liste_faces:
            if central_vertex in face:
                star.append(face)
        return star

    def get_star_external_edges(self, central_vertex):
        """
        Renvoie la liste des arêtes extérieures de l'étoile d'un sommet.
        """
        external_edges = []
        star = self.get_star_faces(central_vertex)
        #DEBUG if central_vertex == 70:
        #     print("startot = ", star)
        for face in star:
            external_edges.append(tuple(vertex for vertex in face if vertex != central_vertex))
        return external_edges
    
    def get_star_vertices_in_cyclic_order(self, central_vertex):
        """
        Renvoie les sommets de l'étoile d'un sommet dans l'ordre cyclique.
        Renvoie aussi si le sommet est sur le bord.
        """
        vertices_in_cyclic_order = []
        is_boundary = False
        # On récupère les arêtes extérieures de l'étoile
        star_external_edges = self.get_star_external_edges(central_vertex)

        # On crée un dictionnaire des voisins pour chaque sommet
        neighbours = {}
        for edge in star_external_edges:
            neighbours.setdefault(edge[0], []).append(edge[1])
            neighbours.setdefault(edge[1], []).append(edge[0])

        # On récupère un bord de l'étoile s'il existe : 
        # cela permet de partir d'un sommet sur le bord s'il y en a un et donc de s'assurer de parcourir toute l'étoile
        current_vertex = next((vertex for vertex, neighbour_list in neighbours.items() if len(neighbour_list) == 1), None)
        #DEBUG print("current_vertex = ", current_vertex)
        # if central_vertex == 70:
        #     print("star_external_edges = ", star_external_edges)
        #     print("neighbours = ", neighbours)
        if current_vertex is None: # S'il n'y a pas de bord
            # On prend un sommet quelconque de l'étoile
            current_vertex = star_external_edges[0][0]
        else: # S'il y a un bord
            is_boundary = True
        # On ajout le sommet à la liste
        vertices_in_cyclic_order.append(current_vertex)

        # On parcourt l'étoile
        while True:
            # On récupère les voisins du sommet actuel
            current_neighbours = neighbours[current_vertex]
            # On récupère le voisin suivant
            next_vertex = next((vertex for vertex in current_neighbours if vertex not in vertices_in_cyclic_order),None)
            # On sort de la boucle si on a fait le tour de l'étoile
            if next_vertex is None:
                break
            # On ajoute le voisin suivant à la liste
            vertices_in_cyclic_order.append(next_vertex)
            # On change de sommet
            current_vertex = next_vertex

        return vertices_in_cyclic_order, is_boundary
            

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
        star = self.get_star_faces(vertex)
        area = 0
        for face in star:
            area += self.compute_area_face(face)
        return area

    def compute_angle_face(self, face, central_vertex):
        """
        Calcule l'angle d'une face.
        """
        # On trouve l'indice du central_vertex dans la face
        index_central = face.index(central_vertex)

        # On associe les points du triangle
        a = self.vertices[face[index_central]]  # Sommet central
        b = self.vertices[face[(index_central + 1) % 3]]  # Point suivant
        c = self.vertices[face[(index_central + 2) % 3]]  # Point précédent
        ab = np.linalg.norm(b - a)
        bc = np.linalg.norm(c - b)
        ac = np.linalg.norm(a - c)
        return np.arccos((ab**2 + ac**2 - bc**2) / (2 * ab * ac))
    
    def compute_curvature_star(self, central_vertex):
        """
        Calcule la courbure associée à l'étoile d'un sommet.
        """
        star = self.get_star_faces(central_vertex)
        curvature = 0
        for face in star:
            curvature += self.compute_angle_face(face, central_vertex)
        return 2*np.pi - curvature
    
    def compute_angle_diedre(self, v0, v1):
        """
        Calcule l'angle diedre entre deux arêtes.
        """
        faces = self.get_faces_from_edge((v0, v1))
        if len(faces) != 2:
            return np.pi
        normal_0 = np.cross(self.vertices[faces[0][1]] - self.vertices[faces[0][0]], self.vertices[faces[0][2]] - self.vertices[faces[0][0]])
        normal_1 = np.cross(self.vertices[faces[1][1]] - self.vertices[faces[1][0]], self.vertices[faces[1][2]] - self.vertices[faces[1][0]])
        return np.arccos(np.clip(np.dot(normal_0, normal_1) / (np.linalg.norm(normal_0) * np.linalg.norm(normal_1)), -1, 1))
        

    def get_vertices_to_remove(self, max_neighbours, lambda_):
        """
        Renvoie les sommets à enlever (leurs indices) pour un niveau dans l'ordre de priorité.
        """
        # On récupère les sommets removable
        vertices_to_remove = []
        for vertex, status_vertex in self.status_vertices.items():
            if status_vertex == 1 and len(self.get_neighbours(vertex)) <= max_neighbours:
                vertices_to_remove.append(vertex)

        # S'il n'y a pas de sommets removable, on renvoie une liste vide
        if not vertices_to_remove:
            return vertices_to_remove

        # On calcul les aires et les courbures associées à chaque sommet
        areas = [self.compute_area_star(vertex) for vertex in vertices_to_remove]
        curvatures = [self.compute_curvature_star(vertex) for vertex in vertices_to_remove]

        # On calcule les poids associés à chaque sommet
        max_area = max(areas)
        max_curvature = max(curvatures)
        weights = [lambda_ * area / max_area + (1 - lambda_) * curvature / max_curvature for area, curvature in zip(areas, curvatures)]

        # On trie les sommets à enlever par ordre de priorité
        vertices_to_remove = [vertex for _, vertex in sorted(zip(weights, vertices_to_remove), reverse=True)]
        return vertices_to_remove

    def compute_mesh_hierarchy(self, max_neighbours=12, lambda_=0.5):
        """
        Calcule les différents mesh.
        Renvoie la liste des opérations.
        """
        # On intialise la liste des opérations
        operations = []
        # On boucle tant qu'il reste des sommets removable
        while 1 in self.status_vertices.values():
            
            # On augmente le niveau
            self.L += 1
            print("L = ", self.L)

            # On initialise les simplicies pour ce niveau comme les simplicies du niveau précédent
            simplicies = self.liste_simplicies[-1]

            # On initialise la liste des opérations pour ce niveau
            operations_l = []

            # On récupère les indices des sommets à enlever
            vertices_to_remove = self.get_vertices_to_remove(max_neighbours, lambda_)

            # On sort de la boucle si on ne peut plus enlever de sommets
            if not vertices_to_remove:
                break

            # On boucle tant que vertices_to_remove n'est pas vide
            while vertices_to_remove:
                # On récupère le sommet à enlever
                vertex_to_remove = vertices_to_remove.pop(0)

                # On change le statut du sommet à -1
                self.status_vertices[vertex_to_remove] = -1

                # Avant de supprimer les faces, arrêtes et le sommet du simplicies courant,
                # on calcule les nouvelles arrêtes et faces à ajouter
                holed_region = HoledRegion(vertex_to_remove, self)
                new_edges, new_faces = holed_region.compute_new_edges_and_faces()

                # On enlève le sommet des simplicies
                # simplicies['vertices'].remove(vertex_to_remove)
                # On enlève les arêtes associées à ce sommet dans les simplicies
                simplicies['edges'] = [edge for edge in simplicies['edges'] if vertex_to_remove not in edge]
                # On enlève les faces associées à ce sommet dans les simplicies
                simplicies['faces'] = [face for face in simplicies['faces'] if vertex_to_remove not in face]
                
                # On ajoute les nouvelles arêtes et les nouvelles faces aux simplicies
                simplicies['edges'] += new_edges
                simplicies['faces'] += new_faces

                # On ajoute les opérations pour les nouvelles faces
                for face in new_faces:
                    operations_l.append(('new_face', 0, obja.Face(face[0],face[1],face[2])))

                # On ajoute les opérations pour les faces à supprimer
                for face in self.get_star_faces(vertex_to_remove):
                    operations_l.append(('face', 0, obja.Face(face[0],face[1],face[2])))

                # On met à jour le dictionnaire des voisins
                for edge in new_edges:
                    for vertex in edge:
                        self.neighbours[vertex].append(edge[0] if edge[1] == vertex else edge[1])

                # On enlève les voisins de ce sommet de vertices_to_remove
                neighbours_vertex_to_remove = self.get_neighbours(vertex_to_remove)
                for neighbour in neighbours_vertex_to_remove:
                    if neighbour in vertices_to_remove:
                        vertices_to_remove.remove(neighbour)
            
            # On ajoute les simplicies pour ce niveau à la liste des simplicies
            self.liste_simplicies.append(simplicies)

            # On ajoute les opérations pour ce niveau à la liste des opérations
            operations.append(operations_l)
        
            #TODO On met à jour les feature edges et les sommets unremovable
        
        # On est au coarsest mesh : on ajoute les operations pour ce niveau à la liste des opérations
        operations_cm = []
        for face in self.liste_simplicies[-1]['faces']:
            operations_cm.append(('face', 0, obja.Face(face[0],face[1],face[2])))

        for vertex in self.liste_simplicies[-1]['vertices']:
            operations_cm.append(('vertex', vertex, self.vertices[vertex]))

        operations.append(operations_cm)

        # On met le sens des opérations à l'envers
        for operation in operations:
            operation.reverse()

        return operations


    def maps_to_model(self, simplicies):
        """
        Transforme les simplicies en modèle.
        """
        model = decimate.Decimater()
        model.vertices = [self.vertices[idx] for idx in simplicies['vertices']]
        model.faces = []
        i = 0
        for face in simplicies['faces']:
            model.faces.append(obja.Face(a=face[0], b=face[1], c=face[2]))

        model.line = len(model.vertices) + len(model.faces)
        return model
            