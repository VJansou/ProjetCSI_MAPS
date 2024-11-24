
import numpy as np

class HoledRegion:
    def __init__(self, vertex_to_remove, maps_model):
        from MapsModel import MapsModel
        self.vertex_to_remove = vertex_to_remove
        self.maps_model = maps_model
        self.star_external_edges = self.maps_model.get_star_external_edges(vertex_to_remove)
        self.star_vertices, self.is_boundary = self.maps_model.get_star_vertices_in_cyclic_order(vertex_to_remove)


    def get_conformal_map_points(self):
        """
        Calcule les coordonnées des points dans la conformal map (les μ du papier)
        """
        # On récupère les faces de l'étoile
        star_faces_temp = self.maps_model.get_star_faces(self.vertex_to_remove)
        # On ordonne les faces par rapport au cycle des sommets
        star_faces = []
        for vertex, next_vertex in zip(self.star_vertices, self.star_vertices[1:]+self.star_vertices[:1]):
            for face in star_faces_temp:
                if vertex in face and next_vertex in face:
                    star_faces.append(face)
                    break

        # On calcule les angles centraux des faces
        angles = np.zeros(len(star_faces))
        for idx, face in enumerate(star_faces):
            angles[idx] = self.maps_model.compute_angle_face(face, self.vertex_to_remove)
        
        # On calcule les angles cumulés (les θ du papier)
        theta_k = np.cumsum(angles)

        # Dans le cas où le sommet à supprimer est un sommet de bord, on defini le premier angle à 0
        if self.is_boundary:
            theta_k = np.insert(theta_k, 0, 0)

        # On calcule le facteur de normalisation (le a du papier)
        if self.is_boundary:
            a = np.pi/theta_k[-1]
        else:
            a = 2*np.pi/theta_k[-1]

        # On calcule les distances des points par rapport au sommet à supprimer (les r du papier)
        r_k = np.zeros(len(self.star_vertices))
        p_i = self.maps_model.vertices[self.vertex_to_remove]
        for idx, vertex in enumerate(self.star_vertices):
            p_jk = self.maps_model.vertices[vertex]
            r_k[idx] = np.linalg.norm(p_jk - p_i)
        
        # On calcule les coordonnées des points dans la conformal map
        mu_k = np.zeros(len(self.star_vertices),2)
        mu_k[:,0]  = r_k**a * np.cos(theta_k * a)
        mu_k[:,1]  = r_k**a * np.sin(theta_k * a)

        return mu_k


    def is_point_inside_star(self, point, points):
        """
        Vérifie si un point est à l'intérieur de l'étoile en utilisant l'algorithme de Ray Casting
        """
        x, y = point
        n = len(points)
        is_inside = False
        p1x, p1y = points[0]
        for i in range(n+1):
            p2x, p2y = points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or x <= xinters:
                            is_inside = not is_inside
            p1x, p1y = p2x, p2y
        return is_inside

    def compute_internal_angle(self, idx, points):
        """
        Calcule l'angle interne lié à un sommet de la star
        """
        nb_vertices = len(points)
        # On récupère les coordonnées des points
        p_i = points[idx] # vecteur courant
        p_j = points[(idx+1)%nb_vertices] # vecteur suivant
        p_k = points[(idx-1)%nb_vertices] # vecteur précédent

        # On calcule les vecteurs
        v_ij = p_j - p_i # vecteur suivant - courant
        v_ik = p_k - p_i # vecteur précédent - courant

        # On calcule les normes
        norm_v_ij = np.linalg.norm(v_ij)
        norm_v_ik = np.linalg.norm(v_ik)

        # On calcule le produit scalaire
        dot_product = np.dot(v_ij, v_ik)

        # On calcule l'angle en degrés
        angle = np.degrees(np.arccos(dot_product / (norm_v_ij * norm_v_ik)))

        # On calcule le baricentre des 3 points
        baricenter = (p_i + p_j + p_k) / 3

        # On vérifie si le baricentre est dans l'étoile
        is_inside = self.is_point_inside_star(baricenter, points)

        # Si le baricentre est à l'intérieur de l'étoile, on renvoie l'angle
        # Sinon, on renvoie 360 - l'angle
        return angle if is_inside else 360 - angle



    def compute_internal_angles(self, points):
        """
        Calcule les angles internes liés à chaque sommet de la star
        """
        internal_angles = np.zeros(len(points))
        for idx in range(len(points)):
            internal_angles[idx] = self.compute_internal_angle(idx, points)
        return internal_angles
         

    def compute_new_edges_and_faces(self):
        """
        Calcule les nouvelles arêtes et les nouvelles faces pour le trou
        """
        new_edges = []
        new_faces = []

        # Dans le cas où le sommet à supprimer est un sommet de bord,
        # on créer une arête entre les sommets de la star qui sont des sommets de bord
        if self.is_boundary:
            boundary_edge = tuple(sorted([self.star_vertices[0], self.star_vertices[-1]]))
            if boundary_edge not in self.star_external_edges:
                new_edges.append(boundary_edge)
        # On a maintenant dans tout les cas une étoile qui fait un cycle

        # On calcule les coordonnées de nos points dans la conformal map
        points = self.get_conformal_map_points()

        # On calcule les angles internes liés à chaque sommet de la star
        internal_angles = self.compute_internal_angles(points)

        # On boucle tant qu'il reste 3 sommet à traiter
        while len(points) > 3:
            # On récupère l'indice du sommet qui a l'angle interne le plus petit
            idx = np.argmin(internal_angles)

            # On récupère les indices des sommets précédent et suivant
            idx_prev = idx-1
            idx_next = (idx+1)%len(points)

            # On ajoute l'arête entre les sommets précédent et suivant
            new_edges.append(tuple(sorted([self.star_vertices[idx_prev], self.star_vertices[idx_next]])))

            # On ajoute la face formée par les sommets courant, précédent et suivant
            new_faces.append(tuple(sorted([self.star_vertices[idx], self.star_vertices[idx_prev], self.star_vertices[idx_next]])))

            # On supprime le sommet courant, son angle interne et ses coordonnées
            points = np.delete(points, idx, axis=0)
            internal_angles = np.delete(internal_angles, idx)
            self.star_vertices.pop(idx)

            # On met à jour les angles internes des sommets précédent et suivant
            internal_angles[idx_prev] = self.compute_internal_angle(idx_prev, points)
            internal_angles[idx%(len(points)-1)] = self.compute_internal_angle(idx%(len(points)-1), points)
        
        # On ajoute la dernière face
        new_faces.append(tuple(self.star_vertices))

        return new_edges, new_faces
