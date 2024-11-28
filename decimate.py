#!/usr/bin/env python

class Decimater():

    def __init__(self):
        super().__init__()

    def contract(self, output_model, operations, compression_level):
        faces = []
        level_weight = 0 # octets

        for (l,operations_l) in enumerate(operations):
            if l > compression_level:
                break

            for (ty, index, value) in operations_l:
                if ty == "vertex":
                    output_model.add_vertex(index, value)
                    level_weight += 13 # 1 + 4 + 4 + 4
                elif ty == "face":
                    if value not in faces:
                        faces.append((value.a, value.b, value.c))
                        output_model.add_face(len(faces)-1, value)   
                        level_weight += 4 # 1 + 1 + 1 + 1
                elif ty == "new_face":
                    index = faces.index((value.a, value.b, value.c))
                    if index != -1:
                        output_model.delete_face(index)
                        level_weight -= 4 # 1 + 1 + 1 + 1
        return level_weight
        
