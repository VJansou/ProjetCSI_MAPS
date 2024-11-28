import numpy as np
import MapsModel

from Mesh import Mesh
from typing import List
import decimate
import obja
import pandas as pd

def main():
    """
    Runs the program on the model given as parameter.
    """
    np.seterr(invalid = 'raise')
    L = -1
    model = MapsModel.MapsModel('./example/suzanne.obj')
    #model.parse_file('./example/suzanne.obj')

    finestMesh = model.model2Mesh()

    # for face in sorted(finestMesh.simplicies['faces']):
    #     print("initial face ",face)

    # print(74," neighbors : ",finestMesh.neighbors[74])
    # for neighbor in finestMesh.neighbors[74]:
    #     print("           ",neighbor," neighbors : ",finestMesh.neighbors[neighbor])

    # for i in range(len(finestMesh.simplicies['vertices'])):
    #     if 75 in finestMesh.neighbors[i] and 74 in finestMesh.neighbors[i] and 302 in finestMesh.neighbors[i]:
    #         print("suspicious vertex ",i)

    #model.getRetriangulation(mesh=finestMesh,vertexToRemove=0)

    meshHierarchy,operations, L, operations_old = model.getMeshHierarchy(initialMesh=finestMesh,maxNeighborsNum=12)
    print("L = ", L)
    operations.reverse()
    operations_old.reverse()
    
    back2model = meshHierarchy[-1].mesh2model()
    # model = decimate.Decimater()
    # model.parse_file('example/suzanne.obj')
    # print("Avant compression : ",len(finestMesh.simplicies['vertices']))

    # model2 = decimate.Decimater()
    # model2.parse_file('example/suzanne.obj')

    with open('example/suzanne_compresse36.obja', 'w+') as output2:
        back2model.contract(output2)

    file_weight_old = []
    for compression_level in range(0,L+1):
        filename = 'example/suzanne_decompresse_old_'+str(compression_level)+'.obja'
        with open(filename, 'w+') as output:
            # back2model.contract(output)
            # Write the result in output file
            output_model = obja.Output(output, random_color=True)

            # for (i,operation) in enumerate(operations[0]):
            #     print("operation ",i," : ",operation)

            faces = []
            level_weight_old = 0 # octets

            for (l,operations_l_old) in enumerate(operations_old):
                #print("l = ", l)
                if l > compression_level:
                    break

                for (ty, index, value) in operations_l_old:
                    if ty == "vertex":
                        output_model.add_vertex(index, value)
                        level_weight_old += 13 # 1 + 4 + 4 + 4
                    elif ty == "face":
                        if value not in faces:
                            faces.append((value.a, value.b, value.c))
                            output_model.add_face(len(faces)-1, value)
                            level_weight_old += 4 # 1 + 1 + 1 + 1
                    elif ty == "new_face":
                        index = faces.index((value.a, value.b, value.c))
                        if index != -1:
                            output_model.delete_face(index)
                            level_weight_old -= 4 # 1 + 1 + 1 + 1
                    #else:
                    #    output_model.edit_vertex(index, value)
        file_weight_old.append(level_weight_old)

    file_weight = []
    for compression_level in range(0,L+1):
        filename = 'example/suzanne_decompresse_'+str(compression_level)+'.obja'
        with open(filename, 'w+') as output:
            # back2model.contract(output)
            # Write the result in output file
            output_model = obja.Output(output, random_color=True)

            # for (i,operation) in enumerate(operations[0]):
            #     print("operation ",i," : ",operation)

            faces = []
            level_weight = 0 # octets

            for (l,operations_l) in enumerate(operations):
                #print("l = ", l)
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
                    #else:
                    #    output_model.edit_vertex(index, value)
        file_weight.append(level_weight)



    # Convert to a DataFrame


    data = []
    data.append(["Ancien OBJA"] + file_weight_old)
    data.append(["Nouvel OBJA"] + file_weight)

    df = pd.DataFrame(data, index=["", ""], columns=["Poids en octets selon la méthode et le niveau de compression"]+["Niveau "+str(i) for i in range(0,L+1)])

    print(df)
        

if __name__ == '__main__':
    main()