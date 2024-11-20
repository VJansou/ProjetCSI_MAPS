import numpy as np
import MapsModel

from Mesh import Mesh
from typing import List
import decimate
import obja

def main():
    """
    Runs the program on the model given as parameter.
    """
    np.seterr(invalid = 'raise')
    model = MapsModel.MapsModel(L=4)
    model.parse_file('./example/suzanne.obj')

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

    meshHierarchy,operations = model.getMeshHierarchy(initialMesh=finestMesh,maxNeighborsNum=12)

    operations.reverse()
    
    back2model = meshHierarchy[-1].mesh2model()
    # model = decimate.Decimater()
    # model.parse_file('example/suzanne.obj')
    # print("Avant compression : ",len(finestMesh.simplicies['vertices']))

    # model2 = decimate.Decimater()
    # model2.parse_file('example/suzanne.obj')

    with open('example/suzanne_compresse.obja', 'w+') as output2:
        back2model.contract(output2)

    with open('example/suzanne_decompresse.obja', 'w+') as output:
        # back2model.contract(output)
        # Write the result in output file
        output_model = obja.Output(output, random_color=True)

        # for (i,operation) in enumerate(operations[0]):
        #     print("operation ",i," : ",operation)

        for (l,operations_l) in enumerate(operations):
            print("l = ", l)
            if l > -1:
                break

            for (ty, index, value) in operations_l:
                if ty == "vertex":
                    output_model.add_vertex(index, value)
                elif ty == "face":
                    output_model.add_face(index, value)   
                elif ty == "new_face" and l <= 1:
                    print(operations_l)
                    output_model.delete_face(index)
                #else:
                #    output_model.edit_vertex(index, value)
        

if __name__ == '__main__':
    main()