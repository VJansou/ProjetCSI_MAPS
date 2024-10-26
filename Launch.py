import numpy as np
import MapsModel

from Mesh import Mesh
from typing import List
import decimate

def main():
    """
    Runs the program on the model given as parameter.
    """
    np.seterr(invalid = 'raise')
    model = MapsModel.MapsModel(L=4)
    model.parse_file('./example/suzanne.obj')

    finestMesh = model.model2Mesh()

    # neighbors = finestMesh.getNeighborsInCyclicOrder(vertexId=3)
    # print('neighbors = ',neighbors)

    # print('POINTS')
    # print(finestMesh.points)
    # print('EDGES')
    # print(finestMesh.simplicies['edges'])
    # print('FACES')
    # print(finestMesh.simplicies['faces'])
    # print('VERTICES')
    # print(finestMesh.simplicies['vertices'])

    # model.getRetriangulation(mesh=finestMesh,vertexToRemove=0)

    meshHierarchy:List[Mesh] = model.getMeshHierarchy(initialMesh=finestMesh,maxNeighborsNum= 54)


    for mesh in meshHierarchy:
        mesh.plot(title='level '+str(mesh.stepNum))
    
    back2model = meshHierarchy[-1].mesh2Model()

    model = decimate.Decimater()
    model.parse_file('example/suzanne.obj')

    with open('example/suzanneeeeeez.obja', 'w+') as output:
        model.contract(output)
    # print(finestMesh.simplicies['faces'][:5])
    # print(model.faces[:5])

if __name__ == '__main__':
    main()