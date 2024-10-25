import MapsModel
from obja import *




model = MapsModel.MapsModel(L=4)
model.parse_file('./example/test1.obj')
print(model.vertices)
print('\n')
print(model.faces)
print('\n')
print(model.facesToList())
print('\n')
print(model.createEdgesList())
print('\n')
print(model.createNeighborsDict())
print('\n')
print(model.getEdgesWithVertex(3))
print('\n')
print(model.getFacesWithVertex(3))
print('\n')
print(model.get1RingExternalEdges(3))


