import numpy as np
from MapsModel import MapsModel
import decimate
import obja

def main():
    """
    Runs the program on the model given as parameter.
    """
    np.seterr(invalid = 'raise')
    maps_model = MapsModel('./example/suzanne.obj')

    # On calcule l'objet compressé
    operations = maps_model.compute_mesh_hierarchy()

    # On récupère le MapsModel compressé
    mesh_hierarchy = maps_model.liste_simplicies
    compressed_model = mesh_hierarchy[-1]

    # On le transforme en model
    model = maps_model.maps_to_model(compressed_model)

    with open('example/suzanne_compresse.obja', 'w+') as output2:
        model.contract(output2)

    for compression_level in range(0, maps_model.L + 1):
        filename = 'example/suzanne_decompresse_'+str(compression_level)+'.obja'
        with open(filename, 'w+') as output:
            # Write the result in output file
            output_model = obja.Output(output, random_color=True)

            faces = []

            for (l,operations_l) in enumerate(operations):
                #print("l = ", l)
                if l > compression_level:
                    break

                for (ty, index, value) in operations_l:
                    if ty == "vertex":
                        output_model.add_vertex(index, value)
                    elif ty == "face":
                        if value not in faces:
                            faces.append((value.a, value.b, value.c))
                            output_model.add_face(len(faces)-1, value)   
                    elif ty == "new_face":
                        index = faces.index((value.a, value.b, value.c))
                        if index != -1:
                            output_model.delete_face(index)

if __name__ == '__main__':
    main()