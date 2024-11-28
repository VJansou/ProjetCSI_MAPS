import numpy as np
import MapsModel
import obja
import pandas as pd
from decimate import Decimater

def main():
    """
    Runs the program on the model given as parameter.
    """
    np.seterr(invalid = 'raise')
    L = -1
    model = MapsModel.MapsModel('./example/suzanne.obj')

    finestMesh = model.model2Mesh()

    _, operations, L, operations_old = model.getMeshHierarchy(initialMesh=finestMesh,maxNeighborsNum=12)
    print("L = ", L)
    operations.reverse()
    operations_old.reverse()

    decimater = Decimater()

    file_weight_old = []

    file_weight = []
    for compression_level in range(0,L+1):
        
        filename = 'example/suzanne_decompresse_old_'+str(compression_level)+'.obja'
        with open(filename, 'w+') as output:
            # Write the result in output file
            output_model = obja.Output(output, random_color=True)
            level_weight_old = decimater.contract(output_model, operations_old, compression_level)
        file_weight_old.append(level_weight_old)

        filename = 'example/suzanne_decompresse_'+str(compression_level)+'.obja'
        with open(filename, 'w+') as output:
            # Write the result in output file
            output_model = obja.Output(output, random_color=True)
            level_weight = decimater.contract(output_model, operations, compression_level)
        file_weight.append(level_weight)



    # Display the weights to compare our old and new version of writing the obja files
    data = []
    data.append(["Ancien OBJA"] + file_weight_old)
    data.append(["Nouvel OBJA"] + file_weight)

    df = pd.DataFrame(data, index=["", ""], columns=["Poids en octets selon la méthode et le niveau de compression"]+["Niveau "+str(i) for i in range(0,L+1)])

    print(df)
        

if __name__ == '__main__':
    main()