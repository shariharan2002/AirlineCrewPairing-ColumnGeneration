import numpy as np
import pickle
def pickle_load_dnal(filename):
    # print("Loading --> "+str(filename))
    file = open(str(filename), 'rb')
    data= pickle.load(file)
    file.close()
    # print(str(filename)+' --> Loaded Successfully')
    return data
def write_pickle_dnal(data,filename):
    # print("Writing --> "+str(filename))
    file_path = str(filename)
    with open(file_path, 'wb') as file:
        # Serialize and write the variable to the file
        pickle.dump(data, file)
    # print(str(filename)+"--> Saved !!!")
    
class DNAL_creator:
    def __init__(self,filename_list):
        self.adj_list=self.DNAL_calculate_adj_list(filename_list)
    
    def DNAL_calculate_adj_list(self,filename_list):
        adj_list = []
        for name in filename_list:
            data=pickle_load_dnal(name)
            neighbors = np.where(data)[0]
            adj_list.append(neighbors.tolist())

        print("Adjacency List Calculation Completed")
        return adj_list
    
    