import numpy as np; import torch

class network_data_struct():
    def __init__(self, field, network_num, types, data_dim):                            # dependent variable, network type, samples, data_dim
        setattr(self, field, np.empty(network_num, data_dim), dtype = object)
        self.num = network_num 
        self.types = types
        self.name = field 
        self.dim = data_dim
        self.samples = np.zeros(network_num) 
              
    def push(self, network_num, new_data):
        ind = self.samples[network_num]
        self.field[network_num][ind, :] = new_data
        self.start_ind[network_num] += 1 

    def add_digit(self, data, new_data):
        data[self.start_ind : self.start_ind + self.batch_len] = new_data
        
    def add_array(self, data, new_data):
        for i in range(self.batch_len):
             data[self.start_ind + i] = new_data[i]
        
    def increment(self):
        self.start_ind += self.batch_len
        
    def to_dictionary(self, d = dict()):
        for f in self.fields:
            d[str(f)] = getattr(self, str(f))
        return d 
    
    def from_dictionary(self, d):
        for f in d.keys():
            setattr(self, f, d[str(f)])
            
    def remove_tensors(self):
        for f in self.fields:
            if getattr(self, f) is not None:
                if type(getattr(self, f)[0]) == torch.Tensor:  
                    setattr(self, f, None)
