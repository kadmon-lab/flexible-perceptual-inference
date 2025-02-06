import numpy as np; import torch; import bz2; import _pickle as c; import json; import gc

""" SINGLE MOUSE """

class mouse_data_struct():
    def __init__(self, fields, size, batch_len):
        self.size = size
        self.start_ind = self.special_ind = 0 
        self.fields = fields
        self.batch_len = batch_len
        for f in fields:
            setattr(self, f, np.empty(self.size, dtype = object))
        
    def push_multiple(self, fields, datas):
          for field, data in zip(fields, datas):
              self.push(field, data)
              
    def push(self, field_name, new_data):
        data = getattr(self, field_name)
        # single = type(new_data) == int or type(new_data) == float 
        single = len(np.array([new_data]).shape) == 1
        special = (not single and len (new_data) != self.batch_len)    
        self.add_single(data, new_data) if single else self.add_special(data, new_data) if special else self.add_array(data, new_data) 
        setattr(self, field_name, data)
              
    def add_single(self, data, new_data):
        data[self.start_ind : self.start_ind + self.batch_len] = new_data
        
    def add_special(self, data, new_data):
        data[self.special_ind] = new_data
        data[self.special_ind + 1] = 'SPECIAL'
        
    def add_array(self, data, new_data):
        for i in range(self.batch_len):
             data[self.start_ind + i] = new_data[i]
        
    def increment(self):
        self.start_ind += self.batch_len
        self.special_ind += 1
        
    def to_dictionary(self, d = dict()):
        for f in self.fields:
            d[f] = getattr(self, f)
        return d 
    
    def from_dictionary(self, d):
        for f in d.keys():
            setattr(self, f, d[f])
            
    def remove_tensors(self):
        for f in self.fields:
            data = getattr(self, f)
            if data is not None:
                if (type(data[0]) == torch.Tensor):
                    if len(data[0].shape) > 1:  
                        setattr(self, f, None)
                    else:
                        numpied = np.array([d.cpu().numpy().astype(float) for d in data], dtype = float)
                        setattr(self, f, numpied)
                
    def postprocess_specials(self):
        for f in self.fields:
            data = getattr(self, f)
            if data is not None:
                if len(data) > self.special_ind and np.array(data[self.special_ind] == 'SPECIAL').any():
                    setattr(self, f, data[:self.special_ind])

""" CAGE FOR COMPARING MICE """

class cage_data_struct():
    def __init__(self, location, names = None, load = False):
        self.location = location + '.pickle'
        if load is False:
            self.new_struct(names)
        else:
            self.load_struct()

    def new_struct(self, names):
        self.d = dict()
        self.d["names"] = dict()
        for self.name in names:
            self.add_name(name = self.name)
    
    def add_name(self, name):
        self.d[name] = dict()
        self.d["names"][name] = dict()
        self.d["names"][name]["sample_num"] = 0               
        
    def add_data(self, name, manager):
        self.name = name
        self.sample = self.d["names"][name]["sample_num"]
        self.d["names"][self.name]["sample_num"] += 1   
        self.d[self.name][self.sample] = dict()
        self.d[self.name][self.sample] = manager.data.to_dictionary()
        self.d[self.name][self.sample]['network'] = manager.agent.state_dict()
        self.compress_mouse()
        self.save()

    def change_name(self, old, new):
        self.d["names"][new] = self.d["names"].pop(old)
        self.d[new] = self.d.pop(old)
        self.save()
        
    def save(self):
        print("saving cage")
        gc.collect()
        with open(self.location, 'wb') as f: 
            c.dump(self.d, f)    

    def load_struct(self):        
        with open(self.location, 'rb') as f:
            self.d = c.load(f)
        print("cage loaded")
          
    def structure(self):
        print("[condition][sample number][field]")
        
    def names(self):
        return list(self.d["names"].keys())

    def info(self):
        return self.d["names"]

    def get_num(self, name): 
        return self.d["names"][name]["sample_num"]

    def get_data(self, name, sample, field = None):
        if field is None:
            return dict_to_attr(self.d[name][sample])
        return self.d[name][sample][field]
    
    def show_data(self, name, sample):
        print(self.d[name][sample].keys())
            
    def compress(self):
        for self.name in self.d["names"]:            
            for self.sample in self.d[self.name]:
                print(f"{self.name} {self.sample}")
                self.compress_mouse()
    
    def compress_mouse(self):        
        self.purge_extra_fields()
        self.compress_fields()
                    
    def purge_extra_fields(self):
        keys =  self.d[self.name][self.sample].keys()
        purge = ['backbone', 'stim','gos', 'nogos',"lick_prob", "value", "Qs", "net_input", "net_output", "LTM", "f_gate", "i_gate", "c_gate", "o_gate"]
        for field in purge:
            if field in keys:
                del(self.d[self.name][self.sample][field])
                
    def compress_fields(self):
        E = self.d[self.name][self.sample]['episode'][-1] + 1 
        T = self.d[self.name][self.sample]['trial'][-1] + 1 
        keys =  self.d[self.name][self.sample].keys()
        for self.field in keys:
            self.OG_data = self.d[self.name][self.sample][self.field]
            if len(self.OG_data) == E * T:                                 # ignore elements that are not episode x trial length
                print(f"compressing {self.field}")
                self.compress_array()
            del(self.OG_data)
                
    def compress_array(self, i = 0):
        while len(np.shape(self.OG_data[i])):                                    # 1 if empty, 0 if leaf
            i += 1 
        self.first = self.OG_data[i]
        self.compress_type()

    def compress_type(self):
        if (np.any(self.OG_data == None) or np.any(np.isnan(self.OG_data.astype(float)))):
            T = 'float16'
        else: 
            is_float = isinstance(self.first, np.float16) or isinstance(self.first, np.float32) or isinstance(self.first, np.float64) or isinstance(self.first, float)
            is_int = isinstance(self.first, int) or self.first.is_integer()
            if is_float:
                T = 'float32'
            if is_int:
                data_min = self.OG_data[self.OG_data != None].min()
                data_max = self.OG_data[self.OG_data != None].max()
                T = 'int8' if (data_max < 100 and data_min > -100) else 'int16'
                if data_min >=0:
                    T = 'u'+T   
        self.d[self.name][self.sample][self.field] = self.OG_data.astype(T)
                    
class dict_to_attr():
    def __init__(self, d):
        for k in d.keys():
            setattr(self, k, d[k])

