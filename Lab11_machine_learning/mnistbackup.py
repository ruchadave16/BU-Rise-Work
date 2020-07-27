import pickle

def save_obj(obj, name="mnist" ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name="mnist"  ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding='bytes')