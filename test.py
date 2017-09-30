from models import *
import h5py

def load_data(file, name):
    with h5py.File(file, 'r') as hf:
        return hf[name][:]

model = DAE()

x = load_data(file='puck_state_pp_first_half.h5', name='state')

model.train(x)