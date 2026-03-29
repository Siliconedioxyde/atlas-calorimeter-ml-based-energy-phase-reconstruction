# this file is used just to "test the waters" and get a feel for the data,
# plotting it, seeing it maximum and minimums, visualizing your data is good in general.


import torch
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the first training file
data = torch.load('data/train_00000.pt') 


samples = data['X'].numpy() 
targets = data['y'].numpy()
OFtargets= data['y_OF'].numpy()    

row_index = 1000

pulse_samples = samples[row_index, 1, :]  
true_energy  = targets[row_index, 0]      
of_energy =OFtargets[row_index, 0]
#plt.figure(figsize=(8,5))

# plt.plot(range(-3,4), pulse_samples, '-o', label='Digital Samples (sample_lo)')
# plt.axhline(y=true_energy, color='r', linestyle='--',
#             label=f'True Energy: {true_energy:.4f}')
# plt.axhline(y=ofenergy, color='g', linestyle='-.',
#             label=f'OF Energy: {ofenergy:.4f}')

# plt.title(f"ATLAS TileCal Pulse (Event {row_index})")
# plt.xlabel("Relative BC index")
# plt.ylabel("ADC counts")
# plt.legend()
# plt.grid(True)
# plt.show()
# Check the maximum values in the entire file
print("Max of Column 0:", np.max(data['y'][:, 0].numpy()))
print("Max of Column 1:", np.max(data['y'][:, 1].numpy()))

# Check if y[n,0] and y[n,1] are identical for all n they seem to differ by a tiny amount, probably because of some precision perhaps
are_identical = np.array_equal(data['y'][:, 0].numpy(), data['y'][:, 1].numpy())
print(f"Are they 100% identical? {are_identical}")

print(OFtargets[1:10], targets[0:10])
for i in range (10):
    print (targets[i][0], targets[i][1])
