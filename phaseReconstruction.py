import torch
import glob
import numpy as np
import matplotlib.pyplot as plt


# grouping all test files together
test_files = sorted( glob.glob("data/test/*.pt"))

y_test_list = []

for file in test_files:
    data = torch.load(file)

    y_lg = data['y'][:, 1].numpy()

    y_test_list.append(y_lg)

y_test = np.concatenate(y_test_list)
samples = data['X'][:, 1, :].numpy()  

# Define indices
bc_indices = np.arange(-3, 4)   # [-3,-2,-1,0,1,2,3]

# calculating the center of mass using the formula shown in the report
numerator = np.sum(samples * bc_indices, axis=1)
denominator = np.sum(samples, axis=1)

time_shift = numerator / (denominator + 1e-8) / 3  # deviding by 3 to map to [-1,1], and the addition of 1e-8 is to avoid division by zero
print("Example time shifts:") #checking few
print(time_shift[:10])

#mean and std of the time shift distribution
mean_time = np.mean(time_shift)
std_time  = np.std(time_shift)
print(mean_time)
print(std_time)

plt.hist(time_shift, bins=100)
plt.xlabel("Estimated Time Shift")
plt.ylabel("Counts")
plt.title("Pulse Timing Distribution")
plt.grid(True)
plt.show()

