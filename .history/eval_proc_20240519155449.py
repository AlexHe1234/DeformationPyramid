import numpy as np
a = np.load('cndp_mc.npy', allow_pickle=True).item()
ate = 0.
d1 = 0.
d2 = 0.
for f in range(len(a['ate'])):
