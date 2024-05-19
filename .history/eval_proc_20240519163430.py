import numpy as np
import statistics
a = np.load('cndp_mc.npy', allow_pickle=True).item()
ate = 0.
d1 = []
d2 = []
time = 0.
for f in range(len(a['ate'])):
    ate += a['ate'][f].cpu().numpy()
    d1.append(a['0.1'][f].cpu().numpy())
    d2.append(a['0.2'][f].cpu().numpy())
    time += a['time'][f]
F = len(a['ate'])
print(f'ate {ate / F} d1 {statistics.median(d1)} d2 {statistics.median(d2)} time {time / F}')



