import numpy as np
a = np.load('cndp_mc.npy', allow_pickle=True).item()
ate = 0.
d1 = 0.
d2 = 0.
for f in range(len(a['ate'])):
    ate += a['ate'][f].cpu().numpy()
    d1 += a['0.1'][f].cpu().numpy()
    d2 += a['0.2'][f].cpu().numpy()
print(f'ate {ate / len(a['ate'])} d1 {d1 / len(a['ate'])} d2 {d2 / len(a['ate'])}')
