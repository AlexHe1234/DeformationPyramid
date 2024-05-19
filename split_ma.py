import os
import json
import numpy as np

# 530
path2ma = 'data/mixamo_cmu'
fs = sorted([f for f in os.listdir(path2ma) if '.npy' in f])
num = len(fs) // 5
val_test_inds = np.random.randint(0, 5, num)
base = np.arange(num) * 5
val_test_inds = base + val_test_inds
val_test_seq = [fs[i] for i in val_test_inds]
train_seq = [s for s in fs if s not in val_test_seq]

val_choice = np.random.choice(val_test_inds, len(val_test_seq) // 2)
val_seq = [fs[i] for i in val_choice]
test_seq =  [f for f in val_test_seq if f not in val_seq]

with open(os.path.join(path2ma, '0train.json'),'w') as f:
    json.dump(train_seq,f)
    
with open(os.path.join(path2ma, '0val.json'),'w') as f:
    json.dump(val_seq,f)
    
with open(os.path.join(path2ma, '0test.json'),'w') as f:
    json.dump(test_seq,f)

