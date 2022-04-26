import os
import numpy as np
import shutil
lst = os.listdir('./Hackathon_data')
print(lst)


if os.path.exists('train'):
    shutil.rmtree('train')
os.mkdir('train')
if os.path.exists('validation'):
    shutil.rmtree('validation')
os.mkdir('validation')

ddict = {}
for d in lst:
    d_dir = os.path.join('train', d)
    os.mkdir(d_dir)
    d_dir = os.path.join('validation', d)
    os.mkdir(d_dir)
    ddict[d] = os.listdir(os.path.join('Hackathon_data', d))

rnds = np.random.choice(500, 100, replace=True)
rnds = np.random.permutation(np.arange(500))[0:100]
print(rnds)
print(len(rnds))
for i in range(1,501):
    for d in lst:
        tmp = [f for f in ddict[d] if '_'+str(i)+'.png' in f]
        filename = tmp[0]
        filepath = os.path.join(os.path.join('Hackathon_data',d), filename)
        #print(filepath)
        if i in rnds:
            #get the file path
            destpath = os.path.join('validation',d)
        else: 
            destpath = os.path.join('train',d)
        destpath = os.path.join(destpath, filename)
        shutil.copyfile(filepath, destpath)
            
        


