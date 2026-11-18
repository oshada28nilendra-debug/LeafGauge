from pathlib import Path
import json
import numpy as np
from PIL import Image
from .core import image_hash,label_image,rgb_image


def synthetic(root,seed=18):
    root=Path(root)
    if (root/'manifest.json').exists():raise ValueError("output already contains a manifest")
    rng=np.random.default_rng(seed);size=224;y,x=np.mgrid[:size,:size]
    payload={'kind':'synthetic_smoke_test','crop':'apple','seed':seed,
             'description':'Procedural apple-like leaf shapes and inserted brown lesions; not botanical observations',
             'train':[],'validation':[],'test':[]}
    for split,count in [('train',24),('validation',8),('test',12)]:
        (root/split).mkdir(parents=True,exist_ok=True)
        for i in range(count):
            angle=rng.uniform(-.4,.4);cx,cy=rng.uniform(100,124,size=2)
            u=(x-cx)*np.cos(angle)+(y-cy)*np.sin(angle)
            v=-(x-cx)*np.sin(angle)+(y-cy)*np.cos(angle)
            rx,ry=rng.uniform(55,70),rng.uniform(78,96)
            leaf=(u/rx)**2+(v/ry)**2<1
            background=rng.uniform(.42,.8,size=3)
            rgb=np.broadcast_to(background,(size,size,3)).copy()
            rgb+=rng.normal(0,.009,rgb.shape)
            color=np.array([.20,.48,.13])+rng.uniform(-.035,.035,size=3)
            rgb[leaf]=color+rng.normal(0,.025,(leaf.sum(),3))
            vein=leaf&(np.abs(u)<1.5);rgb[vein]=[.43,.63,.25]
            lesion=np.zeros_like(leaf)
            spots=int(rng.integers(0,9))
            for j in range(spots):
                sx,sy=rng.uniform(60,164,size=2);r=rng.uniform(4,15)
                lesion|=((x-sx)**2+(y-sy)**2<r*r)&leaf
            rgb[lesion]=np.array([.43,.24,.09])+rng.normal(0,.025,(lesion.sum(),3))
            target=np.zeros((size,size),np.uint8);target[leaf]=1;target[lesion]=2
            name=f'{split}/{i:03}.png';mask=f'{split}/{i:03}_mask.png'
            Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)).save(root/name)
            Image.fromarray(target).save(root/mask)
            payload[split].append({'image':name,'mask':mask,'group':f'{split}-{i}','category':'procedural_apple'})
    (root/'manifest.json').write_text(json.dumps(payload,indent=2)+'\n')
    return root/'manifest.json'


def read_manifest(path):
    path=Path(path);payload=json.loads(path.read_text())
    if payload.get('crop')!='apple':raise ValueError("this prototype supports apple only")
    splits=[];seen=set();groups_seen=set()
    for split in ['train','validation','test']:
        rows=[];hashes=[];groups=set()
        for row in payload.get(split,[]):
            resolve=lambda p:Path(p) if Path(p).is_absolute() else path.parent/p
            record={**row,'image':resolve(row['image']),'mask':resolve(row['mask'])}
            if label_image(record['mask']).shape!=rgb_image(record['image']).shape[:2]:raise ValueError("image/mask dimensions differ")
            hashes.append(image_hash(record['image']))
            if row.get('group'):groups.add(row['group'])
            rows.append(record)
        if not rows:raise ValueError("all manifest splits must be nonempty")
        if len(hashes)!=len(set(hashes)) or set(hashes)&seen:raise ValueError("duplicate image content or split leakage")
        if groups&groups_seen:raise ValueError("group leakage across splits")
        seen.update(hashes);groups_seen.update(groups);splits.append(rows)
    return payload,*splits
