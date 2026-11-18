"""Make an apple-only, image-group-disjoint subset from extracted official ATLDSD.

Source: Feng and Chao, https://doi.org/10.11922/sciencedb.01627
Dataset license: CC BY-NC-SA 4.0. Source images are not bundled with this project.
"""
from pathlib import Path
import argparse,hashlib,json,re,shutil
import numpy as np
from PIL import Image


def prepare(root,output,archive=None,seed=18,train_per_category=32,validation_per_category=8,test_per_category=16):
    root,output=Path(root),Path(output)
    if (output/'manifest.json').exists():raise ValueError('output manifest already exists')
    expected='7441604c86d65090606e40dc7d1f8540'
    archive_md5=archive_sha256=None
    if archive is not None:
        raw=Path(archive).read_bytes();archive_md5=hashlib.md5(raw).hexdigest();archive_sha256=hashlib.sha256(raw).hexdigest()
        if archive_md5!=expected:raise ValueError('archive checksum does not match official ATLDSD.rar V1')
    candidates=[]
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():continue
        images=sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg'} and p.parent.name.lower() in {'image','images'})
        masks={p.stem:p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in {'.png','.bmp'} and p.parent.name.lower() not in {'image','images'}}
        for image in images:
            mask=masks.get(image.stem)
            if mask is None:raise ValueError(f'missing annotation for {image}')
            with Image.open(image) as im:
                rgb=np.asarray(im.convert('RGB'))
                small=np.asarray(im.convert('L').resize((9,8),Image.Resampling.BILINEAR))
            signature=hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()
            bits=(small[:,1:]>small[:,:-1]).ravel()
            dhash=int.from_bytes(np.packbits(bits).tobytes(),'big')
            candidates.append({'image':image,'mask':mask,'category':folder.name,'hash':signature,'dhash':dhash})
    if not candidates:raise ValueError('no category/image/*.jpg with matching annotations found')
    # Union exact/near-identical images before splitting. This is a conservative
    # low-resolution dHash check, not proof of independence of physical leaves.
    parent=list(range(len(candidates)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,row in enumerate(candidates):
        for j in range(i):
            other=candidates[j]
            if row['hash']==other['hash'] or (row['dhash']^other['dhash']).bit_count()<=3:
                parent[find(i)]=find(j)
    grouped={}
    for i,row in enumerate(candidates):grouped.setdefault(find(i),[]).append(row)
    grouped_by_category={};dropped_conflicting_groups=0
    for group_id,rows in grouped.items():
        categories={r['category'] for r in rows}
        if len(categories)>1:
            dropped_conflicting_groups+=1;continue
        # One representative per group; no other representative is used elsewhere.
        row=rows[0].copy();row['group']=f'group-{group_id:05}'
        grouped_by_category.setdefault(row['category'],[]).append(row)
    rng=np.random.default_rng(seed)
    payload={'kind':'real_atldsd_custom_grouped_subset','crop':'apple','seed':seed,
             'description':'Real apple leaf photographs with source semantic leaf/lesion annotations; custom category-balanced grouped subset, not an official benchmark protocol',
             'source':'https://doi.org/10.11922/sciencedb.01627','creators':['Jingze Feng','Xiaofei Chao'],
             'license':'CC BY-NC-SA 4.0','archive_md5':archive_md5,'archive_sha256':archive_sha256,
             'candidate_images':len(candidates),'near_duplicate_grouping':'64-bit grayscale dHash Hamming distance <=3 or exact decoded content; one representative/group',
             'dropped_cross_category_groups':dropped_conflicting_groups,
             'train':[],'validation':[],'test':[]}
    counts=[train_per_category,validation_per_category,test_per_category]
    for category,rows in sorted(grouped_by_category.items()):
        required=sum(counts)
        if len(rows)<required:raise ValueError(f'{category}: only {len(rows)} groups; need {required}')
        rows=[rows[i] for i in rng.permutation(len(rows))[:required]]
        start=0;slug=re.sub(r'[^a-z0-9]+','-',category.lower()).strip('-')
        for split,count in zip(['train','validation','test'],counts):
            (output/split).mkdir(parents=True,exist_ok=True)
            for row in rows[start:start+count]:
                name=f"{slug}_{row['image'].stem}"
                image=f'{split}/{name}.jpg';mask=f'{split}/{name}_mask.png'
                shutil.copy2(row['image'],output/image)
                # Source masks use VOC palette indices: 0 background, 1 normal
                # tissue, 2..5 disease-specific lesion classes. Collapse the
                # symptom classes, preserving their union as visible damage.
                with Image.open(row['mask']) as annotation:raw_labels=np.asarray(annotation)
                if raw_labels.ndim!=2 or not set(np.unique(raw_labels)).issubset(set(range(6))):
                    raise ValueError(f'unsupported ATLDSD label indices: {row["mask"]}')
                target=np.zeros(raw_labels.shape,np.uint8);target[raw_labels==1]=1;target[raw_labels>=2]=2
                Image.fromarray(target).save(output/mask)
                payload[split].append({'image':image,'mask':mask,'group':row['group'],'category':category,
                                       'source_image':str(row['image'].relative_to(root)),
                                       'source_mask':str(row['mask'].relative_to(root)),
                                       'source_label_indices':np.unique(raw_labels).tolist(),
                                       'decoded_image_sha256':row['hash']})
            start+=count
    (output/'manifest.json').write_text(json.dumps(payload,indent=2)+'\n')
    return {k:len(payload[k]) for k in ['train','validation','test']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--archive',type=Path);p.add_argument('--seed',type=int,default=18)
    p.add_argument('--train-per-category',type=int,default=32)
    p.add_argument('--validation-per-category',type=int,default=8)
    p.add_argument('--test-per-category',type=int,default=16)
    a=p.parse_args();print(json.dumps(prepare(a.root,a.output,a.archive,a.seed,a.train_per_category,a.validation_per_category,a.test_per_category)))
