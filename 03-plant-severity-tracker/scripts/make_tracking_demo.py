"""Create a clearly synthetic same-leaf progression and track actual model outputs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from leafseverity.core import SeverityModel,rgb_image,label_image,save_prediction
from leafseverity.tracking import append_observation


def generate(model_path,image_path,mask_path,output):
    output=Path(output)
    if (output/'observations.json').exists():raise ValueError('tracking demo output already exists')
    output.mkdir(parents=True,exist_ok=True)
    rgb=rgb_image(image_path);truth=label_image(mask_path);leaf=truth>0
    if not leaf.any():raise ValueError('example needs a nonempty synthetic leaf')
    healthy=(truth==1)
    base=rgb.copy();base[truth==2]=np.median(rgb[healthy],axis=0)
    ys,xs=np.nonzero(leaf);rng=np.random.default_rng(18)
    y,x=np.mgrid[:leaf.shape[0],:leaf.shape[1]];damage=np.zeros_like(leaf)
    model=SeverityModel.load(model_path);model_id=hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    rows=[]
    for day,spots in enumerate([0,3,7]):
        for _ in range(spots):
            chosen=int(rng.integers(len(ys)));cx,cy=xs[chosen],ys[chosen]
            damage|=((x-cx)**2+(y-cy)**2<10**2)&leaf
        frame=base.copy();frame[damage]=[.43,.24,.09]
        folder=output/f'day_{day}';folder.mkdir()
        Image.fromarray(np.uint8(np.clip(frame,0,1)*255)).save(folder/'input.png')
        summary,pred_leaf,pred_lesion=model.predict(frame)
        save_prediction(frame,summary,pred_leaf,pred_lesion,folder)
        observations=append_observation(output/'observations.json','synthetic-apple-1','front-flat',
                                        f'2026-11-{16+day:02}',model_id,summary)
        rows.append({'day':day,'data_kind':'synthetic','truth_affected_area_percent':float(100*damage.sum()/leaf.sum()),
                     'estimated_affected_area_percent':summary['affected_area_percent'],
                     'delta_percentage_points':observations[-1]['delta_percentage_points']})
    (output/'sequence_results.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--image',type=Path,required=True)
    p.add_argument('--mask',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(generate(a.model,a.image,a.mask,a.output),indent=2))
