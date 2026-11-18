import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from .core import SeverityModel,Config,rgb_image,save_prediction
from .data import synthetic,read_manifest
from .evaluate import evaluate
from .tracking import append_observation,history


def main(argv=None):
    p=argparse.ArgumentParser(description='Apple-only visible leaf damage segmentation and tracking')
    sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('demo');d.add_argument('--output',type=Path,default=Path('outputs/synthetic-demo'))
    d.add_argument('--seed',type=int,default=18)
    for name in ['fit','evaluate']:
        command=sub.add_parser(name);command.add_argument('--manifest',type=Path,required=True)
        command.add_argument('--output',type=Path,required=True);command.add_argument('--seed',type=int,default=18)
    for name in ['predict','track']:
        command=sub.add_parser(name);command.add_argument('--model',type=Path,required=True)
        command.add_argument('--image',type=Path,required=True);command.add_argument('--output',type=Path,required=True)
        command.add_argument('--leaf-mask',type=Path)
        if name=='track':
            command.add_argument('--store',type=Path,required=True);command.add_argument('--leaf-id',required=True)
            command.add_argument('--view-id',required=True);command.add_argument('--date',required=True)
    h=sub.add_parser('history');h.add_argument('--store',type=Path,required=True)
    h.add_argument('--leaf-id');h.add_argument('--view-id')
    a=p.parse_args(argv)
    try:
        if a.command=='demo':evaluate(synthetic(a.output/'data',a.seed),a.output/'evaluation',a.seed)
        elif a.command=='evaluate':evaluate(a.manifest,a.output,a.seed)
        elif a.command=='fit':
            _,train,val,_=read_manifest(a.manifest)
            model=SeverityModel(Config(seed=a.seed)).fit(train).calibrate(val)
            model.save(a.output);print(json.dumps({'model':str(a.output),'leaf_threshold':model.leaf_threshold,'lesion_threshold':model.lesion_threshold}))
        elif a.command in ['predict','track']:
            model=SeverityModel.load(a.model);rgb=rgb_image(a.image);manual=None
            if a.leaf_mask:
                with Image.open(a.leaf_mask) as im:manual=np.asarray(im.convert('L'))>0
            summary,leaf,lesion=model.predict(rgb,manual);save_prediction(rgb,summary,leaf,lesion,a.output)
            if a.command=='track':
                observations=append_observation(a.store,a.leaf_id,a.view_id,a.date,hashlib.sha256(a.model.read_bytes()).hexdigest(),summary)
                print(json.dumps(observations,indent=2))
            else:print(json.dumps(summary,indent=2))
        elif a.command=='history':print(json.dumps(history(a.store,a.leaf_id,a.view_id),indent=2))
    except (ValueError,FileNotFoundError) as exc:p.error(str(exc))


if __name__=='__main__':main()
