"""Frozen pretrained ResNet features + a supervised multinomial pixel classifier.

Classes: 0 background, 1 healthy-looking apple leaf tissue, 2 visible lesion.
Only the linear pixel head is trained. No plant diagnosis or treatment is inferred.
"""
from dataclasses import asdict,dataclass
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class Config:
    size:int=224
    pixels_per_class_per_image:int=160
    seed:int=18
    threads:int=2
    regularization:float=1.0

    def validate(self):
        if not 64<=self.size<=512 or self.size%32:
            raise ValueError("size must be a multiple of 32 between 64 and 512")
        if self.pixels_per_class_per_image<1 or self.threads<1 or self.regularization<=0:
            raise ValueError("invalid sampling, thread or regularization configuration")


def rgb_image(path:Path):
    # Preserve stored pixel orientation to match dataset masks. Users should
    # export a normally oriented image and matching masks before training.
    with Image.open(path) as im:return np.asarray(im.convert('RGB'),np.float32)/255


def image_hash(path:Path):
    array=rgb_image(path)
    return hashlib.sha256(str(array.shape).encode()+np.uint8(array*255).tobytes()).hexdigest()


def label_image(path:Path):
    with Image.open(path) as im:
        raw=np.asarray(im)
        if raw.ndim==2:
            if not set(np.unique(raw)).issubset({0,1,2}):
                raise ValueError("indexed label masks must use 0 background, 1 leaf, 2 lesion")
            return raw.astype(np.uint8)
        rgb=np.asarray(im.convert('RGB'))
    # Exact three-color semantic palette; other source palettes must be
    # explicitly converted in the dataset preparation script.
    output=np.zeros(rgb.shape[:2],np.uint8)
    green=(rgb[:,:,1]>64)&(rgb[:,:,0]<64)&(rgb[:,:,2]<64)
    red=(rgb[:,:,0]>64)&(rgb[:,:,1]<64)&(rgb[:,:,2]<64)
    black=(rgb.max(axis=2)<64)
    if not (green|red|black).all():
        raise ValueError("unsupported label palette; convert masks to indexed 0/1/2")
    output[green]=1;output[red]=2
    return output


def letterbox(rgb,size):
    h,w=rgb.shape[:2];scale=min(size/w,size/h)
    nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
    left,top=(size-nw)//2,(size-nh)//2
    resized=Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)).resize((nw,nh),Image.Resampling.BILINEAR)
    canvas=np.full((size,size,3),.5,np.float32)
    canvas[top:top+nh,left:left+nw]=np.asarray(resized,np.float32)/255
    return canvas,(left,top,nw,nh)


def labels_on_canvas(labels,size,box):
    left,top,w,h=box
    target=np.zeros((size,size),np.uint8)
    target[top:top+h,left:left+w]=np.asarray(Image.fromarray(labels).resize((w,h),Image.Resampling.NEAREST))
    valid=np.zeros_like(target,dtype=bool);valid[top:top+h,left:left+w]=True
    return target,valid


class FeatureExtractor:
    dimensions=207
    def __init__(self,config):
        import torch
        import torchvision
        from torchvision.models import resnet18,ResNet18_Weights
        config.validate();torch.set_num_threads(config.threads)
        self.torch,self.config=torch,config
        self.versions={'torch':torch.__version__,'torchvision':torchvision.__version__}
        backbone=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
        self.stem=torch.nn.Sequential(backbone.conv1,backbone.bn1,backbone.relu,backbone.maxpool)
        self.layer1,self.layer2=backbone.layer1,backbone.layer2
        for module in [self.stem,self.layer1,self.layer2]:
            module.eval()
            for p in module.parameters():p.requires_grad_(False)
        self.mean=torch.tensor([.485,.456,.406])[None,:,None,None]
        self.std=torch.tensor([.229,.224,.225])[None,:,None,None]

    def __call__(self,rgb):
        t=self.torch;canvas,box=letterbox(rgb,self.config.size)
        x=t.from_numpy(canvas.transpose(2,0,1).copy())[None]
        with t.inference_mode():
            a=self.layer1(self.stem((x-self.mean)/self.std));b=self.layer2(a)
            a=t.nn.functional.interpolate(a,size=canvas.shape[:2],mode='bilinear',align_corners=False)
            b=t.nn.functional.interpolate(b,size=canvas.shape[:2],mode='bilinear',align_corners=False)
            dense=t.cat([a,b],dim=1)[0].permute(1,2,0).numpy().copy()
        # Color and local texture preserve details below the CNN feature stride.
        chroma=canvas/np.maximum(canvas.sum(axis=2,keepdims=True),.05)
        hsv=np.asarray(Image.fromarray(np.uint8(canvas*255)).convert('HSV'),np.float32)/255
        local_mean=uniform_filter(canvas,size=(7,7,1),mode='reflect')
        local_std=np.sqrt(np.maximum(uniform_filter(canvas**2,size=(7,7,1))-local_mean**2,0))
        combined=np.concatenate([dense,canvas,chroma,hsv,local_mean,local_std],axis=2)
        return combined.astype(np.float32),box


def dice(a,b):
    a,b=np.asarray(a,bool),np.asarray(b,bool)
    total=int(a.sum()+b.sum())
    return 1.0 if total==0 else float(2*np.logical_and(a,b).sum()/total)


def masks_from_probabilities(probabilities,leaf_threshold,lesion_threshold):
    leaf=(probabilities[:,:,1]+probabilities[:,:,2])>=leaf_threshold
    lesion=leaf&(probabilities[:,:,2]>=lesion_threshold)
    return leaf,lesion


def summarize_masks(leaf,lesion):
    leaf,lesion=np.asarray(leaf,bool),np.asarray(lesion,bool)
    if leaf.shape!=lesion.shape:raise ValueError("mask shapes must match")
    if (lesion&~leaf).any():raise ValueError("lesion must be inside the leaf")
    n=int(leaf.sum());d=int(lesion.sum())
    return {'leaf_pixels':n,'lesion_pixels':d,'affected_area_percent':None if n==0 else 100*d/n}


class SeverityModel:
    def __init__(self,config=None):
        self.config=config or Config();self.config.validate()
        self.extractor=FeatureExtractor(self.config)
        self.mean=self.scale=self.coef=self.intercept=None
        self.leaf_threshold=self.lesion_threshold=.5
        self.metadata={'schema':1,'crop':'apple','classes':['background','healthy_tissue','lesion'],
                       'backbone':'resnet18_IMAGENET1K_V1_layer1_layer2','backbone_finetuned':False,
                       'versions':self.extractor.versions}

    def fit(self,records):
        if not records:raise ValueError("annotated training images are required")
        rng=np.random.default_rng(self.config.seed);rows=[];labels=[];hashes=[]
        for record in records:
            rgb=rgb_image(record['image']);truth=label_image(record['mask'])
            if truth.shape!=rgb.shape[:2]:raise ValueError("image and label dimensions differ")
            hashes.append(image_hash(record['image']))
            dense,box=self.extractor(rgb);target,valid=labels_on_canvas(truth,self.config.size,box)
            for cls in range(3):
                indices=np.flatnonzero((target==cls)&valid)
                if len(indices):
                    indices=rng.choice(indices,min(len(indices),self.config.pixels_per_class_per_image),replace=False)
                    rows.append(dense.reshape(-1,dense.shape[-1])[indices]);labels.append(np.full(len(indices),cls))
        if len(hashes)!=len(set(hashes)):raise ValueError("duplicate training images")
        matrix=np.concatenate(rows);y=np.concatenate(labels)
        if set(np.unique(y))!={0,1,2}:raise ValueError("training requires all three pixel classes")
        scaler=StandardScaler().fit(matrix)
        # Use training pixels only; the frozen backbone receives no gradient.
        classifier=LogisticRegression(C=self.config.regularization,max_iter=500,solver='lbfgs',random_state=self.config.seed)
        classifier.fit(scaler.transform(matrix),y)
        self.mean,self.scale=scaler.mean_.astype(np.float32),scaler.scale_.astype(np.float32)
        self.coef,self.intercept=classifier.coef_.astype(np.float32),classifier.intercept_.astype(np.float32)
        self.metadata.update({'training_hashes':hashes,'training_images':len(records),
                              'sampled_training_pixels':len(y),'head_iterations':classifier.n_iter_.tolist(),
                              'head_converged_before_iteration_limit':bool(np.max(classifier.n_iter_)<500)})
        return self

    def probabilities(self,rgb):
        if self.coef is None:raise ValueError("fit or load a model first")
        if rgb.ndim!=3 or rgb.shape[2]!=3 or not np.isfinite(rgb).all():raise ValueError("expected finite RGB image")
        dense,box=self.extractor(rgb)
        matrix=(dense.reshape(-1,dense.shape[-1])-self.mean)/self.scale
        logits=matrix@self.coef.T+self.intercept
        return softmax(logits,axis=1).reshape(self.config.size,self.config.size,3),box

    def calibrate(self,records):
        if not records:raise ValueError("annotated validation images are required")
        cache=[];hashes=[]
        for record in records:
            h=image_hash(record['image']);hashes.append(h)
            if h in self.metadata['training_hashes']:raise ValueError("training/validation overlap")
            rgb=rgb_image(record['image']);gt=label_image(record['mask'])
            if gt.shape!=rgb.shape[:2]:raise ValueError("validation mask dimensions differ")
            probs,box=self.probabilities(rgb);truth,valid=labels_on_canvas(gt,self.config.size,box)
            cache.append((probs,truth,valid))
        if len(hashes)!=len(set(hashes)):raise ValueError("duplicate validation images")
        candidates=[]
        for lt in [.35,.5,.65]:
            for dt in [.25,.4,.55,.7]:
                scores=[]
                for probs,truth,valid in cache:
                    leaf,lesion=masks_from_probabilities(probs,lt,dt)
                    scores.append((dice(leaf&valid,(truth>0)&valid)+dice(lesion&valid,(truth==2)&valid))/2)
                candidates.append((float(np.mean(scores)),lt,dt))
        best=max(candidates,key=lambda r:r[0])
        _,self.leaf_threshold,self.lesion_threshold=best
        self.metadata.update({'validation_hashes':hashes,'validation_images':len(records),
                              'threshold_objective':'mean of per-image leaf Dice and lesion Dice on validation only',
                              'validation_objective':best[0],'threshold_grid_results':candidates})
        return self

    def predict(self,rgb,manual_leaf=None):
        probs,box=self.probabilities(rgb);left,top,w,h=box
        leaf,lesion=masks_from_probabilities(probs,self.leaf_threshold,self.lesion_threshold)
        native=(rgb.shape[1],rgb.shape[0])
        def restore(mask):return np.asarray(Image.fromarray(np.uint8(mask[top:top+h,left:left+w])*255).resize(native,Image.Resampling.NEAREST))>0
        leaf,lesion=restore(leaf),restore(lesion)
        if manual_leaf is not None:
            if manual_leaf.shape!=leaf.shape:raise ValueError("manual leaf mask dimensions differ")
            leaf=np.asarray(manual_leaf,bool)
            # Recompute lesions using the model score inside the supplied denominator.
            raw=restore(probs[:,:,2]>=self.lesion_threshold);lesion=raw&leaf
        summary=summarize_masks(leaf,lesion);warnings=[]
        if summary['leaf_pixels']==0:warnings.append('no_leaf_detected')
        elif summary['leaf_pixels']/leaf.size<.01:warnings.append('leaf_too_small_in_image')
        if leaf[0].any() or leaf[-1].any() or leaf[:,0].any() or leaf[:,-1].any():warnings.append('leaf_reaches_image_border_visible_area_only')
        visible_probs=probs[top:top+h,left:left+w]
        summary.update({'crop':'apple','quality_warnings':warnings,'manual_leaf_mask':manual_leaf is not None,
                        'low_margin_pixel_fraction':float((visible_probs.max(axis=2)<.6).mean()),
                        'leaf_threshold':self.leaf_threshold,'lesion_threshold':self.lesion_threshold,
                        'quantity':'estimated visibly affected area / estimated visible leaf area; not pathological grade'})
        return summary,leaf,lesion

    def save(self,path):
        if self.coef is None:raise ValueError("cannot save an unfitted model")
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        meta={**self.metadata,'config':asdict(self.config),'leaf_threshold':self.leaf_threshold,'lesion_threshold':self.lesion_threshold}
        with path.open('wb') as f:np.savez_compressed(f,mean=self.mean,scale=self.scale,coef=self.coef,intercept=self.intercept,metadata=np.array(json.dumps(meta)))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['metadata']))
            if meta.get('schema')!=1 or meta.get('crop')!='apple':raise ValueError("unsupported model")
            model=cls(Config(**meta['config']))
            model.mean,model.scale,model.coef,model.intercept=[np.array(data[k],np.float32) for k in ['mean','scale','coef','intercept']]
        d=FeatureExtractor.dimensions
        if model.mean.shape!=(d,) or model.scale.shape!=(d,) or model.coef.shape!=(3,d) or model.intercept.shape!=(3,):raise ValueError("invalid model array dimensions")
        if not all(np.isfinite(a).all() for a in [model.mean,model.scale,model.coef,model.intercept]) or np.any(model.scale<=0):raise ValueError("invalid model parameters")
        model.leaf_threshold,model.lesion_threshold=meta['leaf_threshold'],meta['lesion_threshold']
        if not 0<model.leaf_threshold<1 or not 0<model.lesion_threshold<1:raise ValueError("invalid thresholds")
        model.metadata=meta;return model


def save_prediction(rgb,summary,leaf,lesion,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.uint8(leaf)*255).save(output/'leaf_mask.png')
    Image.fromarray(np.uint8(lesion)*255).save(output/'lesion_mask.png')
    overlay=rgb.copy();overlay[leaf]=.75*overlay[leaf]+.25*np.array([.05,.85,.3])
    overlay[lesion]=.35*rgb[lesion]+.65*np.array([1,.1,.05])
    Image.fromarray(np.uint8(np.clip(overlay,0,1)*255)).save(output/'overlay.png')
    (output/'prediction.json').write_text(json.dumps(summary,indent=2)+'\n')
