from pathlib import Path
import csv,json,platform,time
import numpy as np
from PIL import Image
from .core import SeverityModel,Config,rgb_image,label_image,dice,summarize_masks,save_prediction
from .data import read_manifest


def iou(a,b):
    union=np.logical_or(a,b).sum()
    return 1.0 if union==0 else float(np.logical_and(a,b).sum()/union)


def evaluate(manifest,output,seed=18):
    payload,train,validation,test=read_manifest(manifest)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    model=SeverityModel(Config(seed=seed))
    t0=time.perf_counter();model.fit(train).calibrate(validation);fit_seconds=time.perf_counter()-t0
    model.save(output/'model.npz')
    report={'dataset_kind':payload['kind'],'crop':'apple','description':payload.get('description'),
            'training_images':len(train),'validation_images':len(validation),'test_images':len(test),
            'fit_and_validation_seconds':fit_seconds,'seed':seed,'input_size':model.config.size,
            'sampled_training_pixels':model.metadata['sampled_training_pixels'],
            'head_iterations':model.metadata['head_iterations'],
            'head_converged_before_iteration_limit':model.metadata['head_converged_before_iteration_limit'],
            'leaf_threshold':model.leaf_threshold,'lesion_threshold':model.lesion_threshold,
            'environment':{'python':platform.python_version(),**model.extractor.versions},
            'notes':['Frozen ImageNet-pretrained backbone; supervised linear pixel head fitted to annotated training images',
                     'Threshold grid selected using validation annotations only; test masks are evaluation only',
                     'Affected area is lesion pixels divided by total visible leaf pixels, including lesion pixels',
                     'All segmentation metrics are macro averages across individual images at native annotation resolution',
                     'No disease-type diagnosis, treatment recommendation or pathological severity grade',
                     'No real longitudinal data were collected; history demonstration is synthetic'],
            'results':[]}
    for condition,factor in [('original',1.0),('brightness_0.7',.7),('brightness_1.3',1.3)]:
        rows=[];times=[]
        for idx,record in enumerate(test):
            rgb=np.clip(rgb_image(record['image'])*factor,0,1);gt=label_image(record['mask'])
            t0=time.perf_counter();summary,leaf,lesion=model.predict(rgb);times.append(time.perf_counter()-t0)
            reference=summarize_masks(gt>0,gt==2)
            truth=reference['affected_area_percent'];pred=summary['affected_area_percent']
            row={'condition':condition,'image':record['image'].name,'category':record.get('category','unknown'),
                 'leaf_dice':dice(leaf,gt>0),'lesion_dice':dice(lesion,gt==2),'lesion_iou':iou(lesion,gt==2),
                 'truth_severity_percent':truth,'predicted_severity_percent':pred,
                 'severity_absolute_error_pp':None if pred is None or truth is None else abs(pred-truth),
                 'ground_truth_lesion_present':bool((gt==2).any()),'quality_warnings':';'.join(summary['quality_warnings'])}
            rows.append(row)
            if idx<4 and condition=='original' and payload['kind'].startswith('synthetic'):
                save_prediction(rgb,summary,leaf,lesion,output/f'example_{idx:02}')
                Image.fromarray(np.uint8(rgb*255)).save(output/f'example_{idx:02}'/'input.png')
        defined=[r['severity_absolute_error_pp'] for r in rows if r['severity_absolute_error_pp'] is not None]
        diseased=[r for r in rows if r['ground_truth_lesion_present']]
        diseased_errors=[r['severity_absolute_error_pp'] for r in diseased if r['severity_absolute_error_pp'] is not None]
        healthy_predictions=[r['predicted_severity_percent'] for r in rows if not r['ground_truth_lesion_present'] and r['predicted_severity_percent'] is not None]
        entry={'condition':condition,'mean_leaf_dice':float(np.mean([r['leaf_dice'] for r in rows])),
               'mean_lesion_dice':float(np.mean([r['lesion_dice'] for r in rows])),
               'mean_lesion_iou':float(np.mean([r['lesion_iou'] for r in rows])),
               'diseased_only_mean_lesion_dice':float(np.mean([r['lesion_dice'] for r in diseased])) if diseased else None,
               'severity_mae_percentage_points':float(np.mean(defined)) if defined else None,
               'diseased_only_severity_mae_percentage_points':float(np.mean(diseased_errors)) if diseased_errors else None,
               'healthy_mean_false_affected_area_percent':float(np.mean(healthy_predictions)) if healthy_predictions else None,
               'undefined_severity_count':len(rows)-len(defined),'median_inference_seconds':float(np.median(times))}
        report['results'].append(entry)
        with (output/f'predictions_{condition}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print(json.dumps(entry),flush=True)
    (output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
