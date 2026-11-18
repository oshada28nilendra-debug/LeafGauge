"""Local observation history; only compare measurements with compatible methods."""
from datetime import date
from pathlib import Path
import json
import os
import tempfile


def append_observation(store,leaf_id,view_id,observed_at,model_id,summary):
    if not leaf_id.strip() or not view_id.strip():raise ValueError("leaf and view IDs are required")
    try:date.fromisoformat(observed_at)
    except ValueError:raise ValueError("observed_at must be YYYY-MM-DD")
    if summary.get('crop')!='apple':raise ValueError("only apple observations are supported")
    severity=summary.get('affected_area_percent')
    if severity is not None and not 0<=severity<=100:raise ValueError("severity must be within [0,100]")
    store=Path(store)
    payload=json.loads(store.read_text()) if store.exists() else {'schema':1,'crop':'apple','observations':[]}
    if payload.get('schema')!=1 or payload.get('crop')!='apple':raise ValueError("incompatible observation store")
    for item in payload['observations']:
        if (item['leaf_id'],item['view_id'],item['observed_at'])==(leaf_id,view_id,observed_at):
            raise ValueError("observation already exists for this leaf, view and date")
    record={'leaf_id':leaf_id,'view_id':view_id,'observed_at':observed_at,'model_id':model_id,
            'affected_area_percent':severity,'manual_leaf_mask':summary.get('manual_leaf_mask',False),
            'quality_warnings':summary.get('quality_warnings',[])}
    payload['observations'].append(record)
    payload['observations'].sort(key=lambda r:(r['leaf_id'],r['view_id'],r['observed_at']))
    store.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=store.parent,delete=False,encoding='utf-8') as f:
        json.dump(payload,f,indent=2);f.write('\n');temp=f.name
    os.replace(temp,store)
    return history(store,leaf_id,view_id)


def history(store,leaf_id=None,view_id=None):
    payload=json.loads(Path(store).read_text())
    if payload.get('schema')!=1:raise ValueError("invalid observation store")
    rows=[r.copy() for r in payload['observations'] if (leaf_id is None or r['leaf_id']==leaf_id) and (view_id is None or r['view_id']==view_id)]
    previous={}
    for row in sorted(rows,key=lambda r:r['observed_at']):
        key=(row['leaf_id'],row['view_id']);old=previous.get(key)
        compatible=(old is not None and old['model_id']==row['model_id']
                    and old['manual_leaf_mask']==row['manual_leaf_mask']
                    and old['affected_area_percent'] is not None and row['affected_area_percent'] is not None
                    and not old['quality_warnings'] and not row['quality_warnings'])
        row['delta_percentage_points']=row['affected_area_percent']-old['affected_area_percent'] if compatible else None
        row['comparison_status']='comparable_measurements' if compatible else 'first_observation_or_review_required'
        previous[key]=row
    return sorted(rows,key=lambda r:(r['leaf_id'],r['view_id'],r['observed_at']))
