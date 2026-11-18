import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from leafseverity.core import Config,SeverityModel,rgb_image,label_image,summarize_masks,dice
from leafseverity.data import synthetic,read_manifest
from leafseverity.tracking import append_observation,history


class LeafSeverityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.manifest=synthetic(cls.root/'data',18)
        _,cls.train,cls.val,cls.test=read_manifest(cls.manifest)
        cls.model=SeverityModel(Config(size=128,pixels_per_class_per_image=60)).fit(cls.train[:8]).calibrate(cls.val[:4])

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_severity_denominator_includes_lesions(self):
        leaf=np.ones((4,4),bool);lesion=np.zeros((4,4),bool);lesion[:2,:2]=True
        result=summarize_masks(leaf,lesion)
        self.assertEqual(result['affected_area_percent'],25)
        self.assertEqual(result['leaf_pixels'],16)

    def test_empty_leaf_is_undefined_not_healthy(self):
        result=summarize_masks(np.zeros((4,4),bool),np.zeros((4,4),bool))
        self.assertIsNone(result['affected_area_percent'])

    def test_lesions_outside_leaf_rejected(self):
        with self.assertRaises(ValueError):summarize_masks(np.zeros((4,4),bool),np.ones((4,4),bool))

    def test_detects_synthetic_leaf_and_lesions(self):
        record=next(r for r in self.test if (label_image(r['mask'])==2).any())
        summary,leaf,lesion=self.model.predict(rgb_image(record['image']))
        truth=label_image(record['mask'])
        self.assertGreater(dice(leaf,truth>0),.8)
        self.assertGreater(dice(lesion,truth==2),.5)
        self.assertIsNotNone(summary['affected_area_percent'])

    def test_native_resolution_and_aspect_ratio_restored(self):
        rgb=rgb_image(self.test[0]['image'])[:,:130]
        _,leaf,lesion=self.model.predict(rgb)
        self.assertEqual(leaf.shape,(224,130));self.assertEqual(lesion.shape,leaf.shape)
        self.assertFalse((lesion&~leaf).any())

    def test_manual_leaf_mask_controls_denominator(self):
        rgb=rgb_image(self.test[0]['image']);mask=np.zeros(rgb.shape[:2],bool);mask[60:160,70:150]=True
        summary,leaf,lesion=self.model.predict(rgb,mask)
        self.assertEqual(summary['leaf_pixels'],8000)
        np.testing.assert_array_equal(leaf,mask)
        self.assertFalse((lesion&~mask).any())
        self.assertTrue(summary['manual_leaf_mask'])

    def test_save_load_preserves_masks(self):
        path=self.root/'model.npz';self.model.save(path);loaded=SeverityModel.load(path)
        rgb=rgb_image(self.test[0]['image'])
        a,al,ad=self.model.predict(rgb);b,bl,bd=loaded.predict(rgb)
        np.testing.assert_array_equal(al,bl);np.testing.assert_array_equal(ad,bd)
        self.assertEqual(a['affected_area_percent'],b['affected_area_percent'])

    def test_validation_training_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError,'overlap'):self.model.calibrate(self.train[:1])

    def test_manifest_duplicate_or_group_leakage_rejected(self):
        payload=json.loads(self.manifest.read_text());payload['validation'][0]=payload['train'][0]
        path=self.manifest.with_name('leaky.json');path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError,'leakage'):read_manifest(path)

    def test_tracker_computes_percentage_points_not_percent_growth(self):
        store=self.root/'trend.json';s={'crop':'apple','affected_area_percent':10,'manual_leaf_mask':False,'quality_warnings':[]}
        append_observation(store,'leaf-1','front','2026-11-18','model-1',s)
        s={**s,'affected_area_percent':13}
        rows=append_observation(store,'leaf-1','front','2026-11-19','model-1',s)
        self.assertEqual(rows[-1]['delta_percentage_points'],3)

    def test_tracker_refuses_duplicate_observation(self):
        store=self.root/'duplicate.json';s={'crop':'apple','affected_area_percent':10,'quality_warnings':[]}
        append_observation(store,'leaf-1','front','2026-11-18','model-1',s)
        with self.assertRaisesRegex(ValueError,'already exists'):append_observation(store,'leaf-1','front','2026-11-18','model-1',s)

    def test_tracker_does_not_compare_changed_model_or_view(self):
        store=self.root/'changed-model.json';s={'crop':'apple','affected_area_percent':10,'quality_warnings':[]}
        append_observation(store,'leaf-1','front','2026-11-18','model-1',s)
        rows=append_observation(store,'leaf-1','front','2026-11-19','model-2',s)
        self.assertIsNone(rows[-1]['delta_percentage_points'])
        rows=append_observation(store,'leaf-1','back','2026-11-20','model-2',s)
        self.assertIsNone(rows[-1]['delta_percentage_points'])

    def test_tracker_withholds_comparison_for_quality_warning(self):
        store=self.root/'quality.json';s={'crop':'apple','affected_area_percent':10,'quality_warnings':[]}
        append_observation(store,'leaf-1','front','2026-11-18','model-1',s)
        rows=append_observation(store,'leaf-1','front','2026-11-19','model-1',{**s,'quality_warnings':['no_leaf_detected']})
        self.assertIsNone(rows[-1]['delta_percentage_points'])

    def test_wrong_crop_manifest_rejected(self):
        payload=json.loads(self.manifest.read_text());payload['crop']='tomato'
        path=self.manifest.with_name('tomato.json');path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError,'apple only'):read_manifest(path)


if __name__=='__main__':unittest.main()
