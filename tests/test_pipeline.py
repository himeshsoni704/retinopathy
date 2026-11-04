import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from dr.data import Photos, read_manifest
from dr.evaluation import augrc, metrics
from dr.images import FAMILY_OF, degrade
from dr.policy import errors


class PipelineTests(unittest.TestCase):
    def test_all_corruptions_preserve_black_border_and_reproduce(self):
        arr = np.zeros((48,48,3),dtype=np.uint8)
        arr[4:44,4:44] = np.random.default_rng(1).integers(32,200,size=(40,40,3))
        image = Image.fromarray(arr)
        for op in FAMILY_OF:
            for level in range(1,6):
                a = np.array(degrade(image,op,level,np.random.default_rng(10)))
                b = np.array(degrade(image,op,level,np.random.default_rng(10)))
                np.testing.assert_array_equal(a,b)
                np.testing.assert_array_equal(a[arr.max(2)==0],0)
                self.assertTrue((a != arr).any(),(op,level))

    def test_suite_has_50_variants_while_natural_poor_quality_is_untouched(self):
        frame = pd.DataFrame({"quality":[0,2],"grade":[0,4]})
        self.assertEqual(len(Photos(frame,384,42,"suite")),52)
        self.assertEqual(len(Photos(frame,384,42,"balanced")),2)
        self.assertEqual(len(Photos(frame,384,42,"balanced-real")),3)

    def test_patient_leakage_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"manifest.csv"
            pd.DataFrame({"image":["a","b"],"patient":["p","p"],"grade":[0,1],
                          "quality":[0,0],"split":["train","test"],"path":["a","b"]}).to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,"crosses partitions"):
                read_manifest(p)

    def test_augrc_rewards_correct_first_and_averages_ties(self):
        y = np.array([0,1])
        self.assertAlmostEqual(augrc(y,[0.,1.]),.125)
        self.assertAlmostEqual(augrc(y,[1.,0.]),.375)
        self.assertAlmostEqual(augrc(y,[1.,1.]),.25)

    def test_referral_denominators_and_ungradable(self):
        frame = pd.DataFrame({"grade":[3,2,4,0,-1],"prediction":[0,3,0,0,0],
                              "ref_probability":[.1,.9,.2,.1,.2]})
        a = np.array([True,True,False,True,True])
        self.assertEqual(errors(frame,"grade").tolist()[:2],[True,True])
        m = metrics(frame,a,"referral")
        self.assertAlmostEqual(m["ref_sensitivity_accepted"],.5)
        self.assertAlmostEqual(m["missed_referable_rate"],1/3)
        self.assertAlmostEqual(m["silent_failure"],.25)
        self.assertAlmostEqual(m["selective_risk"],1/3)
        self.assertEqual(m["unknown_grade_accepted"],1)
