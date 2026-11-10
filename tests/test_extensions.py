import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from PIL import Image
import torch
from dr.images import prepare
from dr.training import accumulation_window
from dr.conditioning import load_profile, sample_condition
from dr.runtime import precision, protect
from dr.audit import audit
from dr.data import Photos
from dr.networks import build, restore
from dr.representations import paired_distances, ijepa_encoder


class ExtensionTests(unittest.TestCase):
    def test_cache_is_pixel_exact_and_size_specific(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'x.png'
            Image.fromarray(np.random.default_rng(3).integers(15,255,(45,60,3),dtype=np.uint8)).save(source)
            a=np.array(prepare(source,32)); b=np.array(prepare(source,32,root/'cache'))
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(b,np.array(prepare(source,32,root/'cache')))
            self.assertEqual(prepare(source,64,root/'cache').size,(64,64))

    def test_cpu_precision_and_output_protection(self):
        self.assertEqual(precision(torch.device('cpu'),'auto'),'fp32')
        with self.assertRaises(ValueError): precision(torch.device('cpu'),'fp16')
        with tempfile.NamedTemporaryFile() as f:
            with self.assertRaises(FileExistsError): protect(f.name)
            protect(f.name,True)

    def test_conditioning_blocks_nontraining_donors_and_excluded_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'profile.json'
            profile={'version':1,'donors':[{'patient':'1','split':'train','provenance':'natural_manual',
                        'spec':[{'operator':'defocus','level':3}]}]}
            p.write_text(json.dumps(profile))
            self.assertEqual(sample_condition(load_profile(p,{'1'}),np.random.default_rng(1)),[('defocus',3)])
            with self.assertRaises(ValueError): load_profile(p,{'2'})
            with self.assertRaises(ValueError): sample_condition(profile,np.random.default_rng(1),'defocus')
            profile['donors'][0]['split']='test'; p.write_text(json.dumps(profile))
            with self.assertRaises(ValueError): load_profile(p,{'1'})

    def test_audit_detects_duplicate_pixels_and_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('1_left.png','2_left.png'):
                Image.new('RGB',(32,32),(90,20,50)).save(root/name)
            csv=root/'labels.csv'
            pd.DataFrame({'image':['1_left.png','2_left.png','3_left.png'],'DR_grade':[0,1,2],'quality':[0,1,2]}).to_csv(csv,index=False)
            report=audit([csv],root)
            self.assertEqual(report['status'],'failed')
            self.assertEqual(report['missing'],['3_left.png'])
            self.assertEqual(len(report['duplicate_content']),1)

    def test_r18_grader_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(11); net=build('grade','r18',32).eval()
            x=torch.randn(2,3,32,32)
            with torch.no_grad(): expected=net(x)
            path=Path(tmp)/'model.pt'
            torch.save(dict(state=net.state_dict(),task='grade',arch='r18',size=32),path)
            restored,_=restore(path,'cpu')
            with torch.no_grad(): actual=restored(x)
            self.assertEqual(actual[0].shape,(2,5)); self.assertEqual(actual[1].shape,(2,1))
            torch.testing.assert_close(actual[0],expected[0])

    def test_pair_metrics_do_not_invent_natural_clean_references(self):
        rows=pd.DataFrame([['a','1','original','test',0],['a','1','defocus-1','test',0],
                           ['b','2','original','test',2]],columns=['image','patient','variant','split','quality'])
        table=paired_distances(rows,np.array([[1.,0.],[0.,1.],[2.,3.]]))
        self.assertEqual(len(table),1); self.assertAlmostEqual(table.cosine_distance.iloc[0],1.)

    def test_small_ijepa_equivalent_structure_pools_normalized_tokens(self):
        net=ijepa_encoder(size=28,width=32,depth=2,heads=4,patch=14).eval()
        with torch.no_grad():
            x=torch.randn(2,3,28,28)
            torch.testing.assert_close(net(x),net.forward_features(x).mean(1))
        self.assertNotIn('cls_token',net.state_dict())
        self.assertEqual(net.pos_embed.shape,(1,4,32))

    def test_conditioned_photos_can_use_natural_donor_outside_good_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); image=root/'1_left.png'; Image.new('RGB',(40,40),(100,60,30)).save(image)
            profile=root/'p.json'; profile.write_text(json.dumps({'version':1,'donors':[dict(patient='2',split='train',
                provenance='natural_estimated',spec=[dict(operator='dark',level=3)])]}))
            frame=pd.DataFrame([dict(image=image.name,path=str(image),patient='1',grade=2,quality=0,split='train')])
            data=Photos(frame,32,42,'balanced',condition_profile=profile,donor_patients=['1','2'],corruption_size=64)
            self.assertEqual(data[1]['variant'],'dark-3'); self.assertEqual(data[1]['x'].shape,(3,32,32))

    def test_accumulation_tail_matches_mean_microbatch_objective(self):
        # Explicit legacy objective: mean of each microbatch's mean loss, including short final batches.
        xs=[torch.tensor([[1.],[2.]]),torch.tensor([[3.],[4.]]),torch.tensor([[5.]])]
        a=torch.nn.Linear(1,1,bias=False); b=torch.nn.Linear(1,1,bias=False); b.load_state_dict(a.state_dict())
        oa=torch.optim.SGD(a.parameters(),lr=.01); ob=torch.optim.SGD(b.parameters(),lr=.01)
        for step,x in enumerate(xs):
            window=accumulation_window(step,len(xs),2)
            (a(x).square().mean()/window).backward()
            if (step+1)%2==0 or step+1==len(xs): oa.step(); oa.zero_grad(set_to_none=True)
        for chunk in (xs[:2],xs[2:]):
            sum(b(x).square().mean()/len(chunk) for x in chunk).backward(); ob.step(); ob.zero_grad(set_to_none=True)
        torch.testing.assert_close(a.weight,b.weight)


class FurtherExtensionTests(unittest.TestCase):
    def test_mobile_backbones_use_correct_feature_dimensions(self):
        for arch in ('mobilenetv3','efficientnetb0'):
            net=build('grade',arch,32).eval()
            with torch.no_grad(): grade,ref=net(torch.randn(2,3,32,32))
            self.assertEqual(grade.shape,(2,5)); self.assertEqual(ref.shape,(2,1))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware unavailable')
    def test_cuda_fp16_has_finite_scaled_update(self):
        from dr.runtime import autocast
        model=torch.nn.Linear(8,3).cuda()
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        scaler=torch.amp.GradScaler('cuda')
        with autocast(torch.device('cuda'),'fp16'):
            loss=torch.nn.functional.cross_entropy(model(torch.randn(4,8,device='cuda')),torch.tensor([0,1,2,1],device='cuda'))
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        self.assertTrue(torch.isfinite(model.weight).all())

    def test_reference_resolution_leaves_originals_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            image=Path(tmp)/'1_left.png'
            Image.fromarray(np.random.default_rng(8).integers(15,255,(45,60,3),dtype=np.uint8)).save(image)
            frame=pd.DataFrame([dict(image=image.name,path=str(image),patient='1',grade=1,quality=0,split='test')])
            a=Photos(frame,32,42,'natural')[0]['x']
            b=Photos(frame,32,42,'natural',corruption_size=64)[0]['x']
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_semantic_reference_has_no_pair_requirement(self):
        from dr.semantic_features import reference_scores
        reference=np.array([[1.,0.],[2.,0.]])
        scores=reference_scores(reference,np.array([[1.,0.],[0.,1.]]))
        self.assertAlmostEqual(scores[0,0],0.)
        self.assertAlmostEqual(scores[1,0],1.)
        self.assertTrue(np.isfinite(scores).all())


if __name__=="__main__": unittest.main()
