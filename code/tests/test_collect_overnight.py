"""Finite completion, matching, and transfer tests without SSH/GPU access."""
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE=Path(__file__).resolve().parents[1]/'cluster/collect_overnight_20261024.py'
spec=importlib.util.spec_from_file_location('collect_overnight',SOURCE)
collect=importlib.util.module_from_spec(spec);spec.loader.exec_module(collect)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def fixture(self,arm='exact',seed=42,qkn=True,host='perm'):
        rid=f'v27_50m_{arm}_r400k_dn20_tls1_s{seed}_e3'+('_qkn' if qkn else '')+'_fixed'
        cfg={k:0 for k in collect.CONFIG_KEYS+collect.TABLE_KEYS}
        cfg.update(seed=seed,max_steps=600,qk_norm=qkn,train_shards=[1,11],val_shards=[2],
                   train_max_tokens=1000000000,total_batch_size=147456,val_interval_steps=10,
                   val_batches=12,val_device_batch_size=24,data_seed=42,data_mode='fixed',
                   ngram_extra_orders=[2,3,4,5] if arm=='exact' else [])
        summary={k:0 for k in collect.SUMMARY_KEYS}
        summary.update(seed=seed,run_id=rid,config=cfg,steps=600,ngram_orders=[2,3,4,5] if arm=='exact' else [],
                       train_max_tokens=1000000000,final_train_loss=2.,final_val_loss=2.5 if arm=='exact' else 2.6)
        rows=[{'step':step,'train_loss':2.,'val_loss':summary['final_val_loss']} for step in range(10,601,10)]
        path=self.root/host/rid;path.mkdir(parents=True,exist_ok=True)
        (path/'summary.json').write_text(json.dumps(summary))
        (path/'train_log.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        rec={'host':host,'run_id':rid,'errors':[],'done_line':True,'rc':'0','files':{}}
        return rec,summary,rows,path

    def validate(self,fixture):return collect.validate_run(fixture[0],self.root)

    def test_correct_pair_has_60_common_points_and_sign(self):
        a=self.fixture();b=self.fixture('nogram')
        p=collect.compare_runs(self.validate(a),self.validate(b))
        self.assertEqual(p['state'],'validated_numeric_comparison')
        self.assertEqual(p['comparison'],'add_table_exact_minus_nogram')
        self.assertEqual(len(p['last_60_common_points']),60)
        self.assertAlmostEqual(p['final']['delta_val'],-.1)
        self.assertEqual(p['final']['treatment_val'],2.5)

    def test_missing_completion_and_wrong_steps_block(self):
        f=self.fixture();f[0]['done_line']=False
        self.assertFalse(self.validate(f)['eligible'])
        f[0]['done_line']=True;f[1]['steps']=590
        (f[3]/'summary.json').write_text(json.dumps(f[1]))
        self.assertIn('incomplete_or_wrong_step',self.validate(f)['issues'])
        (f[3]/'summary.json').unlink()
        self.assertFalse(self.validate(f)['eligible'])

    def test_different_seed_and_input_config_block(self):
        a=self.validate(self.fixture());b=self.validate(self.fixture('nogram',seed=43))
        self.assertEqual(collect.compare_runs(a,b)['state'],'manual_review')
        b=self.validate(self.fixture('nogram'));b['_summary']['config']['val_shards']=[3]
        self.assertIn('mismatch:config.val_shards',collect.compare_runs(a,b)['issues'])

    def test_nonfinite_anywhere_and_failed_rc_block(self):
        f=self.fixture();f[2][3]['val_loss']=float('nan')
        (f[3]/'train_log.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in f[2]))
        self.assertIn('nonfinite_or_missing_loss',self.validate(f)['issues'])
        f=self.fixture();f[0]['rc']='1'
        self.assertFalse(self.validate(f)['eligible'])
        f=self.fixture();f[1]['final_val_loss']=float('nan')
        (f[3]/'summary.json').write_text(json.dumps(f[1]))
        result=self.validate(f)
        self.assertFalse(result['eligible']);self.assertIsNone(result['final_val'])
        json.dumps({k:v for k,v in result.items() if not k.startswith('_')},allow_nan=False)

    def test_same_arm_qkn_intervention_is_not_table_effect(self):
        a=self.validate(self.fixture('nogram',qkn=True))
        b=self.validate(self.fixture('nogram',qkn=False))
        p=collect.compare_runs(a,b)
        self.assertEqual(p['comparison'],'qkn_intervention_same_arm_not_add_table')
        mixed=self.validate(self.fixture('exact',qkn=True))
        self.assertEqual(collect.compare_runs(mixed,b)['state'],'manual_review')

    def test_cross_host_and_unknown_config_block(self):
        a=self.validate(self.fixture());b=self.validate(self.fixture('nogram',host='a100_t1'))
        self.assertIn('cross_host',collect.compare_runs(a,b)['issues'])
        b=self.validate(self.fixture('nogram'));del b['_summary']['config']['data_seed']
        self.assertIn('unknown:config.data_seed',collect.compare_runs(a,b)['issues'])

    def test_transfer_checksum_verifies_bytes_and_rejects_corruption(self):
        rec=self.fixture()[0];data=b'{}'
        rec['files']={'summary.json':{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                                     'base64':base64.b64encode(data).decode()}}
        bad=copy.deepcopy(rec);bad['files']['summary.json']['sha256']='0'*64
        with self.assertRaises(ValueError):collect.persist_run(bad,self.root)
        out=collect.persist_run(rec,self.root)
        self.assertEqual(out['files']['summary.json']['sha256'],hashlib.sha256(data).hexdigest())

    def test_select_only_terminal_jobs_and_direct_controls(self):
        r1='v27_50m_exact_r400k_dn20_tls1_s42_e3_qkn_fixed'
        r2=r1.replace('_exact_','_nogram_')
        manifest={'jobs':[{'host':'perm','run_id':r1,'control_refs':[{'host':'perm','run_id':r2}]},
                          {'host':'perm','run_id':r2,'control_refs':[]}]}
        status={'hosts':[{'host':'perm','jobs':[{'run_id':r1,'rc':'1','key':'a'*12},
                                               {'run_id':r2,'rc':None,'key':'b'*12}]}]}
        selected,edges=collect.select_requests(manifest,status)
        self.assertEqual(len(selected),2);self.assertEqual(len(edges),1)
        status['hosts'][0]['jobs'][0]['rc']=None
        self.assertEqual(collect.select_requests(manifest,status),({},[]))

    def test_forbidden_control_is_blocked_without_losing_allowed_run(self):
        rid='v27_50m_exact_r400k_dn20_tls1_s42_e3_qkn_fixed'
        other=rid.replace('_exact_','_nogram_')
        manifest={'jobs':[{'host':'perm','run_id':rid,'control_refs':[
            {'host':'a100_t1_3','run_id':other}]}]}
        status={'hosts':[{'host':'perm','jobs':[{'run_id':rid,'rc':'0','key':'a'*12}]}]}
        selected,edges=collect.select_requests(manifest,status)
        self.assertEqual(selected[('a100_t1_3',other)]['blocked'],'host_not_allowlisted_no_access')
        self.assertNotIn('blocked',selected[('perm',rid)])
        self.assertEqual(len(edges),1)

    def test_registered_budget_blocks_self_consistent_short_run(self):
        f=self.fixture();s=f[1];c=s['config']
        c.update(train_max_tokens=600,sequence_len=1,device_batch_size=1,total_batch_size=1)
        s.update(train_max_tokens=600,train_epochs=1,train_pool_content_tokens=600)
        (f[3]/'summary.json').write_text(json.dumps(s))
        job=dict(seed=42,epochs=1,nominal_tokens=600,qkn=True,lr=0,arm='exact',tls=0)
        self.assertTrue(collect.validate_run(f[0],self.root,job)['eligible'])
        c['max_steps']=300;s['steps']=300
        (f[3]/'summary.json').write_text(json.dumps(s))
        (f[3]/'train_log.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in f[2][:30]))
        self.assertTrue(collect.validate_run(f[0],self.root)['eligible'])
        checked=collect.validate_run(f[0],self.root,job)
        self.assertIn('registered_step_budget_mismatch',checked['issues'])
        job.update(epochs=3,nominal_tokens=1800,lr=.0006,qkn=False,tls=2)
        checked=collect.validate_run(f[0],self.root,job)
        self.assertIn('registered_mismatch:train_epochs',checked['issues'])
        self.assertIn('registered_config_mismatch:qk_norm',checked['issues'])


if __name__=='__main__':unittest.main()
