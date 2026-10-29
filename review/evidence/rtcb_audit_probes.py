"""Synthetic audit probes; repository sources are unchanged. Run with audit paths."""
import sys
import json
import importlib.util
import runpy
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd

BASE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path('/private/tmp/rtcb-audit-20261026')
sys.path[:0] = [str(BASE/'deps'), str(BASE/'ercot-rtcb-bench/src'), str(BASE/'rtcb-gym'), str(BASE/'hybridbid')]
from ercot_rtcb_bench.methods.battery import BESSParams
from ercot_rtcb_bench.methods.perfect_foresight import perfect_foresight
from ercot_rtcb_bench.forecaster.analog import match_analogs
from rtcb_gym.battery import BatteryParams, battery_step
from rtcb_gym.baselines.optimize import solve_dispatch

out = {}
cols = ['lmp','mcpc_regup','mcpc_regdn','mcpc_rrs','mcpc_ecrs','mcpc_nspin']
idx = pd.date_range('2026-02-01', periods=1, freq='5min', tz='UTC')
prices = pd.DataFrame([[-100.,0.,0.,0.,0.,0.]], index=idx, columns=cols)
b = BESSParams('AUDIT',10.,20.,.85,soc_init=1.)
r = perfect_foresight(b, prices)
out['bench_negative_price_full_battery_one_step'] = r.dispatch.iloc[0].to_dict()
out['bench_negative_price_full_battery_one_step']['lp_revenue'] = r.revenue_total
out['bench_negative_price_full_battery_one_step']['physical_single_net_power_optimum'] = 0.
ours = solve_dispatch(np.array([-100.]), BatteryParams(power_mw=10.,energy_mwh=20.,rte=.85,soc_init=1.,degradation_cost_usd_per_mwh=0.), 1., 5/60)
out['rtcb_gym_same_case'] = dict(power_mw=ours.power_mw.tolist(),revenue=ours.revenue_usd,used_milp=ours.used_milp)

prices2 = pd.DataFrame([[1.,1000.,0.,0.,0.,0.]], index=idx, columns=cols)
r2 = perfect_foresight(BESSParams('AUDIT',10.,20.,.85,soc_init=0.), prices2)
out['bench_empty_battery_regup'] = r2.dispatch.iloc[0].to_dict()

target, d1, d2 = date(2026,1,12),date(2026,1,10),date(2026,1,11)
a = np.r_[np.zeros(144),np.linspace(0.,1.,144)]
c = np.r_[np.zeros(144),-np.linspace(0.,1.,144)]
class FakeDataset:
    def __init__(self, profile): self.profile = profile
    def get_net_load(self,d): return self.profile if d==target else (a if d==d1 else c)
    def available_days(self,start,end): return [d1,d2]
out['bench_future_actual_net_load_changes_analog'] = {
    'unchanged_first_144_intervals':bool(np.array_equal(a[:144],c[:144])),
    'selection_A':[str(x) for x in match_analogs(target,FakeDataset(a),n=1)[0]],
    'selection_B':[str(x) for x in match_analogs(target,FakeDataset(c),n=1)[0]],
}

spec=importlib.util.spec_from_file_location('audit_hybrid_eval',BASE/'hybridbid/src/evaluation/eval_t60.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
p=np.zeros((40,12),dtype=np.float32);p[31,0]=1234.
s=np.zeros((40,7),dtype=np.float32)
ts=pd.date_range('2026-01-01',periods=40,freq='5min',tz='US/Central')
obs=h._build_obs(p,s,ts,31,10.)
out['hybrid_current_price_visible_before_action']={'idx':31,'settlement_rt_lmp':float(p[31,0]),'last_observed_rt_lmp':float(obs['price_history'][-1,0])}
out['hybrid_ecrs_projection']={
    'soc_mwh':4.,'soc_min_mwh':h.SOC_MIN,
    'requested_ecrs_mw':10.,
    'accepted_ecrs_mw':float(h.project_action(np.array([0.,0.,0.,0.,10.,0.]),4.)[4]),
    'configured_duration_h':float(h.AS_SUSTAIN_H[3]),
    'same_formula_at_1h_mw':(4.-h.SOC_MIN)*h.ETA,
}
try:
    runpy.run_path(str(BASE/'hybridbid/src/methods/bc/run_bc.py'),run_name='audit_import_only')
    out['hybrid_bc_entry_import']='ok'
except Exception as e:
    out['hybrid_bc_entry_import']={'type':type(e).__name__,'error':str(e)}

from src.models.feasibility import project_co_optimize
import torch
train_proj=project_co_optimize(torch.tensor([0.,0.,0.,0.,10.,0.]),torch.tensor(4.),10.,20.,eta_ch=.95,eta_dch=.95)
out['hybrid_train_vs_eval_ecrs_same_input']={'train_projection':train_proj.tolist(),'eval_projection':h.project_action(np.array([0.,0.,0.,0.,10.,0.]),4.).tolist()}
print(json.dumps(out,indent=2,default=str))
(BASE/'probe_results.json').write_text(json.dumps(out,indent=2,default=str)+'\n')
