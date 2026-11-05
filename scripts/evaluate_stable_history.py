"""One preregistered train/val-only repair investigation; never reads test labels."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import sys
from pathlib import Path
import json
import hashlib
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from falconworld.models import load_models
from falconworld.stable_history import StableHistoryModel,fit_history
from falconworld.planning import LongitudinalMPC,CEMConfig
from scripts.analyze_validation import evaluate_validation,aggregate,csv_write,COMPONENTS


def score(rows):
    selected=[r['recursive_gap_m'] for r in rows if r['horizon_s'] in (.5,1.,3.)]
    return float(np.mean(selected))


def main():
    run=ROOT/'outputs/standard_verified';out=ROOT/'outputs/stable_history_validation';out.mkdir(exist_ok=True)
    plan=dict(scope='single validation-failure repair investigation; train/val only; original test results preserved',
        ridge=[1e-6,1e-3,1.],root_radius=[.95,.999],adaptive_physical_gamma=[.8,.95,1.],
        selection='mean of equal-episode recursive gap MAE at .5/1/3 seconds on val; no test; no extra search',
        own_dynamics='train-identified enriched physical',leader='zero-intercept AR2 increment; CA plus correction',
        fits='one deterministic fit per hyperparameter on shared train; three existing NN seeds are descriptive repeats')
    registration=out/'registration.json'
    if registration.exists() and json.loads(registration.read_text())!=plan:raise ValueError('Registered budget changed')
    registration.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    data={split:dict(np.load(run/'data'/f'{split}.npz',allow_pickle=False)) for split in ('train','val')}
    original=load_models(run/'seed0/models');physics=original['physical_enriched']
    metric_names=[f'{mode}_{c}' for mode in ('recursive','teacher_forced_endpoint') for c in COMPONENTS]
    trials=[];candidates=[];start=perf_counter()
    for ridge in plan['ridge']:
        for radius in plan['root_radius']:
            model=fit_history(data['train'],physics,ridge,radius)
            rows=evaluate_validation({'candidate':model},data['val'],0)
            ep=aggregate(rows,('seed','episode','family','model','horizon_s'),metric_names)
            loss=score(ep)
            trials.append(dict(kind='learned_ar2',ridge=ridge,radius=radius,coefficients=model.coefficients.tolist(),val_gap_score_m=loss))
            candidates.append((loss,model))
    stable=min(candidates,key=lambda x:x[0])[1]
    adaptive=[]
    for gamma in plan['adaptive_physical_gamma']:
        model=StableHistoryModel(physics,[gamma,0.],name='adaptive_damped_ca')
        rows=evaluate_validation({'adaptive':model},data['val'],0)
        ep=aggregate(rows,('seed','episode','family','model','horizon_s'),metric_names)
        loss=score(ep);adaptive.append((loss,model))
        trials.append(dict(kind='adaptive_damped_ca',gamma=gamma,val_gap_score_m=loss))
    damped=min(adaptive,key=lambda x:x[0])[1]
    stable.save(out/'stable_history.npz');damped.save(out/'adaptive_damped_ca.npz')
    rows=[]
    for seed in (0,1,2):
        models=load_models(run/f'seed{seed}/models')
        compare={k:models[k] for k in ('physical_cv','physical_ca','physical_enriched','history','learned_lead')}
        compare.update(stable_history=stable,adaptive_damped_ca=damped)
        rows.extend(evaluate_validation(compare,data['val'],seed))
    ep=aggregate(rows,('seed','episode','family','model','horizon_s'),metric_names)
    summary=aggregate(ep,('family','model','horizon_s'),metric_names)
    csv_write(out/'episode_errors.csv',ep);csv_write(out/'by_family.csv',summary)
    # Controlled host profile, same candidate/horizon/iteration counts; includes whole optimization.
    profile=[];s=data['val']['state'][0];h=data['val']['history'][0]
    for name,model in dict(physical_enriched=physics,history=original['history'],stable_history=stable,adaptive_damped_ca=damped).items():
        planner=LongitudinalMPC(model,CEMConfig(),seed=55)
        for _ in range(2):planner.command(s,h)
        times=[];inference=[]
        for _ in range(10):
            before=perf_counter();_,info=planner.command(s,h);times.append((perf_counter()-before)*1000)
            inference.append(info['inference_s']*1000)
        profile.append(dict(model=name,candidates=64,horizon=30,iterations=3,full_cycle_p50_ms=float(np.median(times)),
            full_cycle_p95_ms=float(np.quantile(times,.95)),rollout_inference_p50_ms=float(np.median(inference)),scope='shared host load, validation state, no vehicle I/O'))
    csv_write(out/'profile.csv',profile)
    fingerprints={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (run/'data/train.npz',run/'data/val.npz',Path(__file__),ROOT/'falconworld/stable_history.py')}
    result=dict(trials=trials,selected_coefficients=stable.coefficients.tolist(),selected_adaptive_coefficients=damped.coefficients.tolist(),
        elapsed_s=perf_counter()-start,accessed_splits=['train','val'],file_sha256=fingerprints,
        baseline_scores={name:score([r for r in ep if r['model']==name]) for name in ('physical_enriched','history','stable_history','adaptive_damped_ca')})
    (out/'selection.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    def table(headers,records):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(str(c) for c in r)+'|' for r in records])
    baseline=result['baseline_scores'];gain=baseline['stable_history']-baseline['adaptive_damped_ca']
    conclusion=('验证集平均gap指标优于本次合理衰减CA对照，可以考虑一次预先登记的附加冻结测试；当前只有验证结果，不能宣称闭环收益。'
                if gain < -1e-5 else '验证集未优于合理衰减CA对照，本候选停止，不值得仅为追求正结果追加测试。')
    text=['# 一次基于验证失稳的修复研究', '',
        '原正式模型、控制器、测试与结果完整保留；本研究只加载train/val。预算在运行前写入 registration.json：6个稳定AR2超参数与3个衰减CA对照参数，按val的episode等权0.5/1/3秒平均gap MAE选择，不再扩展结构搜索。', '',
        '原失效：神经网络前车单步偏差被自身历史反馈复合，匀速场景产生非零漂移。修复假设：零截距令严格匀速成为固定点；AR2根投影到单位圆内限制速度增量的递归自激。自车仍使用train辨识强物理。稳定性仅涉及该线性速度增量递推，不是碰撞、位置误差或闭环安全保证；饱和、停车和未知转折仍有误差。', '',
        f"选中AR2系数 `{stable.coefficients.tolist()}`，选中衰减CA系数 `{damped.coefficients.tolist()}`。同一数据上的确定性拟合只有一次，不将复制到3个NN种子称作3个独立新拟合。平均val分数(m)：`{baseline}`，候选减衰减CA={gain:+.6g}m。", '',
        conclusion, '',
        '### 每族三秒递归误差（全部对照保留）', '',
        table(['族','方法','gap MAE m','ego MAE m/s','lead MAE m/s','actuator MAE m/s²'],
            [[r['family'],r['model'],*[f"{r['recursive_'+c]:.6g}" for c in COMPONENTS]] for r in summary if r['horizon_s']==3.]), '',
        '全部0.1/0.5/1/3秒结果见 by_family.csv，完整搜索记录见 selection.json。物理CA可以精确保持匀速；候选若仅靠更保守的前车趋势衰减获益，必须与同样使用val选择的衰减CA比较。这里没有未来前车脚本、隐藏场景标签或自车动作进入前车头。', '',
        '### 相同完整CEM工作量的计算代价', '',
        table(list(profile[0]),[list(r.values()) for r in profile]), '',
        '主机当前仍运行其他实验，计时包含64候选×30步×3迭代，实际争用可能改变deadline回退。该测量不是Pi端、独占硬实时或车辆I/O性能验收。原标准测试不会被替换；任何附加测试必须另行登记、冻结且一次执行，保持原失败结果可见。', '']
    (ROOT/'docs/STABLE_HISTORY_STUDY.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
