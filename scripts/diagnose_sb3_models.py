"""Current SB3 world-model pool: frozen validation mechanisms, not test selection."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from falconworld.models import load_models,_actuator_features
from falconworld.rl_env import encode
from scripts.analyze_validation import evaluate_validation,aggregate,COMPONENTS,rollout,csv_write
from scripts.audit_ppo import raw_reward

NAMES=('physical_enriched','history','residual','learned_lead')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'outputs/rl_sb3/standard')
    args=parser.parse_args();run=args.run.resolve();out=run/'model_diagnosis';out.mkdir(exist_ok=True)
    common=json.loads((run/'common_data/manifest.json').read_text())
    valfile=run/'common_data/val.npz'
    if digest(valfile)!=common['files']['val']:raise ValueError('Frozen validation NPZ hash mismatch')
    val=dict(np.load(valfile,allow_pickle=False))
    windows=[];components=[];comparisons=[];h1rows=[];provenance={}
    metrics=[f'{mode}_{c}' for mode in ('recursive','teacher_forced_endpoint') for c in COMPONENTS]
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    registration=json.loads((run/'registration.json').read_text())
    # Require the actual inference implementation used by the frozen run.
    for relative,expected in registration['sources'].items():
        if Path(relative.replace('\\','/')).name in ('models.py','rl_env.py') and digest(ROOT/Path(relative.replace('\\','/')))!=expected:
            raise ValueError('Current inference/observation source differs from frozen run')
    for seed in registration['spec']['seeds']:
        folder=run/f'seed{seed}/world_models';modelhash=json.loads((folder/'frozen_sha256.json').read_text())
        for name in NAMES:
            if digest(folder/f'{name}.npz')!=modelhash[f'{name}.npz']:raise ValueError('Model artifact hash mismatch')
        training=json.loads((folder/'training.json').read_text())
        if training['data_sha256']!=common['sha256']:raise ValueError('Model/data provenance mismatch')
        models={name:model for name,model in load_models(folder).items() if name in NAMES}
        windows.extend(evaluate_validation(models,val,seed))
        physics=models['physical_enriched'];s,u,h=val['state'],val['action'],val['history']
        for name in ('history','residual'):
            raw=models[name].actuator_net(_actuator_features(s,u))[:,0]
            effective=models[name].step(s,u,h)[:,3]-physics.actuator(s,u)
            components.append(dict(seed=seed,model=name,validation_rows=len(s),
                raw_actuator_residual_mae_mps2=float(abs(raw).mean()),raw_actuator_residual_max_mps2=float(abs(raw).max()),
                effective_next_actuator_delta_mae_mps2=float(abs(effective).mean()),effective_next_actuator_delta_max_mps2=float(abs(effective).max())))
        # Repeat exactly the registered nonoverlapping 30-step window roots.
        indices=[]
        for episode in np.unique(val['episode']):
            rows=np.flatnonzero(val['episode']==episode)
            for offset in range(0,len(rows)-29,30):indices.append(rows[offset:offset+30])
        ix=np.asarray(indices)
        history_prediction=rollout(models['history'],s[ix[:,0]],u[ix].T,h[ix[:,0]])
        lead_prediction=rollout(models['learned_lead'],s[ix[:,0]],u[ix].T,h[ix[:,0]])
        for horizon in (1,5,10,30):
            difference=abs(history_prediction[horizon-1]-lead_prediction[horizon-1])
            comparisons.append(dict(seed=seed,horizon_s=horizon*.1,windows=len(ix),
                **{f'max_prediction_difference_{c}':float(difference[:,j].max()) for j,c in enumerate(COMPONENTS)}))
        # Same roots and actual applied actions. Previous action is reconstructed
        # only from the preceding row of the same val episode, never another split.
        previous=np.zeros(len(u))
        for episode in np.unique(val['episode']):
            rows=np.flatnonzero(val['episode']==episode);previous[rows[1:]]=u[rows[:-1]]
        predicted={name:model.step(s,u,h) for name,model in models.items()}
        baseline=predicted['physical_enriched']
        baseline_reward=raw_reward(baseline[:,0],baseline[:,1],previous,u)
        # Critic selection is fixed to the registered main wm_ppo's final policy;
        # this is a val-only response probe, not an additional policy evaluation.
        checkpoint=run/f'seed{seed}/wm_ppo/final_policy.zip'
        critic=PPO.load(checkpoint,device='cpu');critic.policy.set_training_mode(False)
        encoded={name:encode(np.concatenate((h[:,1:],future[:,None]),axis=1),u) for name,future in predicted.items()}
        values={}
        with torch.no_grad():
            for name,obs in encoded.items():
                tensor,_=critic.policy.obs_to_tensor(obs);values[name]=critic.policy.predict_values(tensor).cpu().numpy().reshape(-1)
        for name,future in predicted.items():
            reward=raw_reward(future[:,0],future[:,1],previous,u)
            observation_difference=abs(encoded[name]-encoded['physical_enriched'])
            for family in ('constant','mild','brake','stopgo'):
                mask=val['family']==family
                h1rows.append(dict(seed=seed,model=name,family=family,rows=int(mask.sum()),
                    raw_reward_max_abs_difference=float(abs(reward[mask]-baseline_reward[mask]).max()),
                    encoded_next_obs_max_abs_difference=float(observation_difference[mask].max()),
                    leader_next_speed_mae_vs_physics_mps=float(abs(future[mask,2]-baseline[mask,2]).mean()),
                    actuator_next_state_mae_vs_physics_mps2=float(abs(future[mask,3]-baseline[mask,3]).mean()),
                    fixed_critic_next_value_mae_vs_physics=float(abs(values[name][mask]-values['physical_enriched'][mask]).mean()),
                    fixed_critic_next_value_max_abs_difference=float(abs(values[name][mask]-values['physical_enriched'][mask]).max()),
                    critic_reward_unit='training raw_reward*.01'))
        provenance[f'seed{seed}']=dict(models={name:modelhash[f'{name}.npz'] for name in NAMES},
            critic_probe_checkpoint=str(checkpoint.relative_to(ROOT)),critic_checkpoint_sha256=digest(checkpoint))
    episodes=aggregate(windows,('seed','episode','family','model','horizon_s'),metrics)
    family_seed=aggregate(episodes,('seed','family','model','horizon_s'),metrics)
    summary=aggregate(episodes,('family','model','horizon_s'),metrics)
    csv_write(out/'window_errors.csv',windows);csv_write(out/'episode_errors.csv',episodes)
    csv_write(out/'family_seed_errors.csv',family_seed);csv_write(out/'family_errors.csv',summary)
    csv_write(out/'actuator_residuals.csv',components);csv_write(out/'history_vs_learned_lead.csv',comparisons)
    csv_write(out/'h1_reward_and_bootstrap.csv',h1rows)
    evidence=dict(run=str(run),common_data_sha256=common['sha256'],validation_npz_sha256=digest(valfile),
        accessed_labels=['common_data/val.npz only'],models=provenance,
        validation_episode_count=len(np.unique(val['episode'])),seeds=registration['spec']['seeds'],
        recursive_horizons_s=[.1,.5,1.,3.],statistical_unit='episode means then family/seed; same validation episodes reused across model seeds',
        no_training=True,no_test_data=True,no_model_or_policy_selection=True,
        diagnostic_source_sha256=digest(Path(__file__)),recursive_helper_sha256=digest(ROOT/'scripts/analyze_validation.py'))
    (out/'manifest.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    def table(headers,rows):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(str(v) for v in row)+'|' for row in rows])
    max_difference={c:max(r[f'max_prediction_difference_{c}'] for r in comparisons) for c in COMPONENTS}
    max_reward=max(r['raw_reward_max_abs_difference'] for r in h1rows)
    max_critic=max(r['fixed_critic_next_value_max_abs_difference'] for r in h1rows)
    critic_by_model={name:max(r['fixed_critic_next_value_max_abs_difference'] for r in h1rows if r['model']==name) for name in NAMES}
    text=['# 当前SB3模型池的纯验证机制诊断', '',
        f"数据来自 `{run.relative_to(ROOT)}/common_data/val.npz`，common SHA `{common['sha256']}`，val文件SHA `{digest(valfile)}`。只读当前3seed冻结world_models和固定主wm_ppo最终critic进行val响应probe；不加载任何test标签、日志或评价数据，不训练，不重新选择策略/模型。", '',
        f"{evidence['validation_episode_count']}个独立val episode，按完整episode内非重叠30步窗口，先平均窗口，再按family和seed汇总。3seed共享同一组val场景，不把重复评估当新增独立场景。逐seed的0.1/0.5/1/3秒结果见family_seed_errors.csv。此数据不是旧MPC dataset，禁止套用旧0.0034→1.646m/s数字。", '',
        '## 三秒递归误差（当前模型池）', '',
        table(['族','模型','gap MAE m','ego MAE m/s','leader MAE m/s','actuator MAE m/s²'],
            [[r['family'],r['model'],*[f"{r['recursive_'+c]:.6g}" for c in COMPONENTS]] for r in summary if r['horizon_s']==3.]), '',
        '## 单步到多步的变化', '',
        table(['族','时域s','history leader MAE m/s','history gap MAE m','physics leader MAE m/s'],
            [[r['family'],r['horizon_s'],f"{r['recursive_lead_mps']:.6g}",f"{r['recursive_gap_m']:.6g}",
              f"{next(p['recursive_lead_mps'] for p in summary if p['family']==r['family'] and p['horizon_s']==r['horizon_s'] and p['model']=='physical_enriched'):.6g}"] for r in summary if r['model']=='history']), '',
        '## 自车残差与仅学习前车的对照', '',
        table(list(components[0]),[list(row.values()) for row in components]), '',
        f"history与learned_lead在全部登记窗口、四时域和3seed中的预测最大差分别为 `{max_difference}`，单位由字段标明。此比较定位自车残差是否带来可见变化；完整逐seed记录见history_vs_learned_lead.csv。", '',
        '## h1即时奖励与bootstrap分量', '',
        f"同一val根+同一实际连续动作、相同previous_u，各模型analytic raw reward相对丰富物理的最大差为{max_reward:.8g}。显式Euler的下一gap与ego速度由旧状态决定，因此h1的学习前车/actuator预测不会改变该步即时奖励。它们可以改变next_obs中lead/actuator，以及固定critic的bootstrap。", '',
        table(['seed','模型','族','reward最大差','nextobs最大差(编码)','同critic nextV平均绝对差'],
            [[r['seed'],r['model'],r['family'],f"{r['raw_reward_max_abs_difference']:.3g}",f"{r['encoded_next_obs_max_abs_difference']:.6g}",f"{r['fixed_critic_next_value_mae_vs_physics']:.6g}"] for r in h1rows if r['model']=='history']), '',
        f"各预测模型相对物理的同critic nextV最大差为 `{critic_by_model}`，单位为训练reward*.01的价值单位；全部模型最大值{max_critic:.6g}包含无历史residual，不能误称都是history的变化。critic跨预测模型共享；没有比较不同critic的绝对标尺，也没有用此probe选择策略。不能将nextV变化本身称为正确性或收益。", '',
        '## 解释范围', '',
        '这些val诊断只限定当前PPO研究的能力归因：h1主要通过模型next_obs和value bootstrap起作用，h3还会影响后续奖励。真实数据的长GAE与模型短分支目标时域不同，分支根重采样也改变样本权重；丰富物理辅助控制共享这些机制，更适合区分学习组件和通用分支效应。主研究已有h3/nohistory正式消融，此处不追加实验或调参。', '',
        '物理模型可精确匹配ID自车结构；无先兆外生急刹无法由6帧历史可靠预知。预测MAE或value响应不是PPO收益的替代证据；闭环正式结果、3seed不确定性、代价和失败必须按独立报告阅读。该分析不支持将旧MPC数据集的数值直接移植为新RL结果原因。', '']
    (ROOT/'docs/SB3_MODEL_DIAGNOSIS.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps(dict(output=str(out),common_sha=common['sha256'],validation_episodes=evidence['validation_episode_count'],
        max_history_vs_learned_lead=max_difference,max_h1_raw_reward_difference=max_reward,max_critic_value_difference=max_critic),ensure_ascii=False))


if __name__=='__main__':main()
