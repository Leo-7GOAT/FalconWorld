"""Explain frozen-model errors using train/val only; never trains or selects models."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from falconworld.models import load_models, _actuator_features, _leader_features

HORIZONS=(1,5,10,30)
COMPONENTS=('gap_m','ego_mps','lead_mps','actuator_mps2')


def csv_write(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def aggregate(rows,keys,metrics):
    grouped=defaultdict(list)
    for row in rows:grouped[tuple(row[k] for k in keys)].append(row)
    return [dict(zip(keys,key),episode_seed_evaluations=len(v),
                 **{m:float(np.mean([r[m] for r in v])) for m in metrics})
            for key,v in sorted(grouped.items())]


class ComponentSwap:
    """Diagnostic only: replace one learned component in an enriched physical step."""
    def __init__(self,physics,learned,component):
        self.physics,self.learned,self.component=physics,learned,component

    def step(self,state,action,history):
        baseline=self.physics.step(state,action,history)
        learned=self.learned.step(state,action,history)
        baseline[...,self.component]=learned[...,self.component]
        return baseline


def rollout(model,state,actions,history):
    if getattr(model,'members',None):
        return np.mean([rollout(m,state,actions,history) for m in model.members],axis=0)
    s,h=state.copy(),history.copy();result=[]
    for action in actions:
        s=model.step(s,action,h)
        result.append(s.copy())
        h=np.concatenate((h[:,1:],s[:,None]),axis=1) if s.ndim==2 else np.concatenate((h[1:],s[None]))
    return np.asarray(result)


def evaluate_validation(models,data,seed):
    rows=[];windows=[]
    for episode in np.unique(data['episode']):
        ix=np.flatnonzero(data['episode']==episode)
        for offset in range(0,len(ix)-29,30):
            index=ix[offset:offset+30]
            windows.append((episode,offset,index))
    indices=np.asarray([w[2] for w in windows]);flat=indices.reshape(-1)
    target=data['next_state'][indices]
    for name,model in models.items():
        predicted=rollout(model,data['state'][indices[:,0]],data['action'][indices].T,
                          data['history'][indices[:,0]]).transpose(1,0,2)
        # Diagnostic teacher forcing only, never the recursive predictor input.
        teacher=model.step(data['state'][flat],data['action'][flat],data['history'][flat]).reshape(len(indices),30,4)
        for wi,(episode,offset,index) in enumerate(windows):
                for h in HORIZONS:
                    recursive=abs(predicted[wi,h-1]-target[wi,h-1]);one_step=abs(teacher[wi,h-1]-target[wi,h-1])
                    rows.append(dict(seed=seed,episode=int(episode),family=str(data['family'][index[0]]),offset=offset,
                        model=name,horizon_s=h*.1,
                        **{f'recursive_{c}':float(recursive[j]) for j,c in enumerate(COMPONENTS)},
                        **{f'teacher_forced_endpoint_{c}':float(one_step[j]) for j,c in enumerate(COMPONENTS)}))
    return rows


def head_losses(model,data):
    state,action=data['state'],data['action']
    target=data['next_state'][:,3:4].copy()
    if model.residual:target-=model.physics.actuator(state,action)[:,None]
    actuator=model.actuator_net(_actuator_features(state,action))
    leader=model.leader_net(_leader_features(state,data['history'],model.with_history))
    return dict(actuator=float(np.mean(((actuator-target)/model.actuator_net.yscale)**2)),
                leader=float(np.mean(((leader-(data['next_state'][:,2:3]-state[:,2:3]))/model.leader_net.yscale)**2)))


def draw_figures(output,summary,curves,losses):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=('physical_enriched','neural','residual','history','learned_lead','ensemble')
    fig,axes=plt.subplots(2,4,figsize=(15,7),sharex=True)
    for col,family in enumerate(('constant','mild','brake','stopgo')):
        for name in names:
            selected=[r for r in summary if r['family']==family and r['model']==name]
            for ax,component in zip(axes[:,col],('gap_m','lead_mps')):
                ax.plot([r['horizon_s'] for r in selected],[r[f'recursive_{component}'] for r in selected],marker='.',label=name)
        axes[0,col].set_title(family)
        axes[1,col].set_xlabel('Recursive horizon [s]')
    axes[0,0].set_ylabel('Gap MAE [m]');axes[1,0].set_ylabel('Leader MAE [m/s]')
    for ax in axes.flat:ax.grid(alpha=.2)
    axes[0,-1].legend(fontsize=7)
    fig.suptitle('Validation only: each episode weighted equally, three frozen training seeds')
    fig.tight_layout();fig.savefig(output/'recursive_by_family.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(12,12))
    for i,name in enumerate(('neural','residual','history','learned_lead')):
        for j,head in enumerate(('actuator','leader')):
            ax=axes[i,j]
            for seed in (0,1,2):
                curve=[r for r in curves if r['model']==name and r['head']==head and r['seed']==seed]
                ax.plot([r['epoch'] for r in curve],[r['validation_mse_normalized'] for r in curve],label=f'seed{seed} val')
                last=next((r for r in losses if r['model']==name and r['head']==head and r['seed']==seed),None)
                if last:
                    ax.scatter([last['selected_epoch']],[last['train_final_mse_normalized']],marker='x',s=30)
            ax.set_title(f'{name}: {head}');ax.set_xlabel('Epoch');ax.set_ylabel('Normalized head MSE');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Logged validation curves; x = train MSE of final selected checkpoint (no train epoch logs)')
    fig.tight_layout();fig.savefig(output/'learning_curves.png',dpi=140);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'outputs/standard_verified')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/validation_analysis')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    # Explicitly read only these two files: test/ood labels are not loaded.
    data={split:dict(np.load(args.run/'data'/f'{split}.npz',allow_pickle=False)) for split in ('train','val')}
    manifest=json.loads((args.run/'data/manifest.json').read_text(encoding='utf-8'))
    rows=[];curves=[];losses=[];calibration=[];hashes={}
    for split in ('train','val'):
        p=args.run/'data'/f'{split}.npz';hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for seed in (0,1,2):
        folder=args.run/f'seed{seed}'/'models';models=load_models(folder)
        log=json.loads((folder/'training.json').read_text(encoding='utf-8'))
        if log['data_sha256']!=manifest['sha256']:raise ValueError('Training/data provenance mismatch')
        physics=models['physical_enriched']
        calibration.append(dict(seed=seed,tau=physics.tau,enriched=physics.enriched))
        for file in folder.glob('*.npz'):hashes[str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
        evaluated=dict(models)
        for name in ('neural','residual','history'):
            evaluated[name+'__ego_only']=ComponentSwap(physics,models[name],3)
            evaluated[name+'__leader_only']=ComponentSwap(physics,models[name],2)
        rows.extend(evaluate_validation(evaluated,data['val'],seed))
        for name in ('neural','residual','history','learned_lead'):
            train_loss,val_loss=head_losses(models[name],data['train']),head_losses(models[name],data['val'])
            for head in ('actuator','leader'):
                records=log['models'][name][head]
                for record in records:curves.append(dict(seed=seed,model=name,head=head,**record))
                # Match the frozen training checkpoint rule, including tolerance.
                best=records[0]
                for record in records[1:]:
                    if record['validation_mse_normalized']<best['validation_mse_normalized']-1e-7:
                        best=record
                losses.append(dict(seed=seed,model=name,head=head,selected_epoch=best['epoch'],
                    train_final_mse_normalized=train_loss[head],val_final_mse_normalized=val_loss[head],
                    validation_best_logged=best['validation_mse_normalized']))
    metrics=[key for key in rows[0] if key.startswith(('recursive_','teacher_'))]
    episodes=aggregate(rows,('seed','episode','family','model','horizon_s'),metrics)
    summary=aggregate(episodes,('family','model','horizon_s'),metrics)
    csv_write(args.output/'window_errors.csv',rows);csv_write(args.output/'episode_errors.csv',episodes)
    csv_write(args.output/'by_family.csv',summary);csv_write(args.output/'learning_curves.csv',curves)
    csv_write(args.output/'final_head_losses.csv',losses)
    provenance=dict(accessed_splits=['train','val'],training_seeds=[0,1,2],data_sha256=manifest['sha256'],
        file_sha256=hashes,calibration=calibration,purpose='explain fixed-model errors; no training or test selection',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    draw_figures(args.output,summary,curves,losses)
    def tab(headers,records):
        return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
                         ['|'+'|'.join(str(c) for c in r)+'|' for r in records])
    text=['# 冻结模型的训练与验证诊断', '',
        '命令：`python scripts/analyze_validation.py`。本分析显式只打开train.npz与val.npz，不加载test/ood标签，不训练、不调整模型或控制器、不据此挑选本轮测试赢家。3个seed共享冻结验证episode；先在episode内平均窗口，再平均seed×episode；这些重复评估不是新增独立场景。', '',
        '## 结构匹配的强物理与自车残差', '',
        f"丰富物理的3次辨识结果：`{calibration}`。仿真ID真值为加速0.45s、制动0.75s，二者乘(1+0.045v)。该机制在train中可识别，val仍属于相同结构。强物理对ID自车响应可达到浮点误差量级；不应故意限制它来制造学习优势。", '',
        '下表为val按族三秒递归结果。gap与速度运动学是模型公共已知结构；NN只学习执行器和前车分量，不能把全模型称作无结构状态黑盒。`__ego_only`仅换学习执行器分量、前车仍用CA；`__leader_only`只换学习前车分量、自车仍用丰富物理。两者是冻结权重的诊断组合，不是新增训练或部署方案。', '',
        tab(['场景','模型/诊断组合','gap MAE m','ego MAE m/s','lead MAE m/s','actuator MAE m/s²'],
            [[r['family'],r['model'],*[f"{r['recursive_'+c]:.5g}" for c in COMPONENTS]] for r in summary if r['horizon_s']==3.]), '',
        '## 单步与递归复合', '',
        '所有模型在相同30步动作窗口从相同观测/历史递归。teacher_forced_endpoint只用于事后诊断：它在每个端点重新读取该时刻真实val状态/历史，预测下一步，所以绝不是部署可用的3秒预测，也不将其真值输入在线规划。gap与ego速度使用已知显式积分，在teacher forcing下可以精确，但递归时前车和执行器的偏差会经过积分传播。不能把递归误差与单步误差的比值叫做训练泛化倍率。', '',
        '全部0.1/0.5/1/3秒按族、模型、seed和episode的两种误差见 `outputs/validation_analysis/by_family.csv` 及 episode_errors.csv。下图仅val，不混入测试轨迹。', '',
        tab(['场景','history时域s','递归gap MAE m','递归前车MAE m/s','teacher forcing前车单步MAE m/s'],
            [[r['family'],r['horizon_s'],f"{r['recursive_gap_m']:.6g}",f"{r['recursive_lead_mps']:.6g}",f"{r['teacher_forced_endpoint_lead_mps']:.6g}"] for r in summary if r['model']=='history']), '',
        '![按族递归误差](../outputs/validation_analysis/recursive_by_family.png)', '',
        '## 学习曲线与训练目标', '',
        '原始训练只保存了每5 epoch的验证归一化MSE，未保存逐epoch训练MSE。图中实线为真实日志中的val曲线，叉号为最终选中权重在train上重新计算的误差，不伪造训练曲线；final_head_losses.csv保留每个seed、head、最终train/val数值。leader与actuator分别归一化，不能将不同head的归一化loss解释为米或安全指标。', '',
        '![真实val学习曲线与最终train误差](../outputs/validation_analysis/learning_curves.png)', '',
        '## 可验证机制与推断边界', '',
        '代码事实：前车头只读前车当前速度或其6帧历史，不读自车动作、场景身份或未知未来刹车脚本；它被单步前车速度差的均方误差训练。没有先兆的急刹起点不在输入中；同样的短历史可对应不同下一步，因此不能可靠提前预知。', '',
        '推断：均方误差在多种可能未来下偏向条件均值；急刹转折和停车后再启动的多模态未来会被平均。此处固定模型未显式输出混合分布，不能把该解释当作已证明的完整误差分解。', '',
        '代码事实：history模型只给前车头增加历史，自车执行器头仍为当前速度/执行器/动作的MLP。其残差基于已匹配结构的丰富物理；残差接近零与learned_lead接近是预期结果，非训练能力证明。neural自车头要近似一套本可精确辨识的函数，因此可能引入额外近似误差。', '',
        '实证：val匀速族history前车单步MAE约0.003423m/s，但0.5/1/3秒递归前车MAE约0.06589/0.28252/1.64569m/s，三秒gap MAE约1.95641m；物理CA该族前车误差为0。只交换history前车头重现约1.95642m的gap误差，只交换执行器头约1.56e-6m，说明这一失效主要来自前车递归漂移。以上数字来自当前3seed固定验证集，可用CSV核对。', '',
        '实证：history前车头最终train/val归一化MSE均约0.052–0.054，二者接近；不能仅凭递归变差就断言简单的训练过拟合。单步目标不能约束递归稳定性，且对真实零变化输入也存在小偏置，这与上面的匀速递归漂移观测一致。', '',
        '推断：单步teacher-forced验证损失选择的网络不直接优化30步递归损失或MPC代价。递归中的前车速度误差累积为gap偏差，历史窗口也逐渐被自身预测替换，可能偏离训练时由真值构成的历史分布。与强CA基线相比，短历史平滑近似可能在某些族更差；是否发生由上表/CSV验证，不能先验宣称history一定有用。', '',
        '本分析不评价测试外的OOD新分布，不据此重新调本轮测试。主报告保留真实闭环、计算预算、回退与负结果；即使val递归MAE改善，也不能自动推出MPC驾驶改善。额外训练成本和推理成本只有在同等闭环任务的多指标收益足够时才值得。', '']
    (ROOT/'docs/VALIDATION_ANALYSIS.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),windows=len(rows),episode_seed_metrics=len(episodes),splits=['train','val'])))


if __name__=='__main__':main()
