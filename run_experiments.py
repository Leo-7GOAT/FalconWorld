"""One-command reproducible longitudinal experiment; resumes completed episodes."""
import os
# Must precede NumPy imports: small batched networks are faster without BLAS oversubscription.
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from falconworld.experiments import write_json,open_loop,benchmark,profile_models,plot_results
from falconworld.planning import CEMConfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset',choices=('smoke','standard','full'),default='smoke')
    parser.add_argument('--stage',choices=('all','train','evaluate','summarize'),default='all')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--seeds',type=int,nargs='+')
    parser.add_argument('--budget',choices=('equal','time','both'),default='both')
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    output=args.output or Path('outputs')/args.preset
    output.mkdir(parents=True,exist_ok=True)
    if args.stage=='summarize':
        plot_results(output)
        return
    seeds=args.seeds or list(range(dict(smoke=1,standard=3,full=5)[args.preset]))
    if len(set(seeds))!=len(seeds) or args.workers<1:
        raise ValueError('Seeds must be unique and workers positive')
    configuration=dict(preset=args.preset,seeds=seeds,budget=args.budget,
        observation='privileged simulator state; actuator observed',dt=.1,
        protocol='docs/EXPERIMENT_PROTOCOL.md',workers=args.workers)
    run_path=output/'run_config.json'
    if run_path.exists() and json.loads(run_path.read_text())!=configuration:
        raise ValueError('Run configuration changed; use a new output directory')
    from falconworld.data import generate_dataset,load_dataset,DATA_VERSION
    from falconworld.models import train_models,load_models
    manifest=output/'data'/'manifest.json'
    if manifest.exists():
        data=load_dataset(output/'data')
        if data['metadata']['version']!=DATA_VERSION or data['metadata']['preset']!=args.preset:
            raise ValueError('Dataset version/preset mismatch; use a new output directory')
        digest=hashlib.sha256()
        for split in ('train','val','test','ood'):
            for field in ('state','action','next_state','episode'):digest.update(data[split][field].tobytes())
        if digest.hexdigest()!=data['metadata']['sha256']:raise ValueError('Dataset content fingerprint mismatch')
    else:
        if args.stage=='evaluate':raise ValueError('Evaluation requires an existing frozen dataset')
        data=generate_dataset(output/'data',preset=args.preset)
    snapshot=output/'source_snapshot'
    source_paths=[Path('run_experiments.py'),*[Path('falconworld')/name for name in
        ('__init__.py','data.py','models.py','planning.py','experiments.py')]]
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    snapshot_manifest=output/'source_sha256.json'
    if snapshot_manifest.exists() and json.loads(snapshot_manifest.read_text())!=source_hashes:
        raise ValueError('Source changed since run was frozen; choose a new output directory')
    for source in source_paths:
        destination=snapshot/source;destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():shutil.copyfile(source,destination)
    write_json(snapshot_manifest,source_hashes)
    write_json(run_path,configuration)
    for seed in seeds:
        target=output/f'seed{seed}'
        target.mkdir(exist_ok=True)
        fingerprint=dict(data_sha256=data['metadata']['sha256'],seed=seed,preset=args.preset,
            model_source_sha256=hashlib.sha256(Path('falconworld/models.py').read_bytes()).hexdigest())
        recorded=target/'training_fingerprint.json'
        reusable=recorded.exists() and json.loads(recorded.read_text())==fingerprint
        if args.stage=='evaluate' and not reusable:
            raise ValueError('Training/data/source fingerprint mismatch or missing')
        if args.stage=='evaluate' or (args.stage=='all' and reusable):
            models=load_models(target/'models')
            if set(models)!={'physical_cv','physical_ca','physical_enriched','neural',
                             'residual','history','learned_lead','ensemble'}:
                raise ValueError('Saved model set is incomplete or unexpected')
        else:
            models=train_models(data,target/'models',seed=seed,preset=args.preset)
            write_json(recorded,fingerprint)
        if args.stage=='train':
            continue
        open_loop(models,data,target,seed)
        matched=profile_models(models,data['val'],target,CEMConfig())
        for budget in ('equal','time') if args.budget=='both' else (args.budget,):
            benchmark(models,target,seed,args.preset,budget,matched,args.workers)
    plot_results(output)
    print(f'Artifacts: {output.resolve()}',flush=True)


if __name__=='__main__':
    main()
