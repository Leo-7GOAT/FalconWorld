"""Build a checksummed, shadow-only ARM64 transfer bundle; never contact hardware."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', default='outputs/r550/speed_model')
    p.add_argument('--output', default='outputs/r550_deployment.zip')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    models = Path(args.model_dir).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        p.error('Choose a new output path; deployment bundles are immutable')
    names = ['falconworld/__init__.py', 'falconworld/contracts.py',
             'falconworld/adapters.py', 'falconworld/planning.py',
             'falconworld/r550_worldmodel.py', 'requirements-deploy.txt',
             'scripts/r550_prepare.sh', 'scripts/r550_package.py', 'scripts/r550_ros2_shadow.py',
             'scripts/r550_verify_model.py', 'scripts/r550_model_shadow.py',
             'scripts/r550_replay.py', 'docs/R550.md']
    files = {name: (root / name).read_bytes() for name in names}
    for name in ('residual_worldmodel.npz', 'residual_worldmodel.onnx',
                 'manifest.json', 'test_data.npz', 'onnx_verification.json',
                 'closed_residual_worldmodel_401.csv'):
        files['outputs/r550/speed_model/' + name] = (models / name).read_bytes()
    model_manifest = json.loads(files['outputs/r550/speed_model/manifest.json'])
    origin = model_manifest['origin']
    if hashlib.sha256(files['outputs/r550/speed_model/residual_worldmodel.npz']).hexdigest() != model_manifest['model_sha256']:
        raise ValueError('Model differs from training manifest')
    manifest = dict(format='falconworld-r550-shadow-bundle-v1', origin=origin,
                    permits_motion=False, target='ARM64 Ubuntu22.04 ROS2 Humble',
                    verification='Windows host only; run r550_verify_model.py on target',
                    sha256={k: hashlib.sha256(v).hexdigest() for k, v in files.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        entries = dict(files, **{'BUNDLE_MANIFEST.json': json.dumps(manifest, indent=2).encode('utf-8')})
        for name, contents in sorted(entries.items()):
            # Fixed metadata makes identical sources/models produce identical ZIPs
            # on the same Python/zlib toolchain, regardless of build time/path.
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (0o100755 if name.endswith('.sh') else 0o100644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, contents)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise AssertionError('Deployment archive CRC failed')
        for name, digest in manifest['sha256'].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise AssertionError('Deployment archive checksum mismatch')
    print(json.dumps(dict(output=str(output), files=len(files), origin=origin,
                          sha256=hashlib.sha256(output.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
