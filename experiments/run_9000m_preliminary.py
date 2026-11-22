"""Prepare the 500-episode 9 km study, optionally reusing verified checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from experiments import run_9000m_study as study

DEFAULT_OUTPUT = study.ROOT/'runs/local/overtaking_9000m_500ep'
EPISODES = 500


def prepare_preliminary(output=DEFAULT_OUTPUT, previous=None):
    output = Path(output).resolve()
    previous = Path(previous).resolve() if previous is not None else None
    if output == previous or output == study.DEFAULT_OUTPUT.resolve():
        raise ValueError('Preliminary results need a separate output directory')
    old = study.read_protocol(previous) if previous is not None else None
    if previous is not None:
        status = json.loads((previous/'study_status.json').read_text())
        if status.get('phase') == 'running':
            try:os.kill(status['pid'],0)
            except ProcessLookupError:pass
            else:raise RuntimeError('Stop the 5,000-episode coordinator at checkpoint boundaries first')
    # Include this wrapper in the new manifest without changing the legacy
    # runner source or its source hashes. Child workers read recorded paths.
    original = study.SOURCE_PATHS
    try:
        study.SOURCE_PATHS = [*original,'experiments/run_9000m_preliminary.py']
        study.prepare(output,episodes=EPISODES,seeds=old['training_seeds'] if old else study.TRAIN_SEEDS)
    finally:
        study.SOURCE_PATHS = original
    protocol = study.read_protocol(output)
    if previous is None:
        return output
    if protocol['config'] != old['config']:
        raise ValueError('Cannot reuse scenario checks for a different task')
    validation = json.loads((previous/'validation.json').read_text())
    if not validation['passed']:
        raise ValueError('Previous scenario validation did not pass')
    if validation['config_sha256'] != hashlib.sha256((output/'config.json').read_bytes()).hexdigest():
        raise ValueError('Scenario validation configuration differs')
    if validation['backend_sha256'] != protocol['source_hashes']['environments/sensor_highway/overtaking.py']:
        raise ValueError('Scenario validation traffic backend differs')
    for name in ['validation.json','scenario_validation.csv']:
        if not (output/name).exists():shutil.copy2(previous/name,output/name)
    pilot = previous/'pilot'
    study.read_protocol(pilot)
    if json.loads((pilot/'study_status.json').read_text())['phase'] != 'complete':
        raise ValueError('Previous pipeline pilot did not complete')
    if not (output/'pilot').exists():
        staging=output/f'pilot.copying.{os.getpid()}'
        shutil.copytree(pilot,staging)
        os.replace(staging,output/'pilot')
    if not (output/'preliminary_setup.json').exists():
        study.atomic_json(output/'preliminary_setup.json',{
            'created_at':study.timestamp(),'reused_checks_from':str(previous),
            'episodes_per_agent_per_seed':EPISODES,'training_seeds':protocol['training_seeds'],
            'models':len(study.POLICIES)*len(protocol['training_seeds']),
            'total_training_episodes':EPISODES*len(study.POLICIES)*len(protocol['training_seeds']),
            'fresh_training':True,
            'reason':'Decay epsilon from 1.0 to 0.1 over 500 episodes. Existing 5,000-episode-schedule checkpoints are preserved separately.',
            'evaluation':'Unchanged: 20 traffic seeds × 3 weather conditions per model.'})
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--reuse-checks-from',type=Path,
                        help='Reuse validation and pilot results from a matching local run')
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.workers<=6:raise ValueError('workers must be between 1 and 6')
    if args.prepare:prepare_preliminary(args.output,previous=args.reuse_checks_from)
    if args.launch:study.launch(args.output,args.workers)
    if args.status:
        for path in [args.output/'study_status.json',*sorted(args.output.glob('seed*/*/status.json'))]:
            if path.exists():print(path.relative_to(args.output),path.read_text().strip())


if __name__=='__main__':main()
