"""Build reviewable tables and plots from frozen-policy evaluation artifacts."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS={'sarsa':'Expected SARSA','dqn':'DQN','dyna_dqn':'Dyna-DQN',
        'oracle':'Privileged oracle','fixed_left_fast':'Fixed left + accelerate'}


def wilson(successes,trials):
    z=1.959963984540054
    p=successes/trials;denom=1+z*z/trials
    middle=(p+z*z/(2*trials))/denom
    half=z*np.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/denom
    return max(0.,middle-half),min(1.,middle+half)


def analyze(output):
    output=Path(output)
    df=pd.read_csv(output/'evaluation.csv')
    summary=pd.read_csv(output/'summary.csv')
    metadata=json.loads((output/'metadata.json').read_text())
    status=json.loads((output/'status.json').read_text())
    from experiments.evaluate_driving_generalization import DEFAULT_RUN
    source=Path(metadata['source_run'])
    if not source.is_dir():
        source=DEFAULT_RUN
    current_hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest()
                    for name in metadata['source_checkpoints_sha256']}
    assert current_hashes==metadata['source_checkpoints_sha256'], 'Source checkpoints changed'
    assert not df.duplicated(['scenario','weather','policy','seed']).any(), 'Duplicate trials'
    assert df.groupby(['scenario','weather','seed']).initial_traffic.nunique().max()==1, 'Unmatched starting traffic'
    assert df.groupby(['scenario','weather','seed']).initial_lane.nunique().max()==1, 'Unmatched ego lanes'
    assert ((df.completed.astype(int)+df.crashed.astype(int)+df.deadline_failure.astype(int))==1).all(), 'Ambiguous outcomes'
    for batch in metadata['batches']:
        for policy in metadata['policies']:
            count=len(df[(df.scenario==batch['case']['name'])&(df.weather==batch['weather'])&(df.policy==policy)])
            if status['phase']=='complete':assert count==batch['trips_per_policy'], 'Incomplete protocol'
    dqn=df[df.policy=='dqn'].set_index(['scenario','weather','seed']).sort_index()
    dyna=df[df.policy=='dyna_dqn'].set_index(['scenario','weather','seed']).sort_index()
    comparable=['completed','crashed','deadline_failure','elapsed_s','distance_m','overtakes','lane_changes']
    identical_dqn_dyna=(len(dqn)>0 and len(dqn)==len(dyna)
                        and dqn[comparable].equals(dyna[comparable]))
    audit=dict(checkpoints_unchanged=True, matched_starting_traffic=True,
               unique_trials=True, exclusive_outcomes=True, dqn_dyna_identical_outcomes=identical_dqn_dyna,
               completed_trips=len(df), scheduled_trips=sum(b['trips_per_policy']*len(metadata['policies']) for b in metadata['batches']))
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    training=pd.read_csv(source/'training.csv')
    dyna_training=training[training.policy.str.contains('Dyna')]
    planning_updates=int(dyna_training.planning_updates.max()) if len(dyna_training) else None
    intervals=[wilson(int(r.completions),int(r.trips)) for r in summary.itertuples()]
    summary['completion_ci95_low']=[v[0] for v in intervals]
    summary['completion_ci95_high']=[v[1] for v in intervals]
    summary.to_csv(output/'summary_with_intervals.csv',index=False)
    oracle=df[df.policy=='oracle'][['scenario','weather','seed','completed']].rename(
        columns={'completed':'oracle_completed'})
    paired=df[df.policy.isin(['sarsa','dqn','dyna_dqn'])].merge(oracle,on=['scenario','weather','seed'])
    paired['failed_when_oracle_completed']=~paired.completed & paired.oracle_completed
    paired['completed_when_oracle_failed']=paired.completed & ~paired.oracle_completed
    paired.to_csv(output/'paired_oracle_trials.csv',index=False)
    paired_summary=paired.groupby(['scenario','weather','policy'],sort=False).agg(
        trips=('completed','size'),oracle_completed=('oracle_completed','sum'),
        failures_despite_oracle_completion=('failed_when_oracle_completed','sum'),
        completions_despite_oracle_failure=('completed_when_oracle_failed','sum')).reset_index()
    paired_summary.to_csv(output/'paired_oracle_summary.csv',index=False)
    behavior=df.groupby(['scenario','weather','policy'],sort=False).agg(
        left_fraction=('action_left_fraction','mean'),right_fraction=('action_right_fraction','mean'),
        faster_fraction=('action_faster_fraction','mean'),slower_fraction=('action_slower_fraction','mean'),
        mean_lane_changes=('lane_changes','mean'),mean_following_s=('following_s','mean'),
        mean_overtakes=('overtakes','mean'),max_overtakes=('overtakes','max'),
        mean_distance_m=('distance_m','mean')).reset_index()
    behavior.to_csv(output/'behavior_summary.csv',index=False)
    lane_summary=df.groupby(['scenario','weather','policy','initial_lane'],sort=False).agg(
        trips=('completed','size'),completions=('completed','sum'),crashes=('crashed','sum'),
        deadline_failures=('deadline_failure','sum'),mean_lane_changes=('lane_changes','mean'),
        left_fraction=('action_left_fraction','mean'),right_fraction=('action_right_fraction','mean')).reset_index()
    lane_summary.to_csv(output/'initial_lane_summary.csv',index=False)
    cases=[b['case']['name'] for b in metadata['batches'] if b['weather']=='clear']
    policies=[p for p in ['sarsa','dqn','dyna_dqn'] if p in df.policy.unique()]
    fig,axes=plt.subplots(len(policies),1,figsize=(12,3.1*len(policies)),sharex=True,squeeze=False)
    for ax,policy in zip(axes[:,0],policies):
        s=summary[(summary.policy==policy)&(summary.weather=='clear')].set_index('scenario').reindex(cases)
        completed=s.completions/s.trips
        crashed=s.crashes/s.trips
        timeout=s.deadline_failures/s.trips
        ax.bar(cases,completed,color='#27865d',label='Completed')
        ax.bar(cases,crashed,bottom=completed,color='#c84c56',label='Crashed')
        ax.bar(cases,timeout,bottom=completed+crashed,color='#dea53d',label='Deadline failure')
        for i,row in enumerate(s.itertuples()):
            ax.text(i,1.025,f'{int(row.completions)}/{int(row.trips)}',ha='center',fontsize=9)
        ax.set_ylim(0,1.15);ax.set_yticks([0,.5,1],['0%','50%','100%'])
        ax.set_title(LABELS[policy],loc='left');ax.set_ylabel('Trips')
        ax.spines[['top','right']].set_visible(False)
    handles,legend_labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.5,.965),ncol=3,frameon=False)
    axes[-1,0].tick_params(axis='x',labelrotation=20)
    fig.suptitle('Frozen driving policies: new traffic seeds, all sensors on',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(output/'evaluation_outcomes.png',dpi=160);plt.close(fig)
    clear=summary[summary.weather=='clear'].copy()
    clear['finished / trials']=clear.completions.astype(str)+' / '+clear.trips.astype(str)
    pivot=clear.pivot(index='scenario',columns='policy',values='finished / trials').reindex(cases)
    pivot=pivot.reindex(columns=[p for p in LABELS if p in pivot.columns]).rename(columns=LABELS)
    detail=summary[['scenario','weather','policy','trips','completions','crashes','deadline_failures',
                    'mean_finish_s','mean_distance_m','mean_overtakes','max_overtakes']].copy()
    detail['policy']=detail.policy.map(LABELS)
    detail=detail.round(2)
    replays=[]
    for p in sorted((output/'replays').glob('*/*/replay.html')):
        rel=p.relative_to(output)
        replays.append(f'<li><a href="{html.escape(str(rel))}">{html.escape(str(rel.parent))}</a>'
                       f' · <a href="{html.escape(str(rel.parent/"policy.gif"))}">GIF</a></li>')
    report=f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Driving policy generalization results</title><style>
body{{font:16px system-ui;max-width:1450px;margin:32px auto;padding:0 24px;color:#17273c}}
table{{border-collapse:collapse;font-size:14px;display:block;overflow:auto;margin:20px 0}}
th,td{{padding:8px 11px;border:1px solid #d8e0ea;text-align:right}}th{{background:#eef3f8}}
p,li{{line-height:1.6}}img{{max-width:1100px;width:100%}}code{{background:#eef3f8;padding:2px 5px}}
</style><h1>Driving policy generalization</h1>
<p>Source: <code>{html.escape(Path(metadata['source_run']).name)}</code>.
Each policy trained for {metadata['training_settings']['episodes']} episodes, one training seed
({metadata['training_seed']}). Evaluation phase: <b>{html.escape(status['phase'])}</b>,
{len(df)} trips. Checkpoints unchanged: <b>{status.get('checkpoints_unchanged',False)}</b>.</p>
<p>Frozen policies; all sensors stay on. New traffic seeds begin at {metadata['seed_start']}.
Clear-weather cases have 20 trials per policy in the full protocol; rain/fog have 10.
Finish times are conditional on completion. Longer routes also change normalized task observations.
Rain/fog affect sensing; this environment does not change tire/road friction with weather.
Traffic holds lanes: cut-ins and active lane-changing traffic are outside this evaluation.</p>
<p>Dyna performed <b>{planning_updates}</b> planning updates during source training.
DQN and Dyna have identical paired evaluation outcomes: <b>{identical_dqn_dyna}</b>.
This run therefore does not establish any benefit from learned-model planning.</p>
<h2>Clear-weather completions</h2>{pivot.to_html()}
<img src="evaluation_outcomes.png" alt="Completion, crash and deadline outcomes by traffic scenario">
<h2>Per-scenario outcomes and overtakes</h2>{detail.to_html(index=False)}
<h2>Feasibility comparison</h2>
<p>The oracle reads simulator truth and is a heuristic, not an optimal controller. Its successful
trips demonstrate that those particular seeded cases can be completed. Its failures do not
establish that a scenario is impossible. The fixed driver ignores traffic and always moves
left then accelerates; strong performance on a case means that case admits this simple strategy.</p>
{paired_summary.to_html(index=False)}
<h2>Replays</h2><ul>{''.join(replays)}</ul>
<p><a href="evaluation.csv">Every evaluation trip</a> ·
<a href="summary_with_intervals.csv">Summary with 95% Wilson completion intervals</a> ·
<a href="behavior_summary.csv">Action and overtaking metrics</a> ·
<a href="initial_lane_summary.csv">Starting lane breakdown</a> ·
<a href="metadata.json">Protocol and checkpoint hashes</a> · <a href="audit.json">Integrity audit</a></p>
<p>Intervals describe scenario sampling for these checkpoints. One training seed cannot establish
robustness to training randomness, and zero observed crashes is not a safety guarantee.</p>
</html>'''
    (output/'results.html').write_text(report)
    print(pivot.to_string())
    print(f'Report: {output/"results.html"}')
    return summary,paired_summary,behavior


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    analyze(parser.parse_args().output)
