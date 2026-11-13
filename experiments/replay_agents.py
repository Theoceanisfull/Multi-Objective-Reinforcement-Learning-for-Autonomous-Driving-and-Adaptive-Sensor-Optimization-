"""Export matched, inspectable replays of the three saved long-training agents.

Run from the project root:
    .venv/bin/python -m experiments.replay_agents --seed 100000 --weather clear

The HTML viewer embeds all frames and works offline. GIFs provide quick previews;
CSV files record every decision. Replays use frozen policies, not new training.
"""
import argparse
import base64
from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from algorithms.dqn_agent import DQNAgent
from algorithms.sensor_agent import LinearExpectedSarsa
from environments.sensor_highway import ExperimentConfig
from experiments.agent_helpers import run_agent

DEFAULT_RUN = Path("runs/comparisons/dqn_long_training_20261112_170403_004536")
POLICIES = (("sarsa", "Expected SARSA", "sarsa.npz"),
            ("dqn", "DQN", "dqn.pt"), ("dyna_dqn", "Dyna-DQN", "dyna_dqn.pt"))


def png_url(pixels):
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def map_pixels(observation):
    """Show occupancy evidence; unsupported cells are grey, outside the map is dark."""
    grid = observation["grid"]
    rgb = (matplotlib.colormaps["coolwarm"](grid[0])[..., :3] * 255).astype(np.uint8)
    rgb[grid[1] <= .01] = (160, 166, 174)
    rgb[observation["valid_cells"] == 0] = (24, 32, 44)
    center = rgb.shape[0] // 2
    rgb[center - 2:center + 3, center] = (255, 225, 30)
    rgb[center, center - 2:center + 3] = (255, 225, 30)
    return rgb


def replay_frames(rollout, metrics, label):
    """Frame zero is reset; later frames show the action used in the preceding step."""
    data, gifs = [], []
    font = ImageFont.load_default(size=18)
    for i, (road, observation) in enumerate(zip(rollout.frames, rollout.snapshots)):
        row = rollout.history.iloc[i - 1] if i else None
        time_s = float(row.time_s) if i else 0.0
        record = {"time": time_s, "drive": str(row.drive) if i else "RESET",
                  "sensing": str(row.sensing) if i else "off",
                  "speed": float(observation["ego"][0]),
                  "energy": float(row.total_energy_j) if i else 0.0,
                  "distance": float(row.distance_m) if i else 0.0,
                  "reward": float(row.reward) if i else 0.0,
                  "sensors": observation["sensors"].tolist(),
                  "road": png_url(road), "map": png_url(map_pixels(observation))}
        data.append(record)
        canvas = Image.new("RGB", (900, 510), (244, 247, 251))
        draw = ImageDraw.Draw(canvas)
        draw.text((18, 12), f"{label} | t={time_s:.1f} s", font=font, fill=(18, 32, 52))
        draw.text((18, 40), f"Action: {record['drive']} | sensing: {record['sensing']}", font=font, fill=(18, 32, 52))
        canvas.paste(Image.fromarray(road).resize((864, 216)), (18, 76))
        canvas.paste(Image.fromarray(map_pixels(observation)).resize((180, 180), Image.Resampling.NEAREST), (18, 310))
        lines = [f"Speed: {record['speed']:.1f} m/s | distance: {record['distance']:.1f} m",
                 f"Total sensing energy: {record['energy']:.1f} J",
                 f"Requested ON: {int(observation['sensors'][:, 0].sum())}/18 | ready: {int(observation['sensors'][:, 1].sum())}/18",
                 "Map: blue = free evidence; red = occupied evidence",
                 "Grey = unsupported; yellow cross = ego vehicle"]
        if i == len(rollout.frames) - 1:
            lines.append("Outcome: " + ("COMPLETED" if metrics["completed"] else "CRASHED" if metrics["crashed"] else "ENDED"))
        for j, line in enumerate(lines):
            draw.text((218, 314 + j * 29), line, font=font, fill=(18, 32, 52))
        gifs.append(canvas)
    return data, gifs


HTML = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Matched agent replays</title><style>
*{box-sizing:border-box}body{font:15px system-ui,sans-serif;background:#eef2f7;color:#17273c;margin:0;padding:24px}main{max-width:1500px;margin:auto}h1{margin:0 0 8px}p{line-height:1.5}.muted{color:#536379}.controls{position:sticky;top:0;z-index:5;background:#fff;padding:16px;border-radius:12px;box-shadow:0 3px 18px #17273c15;display:flex;gap:12px;flex-wrap:wrap;align-items:center}button,select{font:inherit;padding:8px 13px;border:1px solid #bcc8d8;border-radius:7px;background:white;color:#17273c;cursor:pointer}button:first-child{background:#164ac5;color:white;border-color:#164ac5}input{flex:1;min-width:180px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;margin-top:22px}.card{background:#fff;border-radius:12px;overflow:hidden;border:1px solid #d8e0ea}.pad{padding:16px}.card h2{margin:0 0 4px;font-size:20px}.road{width:100%;display:block;image-rendering:auto}.lower{display:flex;gap:14px;padding:16px}.map{width:150px;height:150px;image-rendering:pixelated}.stats{line-height:1.8;font-variant-numeric:tabular-nums}.tags{display:flex;flex-wrap:wrap;gap:5px}.tag{font-size:11px;border-radius:4px;padding:4px 6px;background:#e6ebf1;color:#526174}.warm{background:#fff0c4;color:#805d00}.ready{background:#d2f3e2;color:#12573b}.status{font-weight:650}.finish{color:#167345}.crash{color:#b42336}footer{margin-top:20px}.small{font-size:12px}@media(max-width:1100px){.grid{grid-template-columns:1fr}.road{max-height:240px;object-fit:contain;background:#646464}}
</style></head><body><main>
<h1>Matched agent replays</h1><p id="scenario"></p>
<p class="muted">Frozen policies from the 1,000-episode run. Same starting scenario; actions produce different traffic interactions. Green car = agent. The road view is simulator truth; the map is observation evidence available to the policy.</p>
<div class="controls"><button id="play">Play</button><button id="previous">Previous frame</button><button id="next">Next frame</button><button id="reset">Reset</button><label>Speed <select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><input aria-label="Replay time" id="timeline" type="range" min="0" step="1" value="0"><strong id="clock"></strong></div>
<div class="grid" id="cards"></div>
<footer class="muted"><p>Frames are captured at decision boundaries (2 Hz, every 0.5 s), not at the 100 Hz physics rate. “Action” is the command applied during the interval ending at the displayed frame. A finished replay holds its final frame while the others continue.</p><p>Map: blue = free-space evidence; red = occupancy evidence; grey = unsupported; yellow cross = ego. Sensor badges: grey = off, amber = warming up, green = ready. Keeping a sensor on does not guarantee a measurement. The map is not a raw camera image or validated ground truth.</p><p>This first held-out scenario is illustrative; it does not replace the aggregate evaluation. No external assets or internet connection are needed. Open each GIF for a separate preview, or inspect its CSV for the complete action and reward history.</p></footer>
</main><script>
const replay = __DATA__;
let index=0,timer=null;
const maxIndex=Math.max(...replay.agents.map(a=>a.frames.length-1));
const slider=document.getElementById('timeline');slider.max=maxIndex;
document.getElementById('scenario').textContent=`Weather: ${replay.weather} · traffic seed: ${replay.seed} · ${replay.route_m === null ? "Continuing driving" : replay.route_m + " m route"} / ${replay.deadline_s} s deadline`;
const cards=document.getElementById('cards');
for(const a of replay.agents){const card=document.createElement('article');card.className='card';card.id=a.id;
card.innerHTML=`<div class="pad"><h2>${a.label}</h2><div class="muted small">Final: ${a.metrics.completed?'completed':a.metrics.crashed?'crashed':'ended'} at ${a.metrics.elapsed_s.toFixed(1)} s · ${a.metrics.energy_j.toFixed(1)} J sensing energy</div></div><img class="road" alt="${a.label} road simulation"><div class="lower"><div><img class="map" alt="${a.label} occupancy evidence"><div class="small muted">Forward ↑ &nbsp; Right →</div></div><div class="stats"></div></div><div class="pad"><div class="small muted" style="margin-bottom:8px">Sensor request/readiness</div><div class="tags"></div><p class="small"><a href="${a.id}.gif">GIF preview</a> · <a href="${a.id}_history.csv">Decision CSV</a></p></div>`;cards.appendChild(card);}
function render(){slider.value=index;document.getElementById('clock').textContent=`t = ${(index*replay.step_s).toFixed(1)} s`;
for(const a of replay.agents){const f=a.frames[Math.min(index,a.frames.length-1)],card=document.getElementById(a.id);card.querySelector('.road').src=f.road;card.querySelector('.map').src=f.map;
const ended=index>=a.frames.length-1;
card.querySelector('.stats').innerHTML=`<div><b>${f.drive}</b></div><div>Sensing: <b>${f.sensing}</b></div><div>Speed: ${f.speed.toFixed(1)} m/s</div><div>Distance: ${f.distance.toFixed(1)} m</div><div>Energy: ${f.energy.toFixed(1)} J</div><div>Step reward: ${f.reward.toFixed(2)}</div>${ended?`<div class="status ${a.metrics.completed?'finish':'crash'}">${a.metrics.completed?'COMPLETED':a.metrics.crashed?'CRASHED':'ENDED'}</div>`:''}`;
const tags=card.querySelector('.tags');tags.replaceChildren();replay.sensor_names.forEach((name,i)=>{const s=f.sensors[i],tag=document.createElement('span');tag.className='tag'+(s[0]>.5?(s[1]>.5?' ready':' warm'):'');tag.textContent=name;tag.title=`${name}: ${s[0]>.5?'requested on':'off'}, ${s[1]>.5?'ready':'not ready'}; recent packet: ${s[2]>.5?'yes':'no'}`;tags.appendChild(tag);});}}
function pause(){clearInterval(timer);timer=null;document.getElementById('play').textContent='Play';}
function play(){if(index>=maxIndex)index=0;render();timer=setInterval(()=>{if(index>=maxIndex){pause();return;}index++;render();},1000*replay.step_s/Number(document.getElementById('speed').value));document.getElementById('play').textContent='Pause';}
document.getElementById('play').onclick=()=>timer?pause():play();
slider.oninput=()=>{pause();index=Number(slider.value);render();};
document.getElementById('previous').onclick=()=>{pause();index=Math.max(0,index-1);render();};
document.getElementById('next').onclick=()=>{pause();index=Math.min(maxIndex,index+1);render();};
document.getElementById('reset').onclick=()=>{pause();index=0;render();};
document.getElementById('speed').onchange=()=>{if(timer){pause();play();}};
render();
</script></body></html>'''


def export_replays(run_dir=DEFAULT_RUN, seed=100000, weather="clear", output=None,
                   duration_s=None, route_distance_m=None):
    torch.set_num_threads(1)
    run_dir = Path(run_dir).resolve()
    config = ExperimentConfig.load(run_dir / "config.json")
    config.weather.preset = weather
    if duration_s is not None:
        config.highway["duration"] = duration_s
    if route_distance_m is not None:
        config.reward.route_distance_m = route_distance_m
    config.validate()
    if output is None:
        stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S_%f")
        task = (f"_{config.highway['duration']:g}s_{config.reward.route_distance_m:g}m"
                if (duration_s is not None or route_distance_m is not None)
                and config.reward.route_distance_m is not None else "")
        output = run_dir / f"replays{task}_{weather}_seed{seed}_{stamp}"
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config.save(output / "config.json")
    replay = {"seed": seed, "weather": weather,
              "step_s": 1 / config.highway["policy_frequency"],
              "route_m": config.reward.route_distance_m, "deadline_s": config.highway["duration"],
              "agents": []}
    for policy_id, label, filename in POLICIES:
        agent = (LinearExpectedSarsa.load(run_dir / filename) if policy_id == "sarsa"
                 else DQNAgent.load(run_dir / filename))
        metrics, rollout = run_agent(config, agent, seed=seed, capture=True)
        metrics = {key: None if isinstance(value, float) and not np.isfinite(value) else value
                   for key, value in metrics.items()}
        frames, gif_frames = replay_frames(rollout, metrics, label)
        durations = [round(replay["step_s"] * 1000)] * len(gif_frames)
        durations[-1] = 2000  # Hold the outcome for inspection.
        gif_frames[0].save(output / f"{policy_id}.gif", save_all=True,
                           append_images=gif_frames[1:], duration=durations, loop=0)
        rollout.history.to_csv(output / f"{policy_id}_history.csv", index=False)
        (output / f"{policy_id}_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
        replay["sensor_names"] = rollout.sensor_names
        replay["agents"].append({"id": policy_id, "label": label, "metrics": metrics, "frames": frames})
        print(f"Saved {label}: {metrics['elapsed_s']:.1f} s, completed={metrics['completed']}", flush=True)
    data = json.dumps(replay, allow_nan=False)
    (output / "replay.html").write_text(HTML.replace("__DATA__", data))
    (output / "manifest.json").write_text(json.dumps({"source_run": str(run_dir), "seed": seed,
        "weather": weather, "capture_rate_hz": config.highway["policy_frequency"],
        "frozen_policies": True, "duration_s": config.highway["duration"],
        "route_distance_m": config.reward.route_distance_m, "policies": [a[0] for a in POLICIES]}, indent=2))
    (output / "README.md").write_text(f"""# Matched simulation replays

Open `replay.html` in a browser for synchronized play/pause, speed control, and frame stepping.
Individual GIFs and decision CSVs are beside it. All assets in the HTML are embedded.

- Source: `{run_dir.name}`; frozen policies trained for 1,000 episodes.
- Traffic seed: **{seed}**; weather: **{weather}**.
- Task: **{config.reward.route_distance_m} m**, up to **{config.highway["duration"]} s**.
- Models are frozen. Changed route/deadline settings test transfer to a different task; crashes or finishes can end a run early.
- Frames: one per decision, every {replay['step_s']:.1f} s. Shorter runs hold their final frame.
- Green car: ego agent. Road view: simulator truth. Map: sensor evidence, not ground truth.
- Actions label the interval ending at that frame. Reset begins with no measurements.
- GIFs hold the final outcome for two seconds before looping.
- This scenario is illustrative, not a representative estimate of policy performance.

Regenerate another scenario from the project root:

```bash
.venv/bin/python -m experiments.replay_agents --seed 100011 --weather clear
```
""")
    print(f"Open: {output / 'replay.html'}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--weather", choices=("clear", "rain", "fog", "night", "glare"), default="clear")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--duration", type=float, help="Override the episode deadline in seconds")
    parser.add_argument("--route-distance", type=float, help="Override the route distance in meters")
    args = parser.parse_args()
    export_replays(args.run, args.seed, args.weather, args.output,
                   duration_s=args.duration, route_distance_m=args.route_distance)


if __name__ == "__main__":
    main()
