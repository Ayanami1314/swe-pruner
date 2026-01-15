## Install & Configure
```bash
uv sync
```

## Run Swebench
We list our experiment configs in `templates` directory.
You can run with the command:
```
mini-extra swebench --subset verified --split test --output runs/pruner --workers 8 -c ./templates/pruner.yaml
```
And don't forget to edit the `pruner.yaml` to give your api configuration and swe-pruner service configuration first.

## Get results
The result file structure is as same as the origin mini-swe-agent.

## View statistics
If you want to reproduce our results, we provide our analysis script `stats.py`, it might need some extra dependencies like plotly or scipy, just install if need.

To view the single experiment statistics, you can run:
```
python stats.py main <traj_dir>
```
where `traj_dir` has structure like
```
trajs_dir
├── astropy__astropy-12907
├── astropy__astropy-13033
```

To compare pruner and baseline, run:
```
python stats.py compare trajs-mini-glm4.6 pruner-glm4.6 --label1 Baseline --label2 SWE-Pruner -p glm-comp.png
```
