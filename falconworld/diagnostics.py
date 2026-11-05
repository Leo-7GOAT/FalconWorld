"""Frozen-data diagnostics: recursive intervals, action branches, sensor ablation.

Calibration uses validation episodes only. Statistical summaries treat episodes,
not neighboring frames or overlapping windows, as the independent sampling unit.
"""
import csv
import json
from pathlib import Path
import numpy as np
from .data import DT, HISTORY, load_dataset, make_episode, scenario_lead_accel, step_truth
from .models import load_models, estimate_sensor_state

HORIZONS = (5, 10, 30)
COMPONENTS = (("gap", "m"), ("ego_speed", "mps"), ("leader_speed", "mps"), ("actuator", "mps2"))


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def recursive_trajectory(model, state, actions, history):
    """Every recursive step sees only its own preceding predictions."""
    members = getattr(model, "members", None)
    if members:
        # Match online MPC: average completed independent member rollouts, rather
        # than feeding the ensemble mean back into every nonlinear member.
        actions = np.asarray(actions)
        return np.mean([recursive_trajectory(member, state, actions, history)
                        for member in members], axis=0)
    s, h = np.array(state, copy=True), np.array(history, copy=True)
    trajectory = []
    for action in actions:
        s = model.step(s, action, h)
        trajectory.append(s.copy())
        h = np.concatenate([h[1:], s[None]])
    return np.asarray(trajectory)


def recursive_members(ensemble, state, actions, history):
    """Members do not share the ensemble mean history: each has its own rollout."""
    trajectories = np.asarray([recursive_trajectory(m, state, actions, history) for m in ensemble.members])
    return trajectories.mean(0), trajectories.std(0), trajectories


def _windows(arrays):
    for episode in np.unique(arrays["episode"]):
        indices = np.flatnonzero(arrays["episode"] == episode)
        for offset in range(0, len(indices)-max(HORIZONS)+1, max(HORIZONS)):
            yield int(episode), offset, indices[offset:offset+max(HORIZONS)]


def calibrate_intervals(ensemble, validation):
    """90th percentile across episode maximum errors, independently per horizon/state.

    This empirical validation calibration is deliberately labeled heuristic. It is
    not a finite-sample conformal guarantee or a calibrated interpretation of std.
    """
    excess = {}
    for episode, _, index in _windows(validation):
        mean, std, _ = recursive_members(ensemble, validation["state"][index[0]],
                                         validation["action"][index], validation["history"][index[0]])
        err = np.abs(mean-validation["next_state"][index])
        score = np.maximum(0, err-1.645*std)[np.asarray(HORIZONS)-1]
        excess[episode] = np.maximum(excess.get(episode, np.zeros_like(score)), score)
    if not excess:
        raise ValueError("Validation requires complete 3-second windows")
    radius = np.quantile(np.asarray(list(excess.values())), .9, axis=0)
    return radius, {"calibration_episodes": len(excess), "horizons_s": [h*DT for h in HORIZONS],
                    "radius_by_horizon_state": radius.tolist(),
                    "formula": "mean +/- (validation episode-max error excess q90 + 1.645*member rollout std)",
                    "scope": "empirical validation heuristic; no finite-sample, OOD or safety guarantee",
                    "member_recursion": "each member uses its own predicted state and history"}


def _mean_ci(values, seed=88):
    x = np.asarray(values, dtype=float)
    r = np.random.default_rng(seed)
    samples = r.choice(x, size=(2000, len(x)), replace=True).mean(1)
    return float(x.mean()), [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]


def interval_evaluation(ensemble, data, radius, output):
    windows, episodes, summaries = [], [], []
    for split in ("test", "ood"):
        arrays = data[split]
        for episode, offset, index in _windows(arrays):
            mean, std, _ = recursive_members(ensemble, arrays["state"][index[0]], arrays["action"][index],
                                             arrays["history"][index[0]])
            for hi, h in enumerate(HORIZONS):
                width = 2*(radius[hi]+1.645*std[h-1])
                errors = np.abs(mean[h-1]-arrays["next_state"][index[h-1]])
                for j, (component, unit) in enumerate(COMPONENTS):
                    windows.append({"split": split, "episode": episode, "family": str(arrays["family"][index[0]]),
                                    "offset": offset, "horizon_s": h*DT, "component": component, "unit": unit,
                                    "covered": int(errors[j] <= width[j]/2 + 1e-12),
                                    "width": float(width[j]), "absolute_error": float(errors[j]),
                                    "member_std": float(std[h-1, j])})
        split_windows = [row for row in windows if row["split"] == split]
        for ep in np.unique(arrays["episode"]):
            for h in HORIZONS:
                for component, unit in COMPONENTS:
                    rows = [row for row in split_windows if row["episode"] == ep and row["horizon_s"] == h*DT and row["component"] == component]
                    if rows:
                        episodes.append({"split": split, "episode": int(ep), "horizon_s": h*DT,
                                         "component": component, "unit": unit, "windows": len(rows),
                                         "coverage": float(np.mean([row["covered"] for row in rows])),
                                         "mean_width": float(np.mean([row["width"] for row in rows])),
                                         "mae": float(np.mean([row["absolute_error"] for row in rows]))})
        for h in HORIZONS:
            for component, unit in COMPONENTS:
                rows = [row for row in episodes if row["split"] == split and row["horizon_s"] == h*DT and row["component"] == component]
                coverage, ci = _mean_ci([row["coverage"] for row in rows])
                width, wci = _mean_ci([row["mean_width"] for row in rows])
                summaries.append({"split": split, "horizon_s": h*DT, "component": component, "unit": unit,
                                  "episodes": len(rows), "episode_mean_coverage": coverage, "coverage_bootstrap95": ci,
                                  "episode_mean_width": width, "width_bootstrap95": wci,
                                  "episode_mean_mae": float(np.mean([row["mae"] for row in rows]))})
    _csv(output / "interval_windows.csv", windows)
    _csv(output / "interval_episodes.csv", episodes)
    _json(output / "interval_summary.json", summaries)
    return summaries


def sensor_history(history):
    """Strip true actuator and reconstruct causally from available speed history.

    Initial repeat padding produces a zero estimate. At standstill hidden braking
    remains unidentifiable. This idealized ablation includes no measurement noise.
    """
    h = np.asarray(history, dtype=float).copy()
    if h.ndim != 2 or h.shape[1] not in (3, 4):
        raise ValueError("Expected one history of observations")
    out = np.column_stack([h[:, :3], np.zeros(len(h))])
    for k in range(1, len(h)):
        out[k] = estimate_sensor_state(h[k, :3], h[:k+1, :3])
    return out


def sensor_pair_evaluation(models, data, output):
    raw, summary = [], []
    for split in ("test", "ood"):
        arrays = data[split]
        for ep, offset, index in _windows(arrays):
            h_priv = arrays["history"][index[0]]
            h_sensor = sensor_history(h_priv)
            for name, model in models.items():
                p = recursive_trajectory(model, arrays["state"][index[0]], arrays["action"][index], h_priv)
                s = recursive_trajectory(model, h_sensor[-1], arrays["action"][index], h_sensor)
                for h in HORIZONS:
                    for j, (component, unit) in enumerate(COMPONENTS):
                        pe, se = abs(p[h-1,j]-arrays["next_state"][index[h-1],j]), abs(s[h-1,j]-arrays["next_state"][index[h-1],j])
                        raw.append({"split": split, "episode": ep, "offset": offset, "model": name,
                                    "horizon_s": h*DT, "component": component, "unit": unit,
                                    "privileged_error": float(pe), "sensor_error": float(se), "paired_delta": float(se-pe)})
        for name in models:
            for h in HORIZONS:
                for component, unit in COMPONENTS:
                    rows = [row for row in raw if row["split"] == split and row["model"] == name and row["horizon_s"] == h*DT and row["component"] == component]
                    ids = sorted(set(row["episode"] for row in rows))
                    epmeans = [{key: np.mean([row[key] for row in rows if row["episode"] == ep])
                                for key in ("privileged_error", "sensor_error", "paired_delta")} for ep in ids]
                    effect, ci = _mean_ci([row["paired_delta"] for row in epmeans])
                    summary.append({"split": split, "model": name, "horizon_s": h*DT, "component": component, "unit": unit,
                                    "episodes": len(ids), "privileged_mae": float(np.mean([row["privileged_error"] for row in epmeans])),
                                    "sensor_mae": float(np.mean([row["sensor_error"] for row in epmeans])),
                                    "sensor_minus_privileged": effect, "paired_episode_bootstrap95": ci})
    _csv(output / "sensor_paired_windows.csv", raw)
    _json(output / "sensor_paired_summary.json", summary)
    return summary


def counterfactual_evaluation(models, data, output):
    arrays = data["test"]
    metadata = {ep["episode_id"]: ep for ep in data["metadata"]["episodes"] if ep["split"] == "test"}
    traces, errors, figure_data = [], [], []
    for family in ("constant", "mild", "brake", "stopgo"):
        ep_id = int(np.min(arrays["episode"][arrays["family"] == family]))
        index = np.flatnonzero(arrays["episode"] == ep_id)
        meta = metadata[ep_id]
        params = meta.get("environment_params")
        if params is None:
            # v1's first manifest stored seeds; generator is deterministic and unchanged.
            params = make_episode(family, meta["seed"], meta["condition"])["params"]
        for offset in (0, 30):
            if len(index) <= offset:
                continue
            initial = arrays["state"][index[offset]].copy()
            hist = sensor_history(arrays["history"][index[offset]])
            # Truth starts from actual latent state; predictors get identical sensor estimate.
            for action in (-4., 0., 2.):
                truth, current = [], initial.copy()
                for k in range(30):
                    lead = scenario_lead_accel(family, (offset+k)*DT, params)
                    current = step_truth(current, action, lead, "id")
                    truth.append(current.copy())
                    if current[0] <= 0:
                        break
                truth = np.asarray(truth)
                predictions = {name: recursive_trajectory(model, hist[-1], np.full(len(truth), action), hist)
                               for name, model in models.items()}
                if offset == 30:
                    figure_data.append((family, action, truth, predictions))
                for name, prediction in predictions.items():
                    for k, (p, t) in enumerate(zip(prediction, truth)):
                        row = {"family": family, "episode": ep_id, "offset": offset, "command_mps2": action,
                               "model": name, "horizon_s": (k+1)*DT, "truth_collision": int(t[0] <= 0)}
                        for j, (component, unit) in enumerate(COMPONENTS):
                            row[f"truth_{component}_{unit}"] = float(t[j])
                            row[f"predicted_{component}_{unit}"] = float(p[j])
                        traces.append(row)
                    for h in HORIZONS:
                        if h <= len(truth):
                            err = abs(prediction[h-1]-truth[h-1])
                            errors.append({"family": family, "episode": ep_id, "offset": offset, "command_mps2": action,
                                           "model": name, "horizon_s": h*DT,
                                           **{f"{c}_mae_{unit}": float(err[j]) for j, (c, unit) in enumerate(COMPONENTS)}})
    _csv(output / "counterfactual_trajectories.csv", traces)
    _csv(output / "counterfactual_errors.csv", errors)
    summaries = []
    for name in models:
        for action in (-4., 0., 2.):
            for h in HORIZONS:
                rows = [row for row in errors if row["model"] == name and row["command_mps2"] == action and row["horizon_s"] == h*DT]
                if rows:
                    summaries.append({"model": name, "command_mps2": action, "horizon_s": h*DT,
                                      "scenario_initializations": len(rows),
                                      **{f"{c}_mae_{unit}": float(np.mean([row[f"{c}_mae_{unit}"] for row in rows]))
                                         for c, unit in COMPONENTS}})
    _json(output / "counterfactual_summary.json", summaries)
    _plot_branches(figure_data, output / "counterfactual_branches.png")
    return {"scenario_instances": 8, "actions": [-4, 0, 2], "prediction_access": "same sensor-derived initial state/history",
            "truth_only_access": "future exogenous leader script; true actuator initial state",
            "collision_policy": "truth branches terminate at first nonpositive gap; unavailable horizons omitted",
            "purpose": "counterfactual ego-action sensitivity, not learned causal interaction with exogenous leader",
            "raw_prediction_rows": len(traces), "error_rows": len(errors)}


def _plot_branches(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = {-4.: "tab:blue", 0.: "tab:orange", 2.: "tab:green"}
    fig, axes = plt.subplots(4, 2, figsize=(11, 11), sharex=True)
    for fi, family in enumerate(("constant", "mild", "brake", "stopgo")):
        for f, action, truth, predictions in data:
            if f != family:
                continue
            t = np.arange(1, len(truth)+1)*DT
            for ax, j in zip(axes[fi], (0, 1)):
                ax.plot(t, truth[:, j], color=colors[action], lw=2)
                for name, style in (("physical_enriched", ":"), ("history", "--")):
                    if name in predictions:
                        ax.plot(t, predictions[name][:, j], style, color=colors[action], lw=1.2)
                ax.grid(alpha=.2)
        axes[fi, 0].set_ylabel(f"{family}\ngap [m]")
        axes[fi, 1].set_ylabel("ego speed [m/s]")
    for ax in axes[-1]:
        ax.set_xlabel("prediction horizon [s]")
    legend = [Line2D([0],[0],color=c,label=f"command {a:g} m/s²") for a,c in colors.items()]
    legend += [Line2D([0],[0],color="black",ls=style,label=name) for name,style in (("environment", "-"),("enriched physics", ":"),("history model", "--"))]
    fig.legend(handles=legend, ncol=3, loc="upper center")
    fig.suptitle("Independent action branches from the same sensor history (test episodes, t=3s)", y=.93)
    fig.tight_layout(rect=(0,0,1,.91)); fig.savefig(path, dpi=150); plt.close(fig)


def run_diagnostics(data_dir, model_dir, output_dir):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    data, models = load_dataset(data_dir), load_models(model_dir)
    radius, calibration = calibrate_intervals(models["ensemble"], data["val"])
    _json(output / "interval_calibration.json", calibration)
    intervals = interval_evaluation(models["ensemble"], data, radius, output)
    sensors = sensor_pair_evaluation(models, data, output)
    branches = counterfactual_evaluation(models, data, output)
    summary = {"data_sha256": data["metadata"]["sha256"], "model_directory": str(Path(model_dir).resolve()),
               "calibration_split": "val", "evaluation_splits": ["test", "ood"],
               "statistical_unit": "episode; windows aggregated before bootstrap", "bootstrap_samples": 2000,
               "sensor_mode": "ideal noiseless observed gap/speeds; causal speed difference only; hidden actuator ignored",
               "interval_summary": intervals, "sensor_summary": sensors, "counterfactual": branches}
    _json(output / "summary.json", summary)
    lines = ["# 递归不确定性、动作分支与传感观测诊断", "",
             f"数据摘要 `{summary['data_sha256']}`。模型目录 `{summary['model_directory']}`。",
             "",
             "区间在 val 上按独立 episode 的最大超额误差取 90% 分位数，测试前固定。每个集成成员递归自己的预测状态和历史；绝不把均值轨迹作为各成员共同历史。测试覆盖和宽度先按 episode 汇总，再 bootstrap 2000 次。结果是经验覆盖，不是校准置信度、OOD 或安全保证。",
             "", "|集合|时域(s)|间距覆盖|间距宽度(m)|自车速度覆盖|速度宽度(m/s)|", "|---|---:|---:|---:|---:|---:|"]
    for split in ("test", "ood"):
        for h in HORIZONS:
            g = next(row for row in intervals if row["split"]==split and row["horizon_s"]==h*DT and row["component"]=="gap")
            v = next(row for row in intervals if row["split"]==split and row["horizon_s"]==h*DT and row["component"]=="ego_speed")
            lines.append(f"|{split}|{h*DT:g}|{g['episode_mean_coverage']:.3f}|{g['episode_mean_width']:.3g}|{v['episode_mean_coverage']:.3f}|{v['episode_mean_width']:.3g}|")
    lines += ["", "ID 自车动力学被强物理基线精确描述，三个 residual 成员共享这一结构，故自车速度区间可窄至浮点量级。OOD 改变时常后，同样的成员分歧不能识别共同结构误差：极低覆盖直接暴露了失效边界。3 s 间距高覆盖还伴随较宽区间，不能单凭覆盖率认定风险信息有用。"]
    lines += ["", "动作分支从固定 test episode 的 t=0/3s 相同传感历史出发，分别恒定施加 -4/0/2 m/s²。truth 独立运行且仅 truth 读取前车脚本；预测仅读取可观测历史和候选动作。图和 CSV 包含预测、真实分支及误差。前车是外生的，因此不能据此声称学到了自车对前车行为的因果作用。",
              "", "![动作分支](counterfactual_branches.png)", "",
              "传感消融把当前及历史的真实 a_actuator 全部删除，再以过去速度差分估计上一时刻加速度。所有模型共用估计器。停车时恒零速度不能辨识内部负制动，因此误差可能增加。该实验只验证理想无噪声观测的开环敏感性，尚未证明传感器噪声、延迟或实车闭环能力。逐 episode 配对结果见 sensor_paired_summary.json。"]
    (output / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary
