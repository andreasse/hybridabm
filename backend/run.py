# run.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2024 Kart Padur

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# ---------------------------------------------------------------------------
#  ⚙️  Hybrid‑Article main driver  (stream-friendly, O(1)-RAM rewrite) (full original logic + perf & head‑less fixes)
# ---------------------------------------------------------------------------

import os, gzip, json, warnings
from datetime import datetime, timezone
import matplotlib, matplotlib.pyplot as plt
import numpy as np, pandas as pd, orjson

from timer import Timer
from experiment import experiment
from setup_values import (
    timestep, nTimesteps, nExperiments, nProviders, nAgents, nMalAgents, W,
    save_fig, output_dir, alpha, delta, epsilon, eta, theta, kappa, Lambda,
    lambda_ as lambda_arr, xi, rho, tau, Upsilon, upsilon_, experiment_id,
)

# Export every nth timestep
EXPORT_INTERVAL = 24

ORJSON_OPTS = (
    orjson.OPT_SERIALIZE_NUMPY
    | orjson.OPT_NAIVE_UTC
    | orjson.OPT_NON_STR_KEYS
)
def dumps(obj):          # keep it tiny
    return orjson.dumps(obj, option=ORJSON_OPTS)

def edge_id(src, tgt): 
    return f"{src}-{tgt}"

###############################################################################
# IO PATHS & CONSTANTS
###############################################################################
OUTPUT_DATA_DIR = os.path.join(os.getcwd(), "output_data")
os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
SUMMARY_CSV  = os.path.join(OUTPUT_DATA_DIR, f"experiment{experiment_id}_summary.csv")
FRAMES_JSONL = os.path.join(
    OUTPUT_DATA_DIR,
    f"frames_exp{experiment_id}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.jsonl.gz",
)

# header carries meta + events so the UI can find everything
HEADER_JSON = FRAMES_JSONL.replace("frames_", "header_").replace(".jsonl.gz", ".json")

if save_fig:
    os.makedirs(output_dir, exist_ok=True)

# Head‑less backend detection – "Agg", "PDF", etc. never open a window ------
HEADLESS_BACKENDS = {"agg", "pdf", "svg"}
SHOW_FIGS = (not save_fig) and matplotlib.get_backend().lower() not in HEADLESS_BACKENDS

def _maybe_show():
    """Show figure only if interactive; always close to free memory."""
    if SHOW_FIGS:
        plt.show()
    plt.close()
# ---------------------------------------------------------------------------

# All dataframe columns (kept verbatim from original) -----------------------
COLUMNS = [
    "regret", "provider", "avguse", "avgreward", "posopinion", "negopinion",
    "avgusebefore", "avguseduring", "avguseafter", "avgrewbefore", "avgrewduring",
    "avgrewafter", "posopinionbefore", "posopinionduring", "posopinionafter",
    "negopinionbefore", "negopinionduring", "negopinionafter", "target_id",
    "attack_beginning", "misinfo_beginning", "misinfo_length", "cyber_beginning",
    "cyber_length", "combined_beginning", "combined_length", "impactusage",
    "impactposop", "impactnegop", "attackreward", "detectionreward", "malreward",
    "alpha", "delta", "epsilon", "eta", "theta", "kappa", "Lambda", "lambda", "xi",
    "rho", "tau", "Upsilon", "upsilon",
]
# ---------------------------------------------------------------------------

run = 1  # You can bump this to run the same parameter set multiple times

for _run_idx in range(run):
    timer = Timer(); timer.start()

    rows:   list[dict] = []       # provider-level summary rows
    events: list[dict] = []       # attack timeline
    jsonl  = gzip.open(FRAMES_JSONL, "wb")   # ← stream frames here

    # ── MAIN EXPERIMENT LOOP ───────────────────────────────────────────────
    for exp in range(nExperiments):
        print(f"[Experiment {exp + 1}/{nExperiments}]")

        actions, rewards, opinionvalues, regrets, attack_decisions, targets, \
        attack_methods, attack_rewards, detection_rewards, links, cyber_matrix = experiment(timestep, nTimesteps)

        # Average regret after warm‑up window W
        avgregret = np.mean(regrets[W + 1:]) / nAgents

        # ── STREAM -- one JSONL per timestep with nodes and edge deltas ──
        prev_edges = set()
        accumulated_adds = set()
        accumulated_removes = set()
        
        for t in range(nTimesteps):
            # Build current edges for this timestep
            current_edges = set()
            for agent in range(nAgents):
                provider = cyber_matrix[t, agent]
                if provider != -1:  # agent is connected to a provider
                    edge = (str(agent), f"P-{provider}")
                    current_edges.add(edge)
            
            # Track changes since last export
            newly_added = current_edges - prev_edges
            newly_removed = prev_edges - current_edges
            
            # Accumulate changes, handling cancellations
            for edge in newly_added:
                if edge in accumulated_removes:
                    # Cancel out: edge was removed then re-added
                    accumulated_removes.discard(edge)
                else:
                    accumulated_adds.add(edge)
            
            for edge in newly_removed:
                if edge in accumulated_adds:
                    # Cancel out: edge was added then removed
                    accumulated_adds.discard(edge)
                else:
                    accumulated_removes.add(edge)
            
            prev_edges = current_edges
            
            # Export on interval
            if t % EXPORT_INTERVAL == 0:
                # Build nodes for this export timestep - use actual agent IDs from cyber_matrix
                nodes_for_frame = []
                
                # Add agent nodes
                for agent in range(nAgents):
                    provider = cyber_matrix[t, agent]
                    if provider != -1:  # agent is connected to a provider
                        # Calculate reward for this agent's provider
                        n_here = int(actions[t, provider])
                        r_per_user = rewards[t, provider] / n_here if n_here > 0 else 0
                        
                        nodes_for_frame.append({
                            "id": str(agent),
                            "prov": provider,
                            "reward": r_per_user,
                            "mal": int(agent < nMalAgents),
                            "opin": opinionvalues[t, provider + 3] if (provider + 3) < opinionvalues.shape[1] else 0
                        })
                
                # Add provider nodes
                for prov in range(nProviders):
                    nodes_for_frame.append({
                        "id": f"P-{prov}",
                        "prov": -1,    # Special marker for provider nodes
                        "reward": 1.0,  # Frontend expects providers to have reward 1.0
                        "mal": 0,
                        "opin": 0
                    })
                
                # Convert accumulated changes to JSON format
                edge_add = [{"id": edge_id(s, tgt), "source": s, "target": tgt, "type": "cyber"}
                           for s, tgt in accumulated_adds]
                edge_remove = [{"id": edge_id(s, tgt), "source": s, "target": tgt, "type": "cyber"}
                              for s, tgt in accumulated_removes]
                
                # Write one JSONL line per frame
                jsonl.write(dumps({
                    "t": t + 1,
                    "nodes": nodes_for_frame,
                    "edgeDeltas": {"add": edge_add, "remove": edge_remove}
                }) + b"\n")
                
                # Reset accumulators after export
                accumulated_adds.clear()
                accumulated_removes.clear()

        # --- Attack meta extraction --------------------------------------
        attack_happened = nMalAgents > 0 and np.any(attack_decisions == 1)
        if attack_happened:
            attack_ts = np.flatnonzero(attack_decisions)
            target_provider = targets[attack_ts[0]]

            misinfo_ts  = np.flatnonzero(((attack_methods == 1) | (attack_methods == 2)) & attack_decisions)
            cyber_ts    = np.flatnonzero(((attack_methods == 0) | (attack_methods == 2)) & attack_decisions)
            combined_ts = np.flatnonzero((attack_methods == 2) & attack_decisions)

            misinfo_len, cyber_len, combined_len = len(misinfo_ts), len(cyber_ts), len(combined_ts)
            misinfo_begin = misinfo_ts.min() + 1 if misinfo_len else ""
            cyber_begin   = cyber_ts.min() + 1 if cyber_len else ""
            comb_begin    = combined_ts.min() + 1 if combined_len else ""

            # Malicious reward windows -----------------------------------
            reward_ts = np.append(attack_ts[1:], attack_ts[-1] + (attack_ts[-1] != nTimesteps - 1))
            attackreward    = attack_rewards[reward_ts].sum()
            detectionreward = detection_rewards[reward_ts].sum()
            malreward       = attackreward + detectionreward

            print(
                f"Target: {target_provider + 1} , attack timesteps: {attack_ts} , "
                f"length of attack campaign: {attack_ts.size} , misinfo campaign: {misinfo_ts.tolist()} , "
                f"cyber campaign: {cyber_ts.tolist()}"
            )

            # ── Build events list from ground-truth arrays ───────────────
            typ_map = {0: "CYBER", 1: "MISINFO", 2: "COMBO"}
            last_typ = None
            for t_step in np.flatnonzero(attack_decisions):
                typ = typ_map[int(attack_methods[t_step])]
                if typ != last_typ:
                    events.append({"t": int(t_step) + 1, "type": typ})
                    last_typ = typ
        # ------------------------------------------------------------------

        # Provider‑level metrics -------------------------------------------
        for provider in range(nProviders):
            # Averages after warm‑up
            avguse = np.mean(actions[W + 1:, provider]) / nAgents
            denom_total = np.sum(actions[W + 1:, provider])
            avgreward = 0 if denom_total == 0 else np.sum(rewards[W + 1:, provider]) / denom_total
            posopinion = np.mean(opinionvalues[W + 1:, provider + 3])
            negopinion = np.mean(opinionvalues[W + 1:, provider])

            if attack_happened:
                # Slices before / during / after attack of equal length to attack window
                win = attack_ts.size
                before_sl = slice(max(0, attack_ts[0] - win), attack_ts[0])
                during_sl = attack_ts
                after_sl  = slice(attack_ts[-1], min(nTimesteps, attack_ts[-1] + win))

                def _mean_safe(arr):
                    return float(arr.mean()) if arr.size else 0

                avgusebefore = np.mean(actions[before_sl, provider]) / nAgents
                avguseduring = np.mean(actions[during_sl, provider]) / nAgents
                avguseafter  = np.mean(actions[after_sl,  provider]) / nAgents

                avgrewbefore = _mean_safe(rewards[before_sl, provider] / np.where(actions[before_sl, provider] == 0, 1, actions[before_sl, provider]))
                avgrewduring = _mean_safe(rewards[during_sl, provider] / np.where(actions[during_sl, provider] == 0, 1, actions[during_sl, provider]))
                avgrewafter  = _mean_safe(rewards[after_sl,  provider] / np.where(actions[after_sl,  provider] == 0, 1, actions[after_sl,  provider]))

                posopinionbefore = _mean_safe(opinionvalues[before_sl, provider + 3])
                posopinionduring = _mean_safe(opinionvalues[during_sl, provider + 3])
                posopinionafter  = _mean_safe(opinionvalues[after_sl,  provider + 3])
                negopinionbefore = _mean_safe(opinionvalues[before_sl, provider])
                negopinionduring = _mean_safe(opinionvalues[during_sl, provider])
                negopinionafter  = _mean_safe(opinionvalues[after_sl,  provider])

                impactusage = avguse - avguseduring
                impactposop = posopinion - posopinionduring
                impactnegop = abs(negopinion) - abs(negopinionduring)
            else:
                # Placeholders ------------------------------------------------
                (avgusebefore, avguseduring, avguseafter, avgrewbefore, avgrewduring, avgrewafter,
                 posopinionbefore, posopinionduring, posopinionafter, negopinionbefore,
                 negopinionduring, negopinionafter, impactusage, impactposop, impactnegop) = ("",) * 15

            # Row assembly ---------------------------------------------------
            rows.append({
                "regret": avgregret,
                "provider": provider + 1,
                "avguse": avguse,
                "avgreward": avgreward,
                "posopinion": posopinion,
                "negopinion": negopinion,
                "avgusebefore": avgusebefore,
                "avguseduring": avguseduring,
                "avguseafter": avguseafter,
                "avgrewbefore": avgrewbefore,
                "avgrewduring": avgrewduring,
                "avgrewafter": avgrewafter,
                "posopinionbefore": posopinionbefore,
                "posopinionduring": posopinionduring,
                "posopinionafter": posopinionafter,
                "negopinionbefore": negopinionbefore,
                "negopinionduring": negopinionduring,
                "negopinionafter": negopinionafter,
                "target_id": (target_provider + 1) if attack_happened else "",
                "attack_beginning": (attack_ts[0] + 1) if attack_happened else "",
                "misinfo_beginning": misinfo_begin if attack_happened else "",
                "misinfo_length": misinfo_len if attack_happened else "",
                "cyber_beginning": cyber_begin if attack_happened else "",
                "cyber_length": cyber_len if attack_happened else "",
                "combined_beginning": comb_begin if attack_happened else "",
                "combined_length": combined_len if attack_happened else "",
                "impactusage": impactusage,
                "impactposop": impactposop,
                "impactnegop": impactnegop,
                "attackreward": attackreward if attack_happened else "",
                "detectionreward": detectionreward if attack_happened else "",
                "malreward": malreward if attack_happened else "",
                # parameters – keep identical names -------------------------
                "alpha": alpha,
                "delta": delta,
                "epsilon": epsilon,
                "eta": eta,
                "theta": theta,
                "kappa": kappa,
                "Lambda": Lambda[provider],
                "lambda": lambda_arr[provider],
                "xi": xi,
                "rho": rho,
                "tau": tau,
                "Upsilon": Upsilon[provider],
                "upsilon": upsilon_[provider],
            })
    # ───────────────────────────────────────────────────────────────────────

    # Summary CSV is tiny → easy
    data_df = pd.DataFrame.from_records(rows, columns=COLUMNS)
    
    # ── SAVE PROVIDER-LEVEL SUMMARY (unchanged filename) ───────────────────
    summary_path = os.path.join(OUTPUT_DATA_DIR, f"experiment{experiment_id}_summary.csv")
    data_df.to_csv(summary_path, mode="a", header=False, index=False, encoding="utf-8")

    # ---- persist header so front-end sees meta + attack timeline ---------
    jsonl.close()          # flush the frame stream first


    # TEST_FRAMES = FRAMES_JSONL.replace('.jsonl.gz', '_small.jsonl.gz')
    # with gzip.open(FRAMES_JSONL, 'rt') as f_in:
    #     with gzip.open(TEST_FRAMES, 'wt') as f_out:
    #         for i, line in enumerate(f_in):
    #             if i < 10000:  # Only first 10k lines
    #                 f_out.write(line)
    #             else:
    #                 break
    # print(f"Created small test file: {TEST_FRAMES}")

    header_json = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "nAgents":    nAgents,
            "nTimesteps": nTimesteps,
            # add whatever params you want surfaced in the UI ↓
        },
        "summary_csv": os.path.basename(SUMMARY_CSV),
        "frames_gzip": os.path.basename(FRAMES_JSONL),
        "events": events,          # ← now actually persisted
        "links": [{"source": s, "target": t, "type": typ} for s, t, typ in links]
    }
    # std-lib json is fine (events are plain ints/strs)
    with open(HEADER_JSON, "w", encoding="utf-8") as fh:
        json.dump(header_json, fh)

    timer.stop()

    # ----------------------------------------------------------------------
    #  📊  Visualisations (identical logic, but head‑less safe) 
    # ----------------------------------------------------------------------

    plt.rcParams["figure.figsize"] = (20, 4)

    # Plot 1 – provider usage over time ------------------------------------
    A_over_time = (actions / nExperiments) * 100 / nAgents
    for prov in range(nProviders):
        ts = np.arange(1, len(A_over_time) + 1)
        plt.plot(ts, A_over_time[:, prov], "-", label=f"Provider {prov + 1}")
    plt.xlabel("Time steps")
    plt.ylabel("Average provider usage (%)")
    plt.xlim(1, nTimesteps)
    plt.ylim(1, 100)
    plt.grid()
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save_fig:
        plt.savefig(os.path.join(output_dir, "actions.pdf"), dpi=500, bbox_inches="tight")
    _maybe_show()

    # Plot 2 – average reward per provider ---------------------------------
    aveR_A = rewards / np.where(actions == 0, 1, actions)
    for prov in range(nProviders):
        ts = np.arange(1, len(aveR_A) + 1)
        plt.plot(ts, aveR_A[:, prov], ".", label=f"Provider {prov + 1}")
    plt.xlabel("Time steps")
    plt.ylabel("Average reward per service provider")
    plt.xlim(1, nTimesteps)
    plt.ylim(-0.01, 1.01)
    plt.grid()
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save_fig:
        plt.savefig(os.path.join(output_dir, "rewards.pdf"), dpi=500, bbox_inches="tight")
    _maybe_show()

    # Plot 3 – opinion dynamics -------------------------------------------
    for opinion in range(nProviders * 2):
        ts = np.arange(1, len(opinionvalues) + 1)
        style = "-" if opinion >= nProviders else "--"
        color = ["tab:blue", "tab:orange", "tab:green"][opinion % nProviders]
        sign  = "+" if opinion >= nProviders else "-"
        label = f"Provider {opinion % nProviders + 1} {sign}"
        plt.plot(ts, opinionvalues[:, opinion], style, color=color, label=label)
    plt.xlabel("Time steps")
    plt.ylabel("Average evaluation of opinion value per service provider")
    plt.xlim(1, nTimesteps)
    plt.grid()
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save_fig:
        plt.savefig(os.path.join(output_dir, "opinionvalues.pdf"), dpi=500, bbox_inches="tight")
    _maybe_show()

    # Plot 4 – malicious rewards ------------------------------------------
    ts = np.arange(1, len(attack_rewards) + 1)
    plt.plot(ts, attack_rewards, ":", label="Attack rewards")
    plt.plot(ts, detection_rewards, ":", label="Detection rewards")
    plt.xlabel("Time steps")
    plt.ylabel("Malicious users' rewards")
    plt.xlim(1, nTimesteps)
    plt.grid()
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save_fig:
        plt.savefig(os.path.join(output_dir, "malrewards.pdf"), dpi=500, bbox_inches="tight")
    _maybe_show()

    print("✅  Run {}/{} finished in {:.2f} s.\n"
          "   ↳ summary CSV   → {}\n"
          "   ↳ frames JSONL  → {}\n"
          "   ↳ header JSON   → {}"
          .format(_run_idx + 1, run, timer.elapsed,
                  SUMMARY_CSV, FRAMES_JSONL, HEADER_JSON))