import json

dev = json.load(open("experiments/LAB-2084/out/dev_6sat.json"))
print("=== PLACER CHECK (seed 101) ===")
for r in dev["placer_check"]:
    st = r["Static"]
    cf = r["Dyn-CF"]
    print(f"{r['scenario_id']:10s} | {r['placer']:14s} | Static L={st['good_long_frac']*100:5.1f}% GP={st['goodput_tok_s']:6.1f} | Dyn-CF L={cf['good_long_frac']*100:5.1f}% GP={cf['goodput_tok_s']:6.1f}")

logs = json.load(open("experiments/LAB-2084/out/sample_logs_seed142.json"))
print("\n=== SAMPLE LOGS (seed 142) ===")
for item in logs:
    sc = item["scenario_id"]
    dlog = item["decision_log_dyn"]
    migs = [d for d in dlog if d["chosen_dst"] is not None]
    ev_mig = [e for e in item["events_by_arm"]["Dyn-CF"] if "mig" in str(e[1])]
    print(f"{sc}: decision_triggers={len(dlog)}, chosen_migs={len(migs)}, mig_events={len(ev_mig)}")
    if dlog:
        print("  first decision:", dlog[0])
    if ev_mig:
        print("  first 2 mig events:", ev_mig[:2])
