#!/usr/bin/env python3
import json,pathlib,subprocess,sys,tempfile
def write(p,v): p.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def main():
  with tempfile.TemporaryDirectory(prefix="orbi-osm46c-") as td:
    r=pathlib.Path(td)
    seal={
      "schema_version":1,"stage":"windows-official-pilot-evidence-seal","seal_id":"abc",
      "source":{"model":"Qwen/Qwen3-Next-80B-A3B-Instruct","snapshot":"f5e99a3698d364cf77584543481b778afee26177"},
      "metrics":{
        "performance":{"cold_end_to_end_generated_tokens_per_second":2.5,"wall_time_ns":2_000_000_000,"peak_sampled_process_rss_bytes":123,"gpu_cache_resident_bytes":456},
        "phase_latency":{"prompt_prefill_ns":800,"decode_ns":200},
        "cache":{
          "prefill":{"gpu_hits":1,"gpu_misses":3,"host_hits":1,"host_misses":3},
          "decode":{"gpu_hits":8,"gpu_misses":2,"host_hits":6,"host_misses":4},
          "totals":{}
        }
      },
      "claims":{"real_windows_pilot_executed":True,"performance_target_met":False}
    }
    sp=r/"seal.json"; op=r/"baseline.json"; write(sp,seal)
    tool=pathlib.Path(__file__).with_name("analyze_qwen_windows_pilot.py")
    subprocess.run([sys.executable,str(tool),"--seal",str(sp),"--output",str(op)],check=True)
    out=json.loads(op.read_text(encoding="utf-8"))
    if "prefill_latency_share_high" not in out["descriptive_flags"]: raise RuntimeError("prefill flag missing")
    if out["claims"]["causal_bottleneck_proven"] is not False: raise RuntimeError("causality boundary crossed")
    subprocess.run([sys.executable,str(tool),"--seal",str(sp),"--output",str(op)],check=True)
    bad=dict(seal); bad["claims"]=dict(seal["claims"]); bad["claims"]["real_windows_pilot_executed"]=False
    bp=r/"bad.json"; write(bp,bad)
    p=subprocess.run([sys.executable,str(tool),"--seal",str(bp),"--output",str(r/"bad-out.json")])
    if p.returncode==0: raise RuntimeError("accepted unexecuted pilot")
    print("OSM-46C pilot optimization baseline: PASS")
  return 0
if __name__=="__main__": raise SystemExit(main())
