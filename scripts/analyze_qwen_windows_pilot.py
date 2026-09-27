#!/usr/bin/env python3
import argparse, json, pathlib

def load(path):
    v=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(v,dict): raise RuntimeError("expected JSON object")
    return v

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seal",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    seal=load(pathlib.Path(a.seal))
    if seal.get("schema_version")!=1 or seal.get("stage")!="windows-official-pilot-evidence-seal":
        raise RuntimeError("not an OSM-46B seal")
    claims=seal.get("claims",{})
    if claims.get("real_windows_pilot_executed") is not True:
        raise RuntimeError("real pilot execution not sealed")
    if claims.get("performance_target_met") is not False:
        raise RuntimeError("OSM-46B claim boundary changed")

    metrics=seal.get("metrics",{})
    perf=metrics.get("performance",{})
    lat=metrics.get("phase_latency",{})
    cache=metrics.get("cache",{})
    prefill=cache.get("prefill",{})
    decode=cache.get("decode",{})

    required=[
      ("cold_end_to_end_generated_tokens_per_second",perf),
      ("wall_time_ns",perf),
      ("prompt_prefill_ns",lat),
      ("decode_ns",lat),
    ]
    for key,obj in required:
        if obj.get(key) is None: raise RuntimeError(f"missing metric: {key}")

    total_phase=lat["prompt_prefill_ns"]+lat["decode_ns"]
    prefill_share=(lat["prompt_prefill_ns"]/total_phase) if total_phase else None
    decode_share=(lat["decode_ns"]/total_phase) if total_phase else None

    observations=[]
    if prefill_share is not None and prefill_share>=0.6:
        observations.append("prefill_latency_share_high")
    elif decode_share is not None and decode_share>=0.6:
        observations.append("decode_latency_share_high")
    else:
        observations.append("prefill_decode_latency_balanced")

    for phase_name,phase in (("prefill",prefill),("decode",decode)):
        gh=phase.get("gpu_hits",0); gm=phase.get("gpu_misses",0)
        hh=phase.get("host_hits",0); hm=phase.get("host_misses",0)
        if gh+gm:
            rate=gh/(gh+gm)
            if rate<0.5: observations.append(f"{phase_name}_gpu_hit_rate_below_half")
        if hh+hm:
            rate=hh/(hh+hm)
            if rate<0.5: observations.append(f"{phase_name}_host_hit_rate_below_half")

    payload={
      "schema_version":1,
      "stage":"windows-pilot-optimization-baseline",
      "source_seal":{
        "seal_id":seal.get("seal_id"),
        "model":seal.get("source",{}).get("model"),
        "snapshot":seal.get("source",{}).get("snapshot"),
      },
      "observed":{
        "cold_end_to_end_generated_tokens_per_second":perf["cold_end_to_end_generated_tokens_per_second"],
        "wall_time_ns":perf["wall_time_ns"],
        "prompt_prefill_ns":lat["prompt_prefill_ns"],
        "decode_ns":lat["decode_ns"],
        "prefill_latency_share":prefill_share,
        "decode_latency_share":decode_share,
        "peak_sampled_process_rss_bytes":perf.get("peak_sampled_process_rss_bytes"),
        "gpu_cache_resident_bytes":perf.get("gpu_cache_resident_bytes"),
        "prefill_cache":prefill,
        "decode_cache":decode,
      },
      "descriptive_flags":observations,
      "claims":{
        "descriptive_baseline_created":True,
        "causal_bottleneck_proven":False,
        "optimization_recommendation_certified":False,
        "representative_steady_state_performance_certified":False,
        "performance_target_met":False,
      }
    }
    out=pathlib.Path(a.output)
    rendered=json.dumps(payload,indent=2,sort_keys=True)+"\n"
    if out.exists() and out.read_text(encoding="utf-8")!=rendered:
        raise RuntimeError(f"existing OSM-46C baseline conflicts: {out}")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(rendered,encoding="utf-8")
    print(rendered,end="")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
