# OSM-46C — designated-host pilot review / optimization baseline

Status: **IMPLEMENTED / STACKED CI CERTIFICATION PENDING**

OSM-46C consumes a valid OSM-46B seal and creates a descriptive baseline for
future optimization work.

It records observed cold output rate, wall time, prefill/decode latency shares,
sampled RSS, GPU cache residency, and prefill/decode cache counters.

The tool may emit descriptive flags such as a high prefill share or a cache hit
rate below one half. These are observations only. They are not causal diagnoses.

OSM-46C explicitly keeps these claims false:

- causal bottleneck proven;
- optimization recommendation certified;
- representative steady-state performance certified;
- performance target met.

The next step after a real pilot is to use this baseline to design controlled
optimization experiments one variable at a time.
