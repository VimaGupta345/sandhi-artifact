# SANDHI serving benchmark harness — how to run

Full scenario descriptions, requirements, and the config table are in
[`../README.md`](../README.md). The steps below run one deployment scenario
end to end.

1. Pull the prebuilt serving image and tag it with the short name used below:

```bash
docker pull nandanmeda1999/sandhi-inference:latest
docker tag  nandanmeda1999/sandhi-inference:latest sandhi:latest
```

2. Start the container. Adjust `--gpus` to the scenario (1 GPU for the
   single-pool scenarios, 2 for the cross-family pools) and point the volume
   at a persistent Hugging Face cache directory on the host so model downloads
   survive container restarts:

```bash
docker run --rm -it --runtime nvidia --name sandhi_eval \
    --gpus '"device=0,1"' \
    --ipc=host \
    -p 8000:8000 \
    -v /path/to/hf_cache:/root/.cache/huggingface \
    --entrypoint /bin/bash \
    sandhi:latest
```

3. From the host, copy the harness **and the merge specs** into the container
   (run from the repository root):

```bash
docker cp serving/sandhi_scripts/ sandhi_eval:/vllm-workspace/
docker cp serving/specs/. sandhi_eval:/vllm-workspace/sandhi_scripts/
```

4. Inside the container, run one scenario. Use any of the `*_config.sh` files
   in this directory (one per deployment scenario; see the table in
   `../README.md`), or copy `template_config.sh` to create a new pool:

```bash
cd /vllm-workspace/sandhi_scripts
bash run_all.sh --config <config_file> --run-base-dir /vllm-workspace/<result_dir>
```

   `run_all.sh` does everything for the scenario: starts the ballast
   allocators, launches one vLLM server per model in baseline mode, sweeps the
   configured request rates, restarts the servers in sandhi mode, repeats the
   sweep, and renders the comparison plots.

5. Logs and results land in `<result_dir>`: server logs in `logs/servers/`,
   benchmark sweeps in `logs/benchmarks/`, plots and parsed metrics in
   `results/`.

6. To watch progress while a run is going, attach from another terminal:

```bash
docker exec -it sandhi_eval /bin/bash
```
