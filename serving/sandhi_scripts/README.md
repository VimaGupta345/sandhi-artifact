1. Run the Docker container by running the following command. Change the container_name and gpus as required. 
```
docker run --rm -it  --runtime nvidia --name <container_name> \
    --gpus '"device=0,1"' \
    --ipc=host \
    -p 8000:8000 \
    -v /scratch/shared_dir/hf_cache:/root/.cache/huggingface \
    --entrypoint /bin/bash \
    sandhi:latest
```

2. In another terminal, copy the scripts from the host to the container.
```
docker cp vllm_merged_model/sandhi_scripts/ <container_name>:/vllm-workspace/
```

3. In the container, run the following commands. You can use any of the config files already present in the `sandhi_scripts` directory or create a new one.
```
cd sandhi_scripts/
bash run_all.sh --config <config_file> --run-base-dir /vllm-workspace/<result_dir_name>
```

4. The logs and results will be stored in `<result_dir_name>`.

5. To view the logs while the scripts are running, you can connect to the container from another terminal using the following command and view the logs in `<result_dir_name>`.
```
docker exec -it <container_name> /bin/bash
```
