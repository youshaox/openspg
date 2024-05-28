# APP: An Adaptive Placement and Parallelism Framework for Accelerating RLHF Training
We present our implementation for the research paper: "APP: An Adaptive Placement and Parallelism Framework for Accelerating RLHF Training." 
At present, we have released the majority of the APP framework's source code. Due to the significant effort required to decouple the framework from our proprietary systems, we have not yet released the complete codebase. However, our team is diligently working to prepare the entire APP framework for open-source release.

# 🏃 How to train RLHF in APP framework
## Run Interleaving Strategy
```shell
cd python/nn4k/nn4k/alignment/examples/interleaving
export PLACEMENT_STRATEGY=interleaving; python app_main.py
```

## Run Separation Strategy
```shell
cd python/nn4k/nn4k/alignment/examples/sepearation
export PLACEMENT_STRATEGY=separation; python app_main.py
```

## Run Flattening Strategy
```shell
cd python/nn4k/nn4k/alignment/examples/sepearation
export PLACEMENT_STRATEGY=flattening; python app_main.py
```
