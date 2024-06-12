import os

for dataset in ["fake", "qaa", "qaq", "airbnb","channel", "cloth"]:
    os.system(f"aws s3 cp /home/ubuntu/drive2/ag_bench_runs/multimodal/{dataset} s3://automl-mm-bench/bag-of-tricks/results/{dataset} --recursive ")