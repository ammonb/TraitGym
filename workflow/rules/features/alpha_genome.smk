rule alphagenome_features:
    input:  "results/dataset/{dataset}/test.parquet"
    output: "results/dataset/{dataset}/features/AlphaGenome.parquet"
    shell:  "python workflow/scripts/run_vep_alpha_genome.py --input {input} --out-features {output} --seq-len 1MB --chunk 50"

rule alphagenome_zeroshot:
    input:  "results/dataset/{dataset}/test.parquet"
    output: "results/dataset/{dataset}/preds/all/AlphaGenome.plus.score.parquet"
    shell:  "python workflow/scripts/run_vep_alpha_genome.py --input {input} --out-score {output} --seq-len 1MB --chunk 50"
