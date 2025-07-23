#!/usr/bin/env python
"""
run_alphagenome.py
Usage:
  python run_alphagenome.py \
    --input  results/dataset/complex_traits_all/test.parquet \
    --output results/dataset/complex_traits_all/features/AlphaGenome.parquet \
    --seq-len 1MB \
    --chunk 256
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers

SUPPORTED_SEQ = dna_client.SUPPORTED_SEQUENCE_LENGTHS  # e.g. SEQUENCE_LENGTH_1MB

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to TraitGym test.parquet")
    p.add_argument("--out-features", help="Parquet: n_variants x n_features")
    p.add_argument("--out-score",    help="Parquet: single 'score' column for zero-shot")
    p.add_argument("--seq-len", default="1MB",
                   choices=["2KB", "16KB", "100KB", "500KB", "1MB"],
                   help="Context length around variant")
    p.add_argument("--organism", default="human", choices=["human", "mouse"])
    p.add_argument("--chunk", type=int, default=256, help="Variants per API batch (looped anyway)")
    p.add_argument("--api-key", default=os.environ.get("ALPHAGENOME_API_KEY"),
                   help="If unset, read from env ALPHAGENOME_API_KEY")
    
    return p

def peek(df, name, n=3):
    print(f"\n==== {name} ====")
    print(f"shape: {df.shape}")
    print(df.dtypes)
    print(df.head(n).to_string())
    # show the python types inside any object cols
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        print(f"- sample types in {c}:",
              df[c].head(20).map(type).unique()[:5])
        
def main():
    args = build_parser().parse_args()
    if not args.api_key:
        raise SystemExit("Set --api-key or ALPHAGENOME_API_KEY")

    # Load variants
    df = pd.read_parquet(args.input)

    df["variant_id"] = (
        df["chrom"].astype(str).str.replace("^chr", "", regex=True).radd("chr")
        + ":" + df["pos"].astype(int).astype(str)
        + ":" + df["ref"] + ":" + df["alt"]
    )
    
    # Map/standardize columns if needed
    rename_map = {}
    for need, cand in [("chr", ["chr", "CHROM", "chromosome","chrom"]),
                       ("pos", ["pos", "POS", "position"]),
                       ("ref", ["ref", "REF", "reference_bases"]),
                       ("alt", ["alt", "ALT", "alternate_bases"])]:
        if need not in df.columns:
            for c in cand:
                if c in df.columns:
                    rename_map[c] = need
                    break
    df = df.rename(columns=rename_map)
    required = {"variant_id", "chr", "pos", "ref", "alt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input parquet missing required cols: {missing}")

    # Init model
    model = dna_client.create(args.api_key)
    organism = {
        "human": dna_client.Organism.HOMO_SAPIENS,
        "mouse": dna_client.Organism.MUS_MUSCULUS,
    }[args.organism]
    seq_len = SUPPORTED_SEQ[f"SEQUENCE_LENGTH_{args.seq_len}"]

    # Choose scorers – start with all recommended, filter by organism
    all_scorers = variant_scorers.RECOMMENDED_VARIANT_SCORERS
    selected = [
        sc for sc in all_scorers.values()
        if organism.value in variant_scorers.SUPPORTED_ORGANISMS[sc.base_variant_scorer]
    ]

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring variants"):
        v = genome.Variant(
            chromosome=f"chr{row.chr}" if not str(row.chr).startswith("chr") else str(row.chr),
            position=int(row.pos),
            reference_bases=row.ref,
            alternate_bases=row.alt,
            name=row.variant_id,
        )
        interval = v.reference_interval.resize(seq_len)
        scores = model.score_variant(
            interval=interval,
            variant=v,
            variant_scorers=selected,
            organism=organism,
        )
        results.append(scores)

    tidy = variant_scorers.tidy_scores(results)  # long-form data frame

    # Cast to str and numeric
    tidy["variant_id"]      = tidy["variant_id"].astype(str)
    tidy["output_type"]     = tidy["output_type"].astype(str)
    tidy["variant_scorer"]  = tidy["variant_scorer"].astype(str)
    tidy["raw_score"]       = pd.to_numeric(tidy["raw_score"], errors="coerce")
    
    # Log stats
    n_rows = len(tidy)
    n_var  = tidy["variant_id"].nunique()
    n_sc   = tidy["variant_scorer"].nunique()
    print ("using {} variants".format(len(df)))
    print ("using {} scorers".format(len(selected)))
    print ("total rows {}".format(len(tidy)))
    print("rows per track_name:", tidy.groupby("track_name").size().describe())
    print("unique biosamples:", tidy["biosample_name"].nunique())
    print("unique genes (gene-specific scorers):", tidy["gene_id"].nunique())

    # Keep only needed cols
    KEEP = ["variant_id", "output_type", "variant_scorer", "raw_score"]
    tidy = tidy[KEEP].copy()
    
    # one scalar per (variant, track/scorer)
    scores = (tidy.groupby(["variant_id","output_type","variant_scorer"], observed=True)
                .raw_score.mean()
                .rename("score")
                .reset_index())

    print("[AG] sample variant/track scores:\n",
      scores.head(20).to_string(index=False))
    
    # wide matrix
    wide = scores.pivot(index="variant_id",
                        columns=["output_type","variant_scorer"],
                        values="score")
    wide.columns = [".".join(c) for c in wide.columns]

    # optional zero-shot single column
    zs = wide.mean(axis=1).rename("score").to_frame()

    for p in [args.out_features, args.out_score]:
        if p:
            Path(p).parent.mkdir(parents=True, exist_ok=True)

    # write only if paths were provided
    if args.out_features: wide.reset_index().to_parquet(args.out_features, index=False)
    if args.out_score:    zs.reset_index().to_parquet(args.out_score, index=False)


    if not args.out_features and not args.out_score:
        print("Nothing to write: provide --out-features and/or --out-score")


if __name__ == "__main__":
    main()