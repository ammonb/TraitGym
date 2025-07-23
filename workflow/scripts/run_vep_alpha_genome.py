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

import os, uuid, pickle
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
    p.add_argument("--cache-dir", default="cache/alphagenome_raw",
                   help="Directory to store/read raw per-chunk tidy outputs (parquet).")
    p.add_argument("--no-resume", action="store_true",
                   help="Do NOT use cache; always rescore (default is resume).")
    
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




def load_cache_dict(cache_dir):
    print("Loading cached chunks...")
    cache = {}
    for p in Path(cache_dir).glob("batch_*.pkl"):
        with open(p, "rb") as f:
            batch = pickle.load(f)          # list of score objects
        for obj in batch:
            ad = obj[0]          # first AnnData for that variant
            v_id=str(ad.uns["variant"])  
            cache[v_id] = obj                      
    return cache

def main():
    args = build_parser().parse_args()
    if not args.api_key:
        raise SystemExit("Set --api-key or ALPHAGENOME_API_KEY")

    # ---- caching setup ----
    cached_variants = {}
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_resume:
            cached_variants = load_cache_dict(args.cache_dir)


    # Load variants
    df = pd.read_parquet(args.input)      

    df["variant_id"] = (
        df["chrom"].astype(str).str.replace("^chr", "", regex=True).radd("chr")
        + ":" + df["pos"].astype(int).astype(str)
        + ":" + df["ref"] + ">" + df["alt"]
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
    for start in tqdm(range(0, len(df), args.chunk), desc=f"Scoring variants (chunk={args.chunk})"):
        
        batch = df.iloc[start:start + args.chunk]
        variants, intervals = [], []
        for r in batch.itertuples(index=False):
            
            if r.variant_id in cached_variants:
                results.append(cached_variants[r.variant_id])
                continue

            v = genome.Variant(
                chromosome=f"chr{r.chr}" if not str(r.chr).startswith("chr") else str(r.chr),
                position=int(r.pos),
                reference_bases=r.ref,
                alternate_bases=r.alt,
                name=r.variant_id,
            )
            variants.append(v)
            intervals.append(v.reference_interval.resize(seq_len))

        if len(variants):
            batch_scores = model.score_variants(
                intervals=intervals,
                variants=variants,
                variant_scorers=selected,
                organism=organism,
                progress_bar=False,
            )
            
            uid = uuid.uuid4().hex
            cache_path = cache_dir / f"batch_{start:07d}_{uid}.pkl"
            with open(cache_path, "wb") as f:
                pickle.dump(batch_scores, f, protocol=pickle.HIGHEST_PROTOCOL)

             
            results.extend(batch_scores)

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