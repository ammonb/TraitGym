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
from enum import Enum, auto


class AggregationMode(Enum):
    AGGREGATE_ALL = auto()
    BY_GENE = auto()
    BY_BIO_SAMPLE = auto()
    NONE = auto()


AGGREGATION_POLICY = {
    "CenterMaskScorer(requested_output=ATAC, width=501, aggregation_type=ACTIVE_SUM)": AggregationMode.BY_BIO_SAMPLE,
    "CenterMaskScorer(requested_output=CHIP_HISTONE, width=2001, aggregation_type=ACTIVE_SUM)": AggregationMode.BY_BIO_SAMPLE,

}


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


def process_chunk(results, out_features, out_score, chunk_count):
    tidy = variant_scorers.tidy_scores(results)  # long-form data frame

    # Cast to str and numeric
    tidy["variant_id"]      = tidy["variant_id"].astype(str)
    tidy["output_type"]     = tidy["output_type"].astype(str)
    tidy["variant_scorer"]  = tidy["variant_scorer"].astype(str)
    tidy["raw_score"]       = pd.to_numeric(tidy["raw_score"], errors="coerce")


    # grouped_stats = []

    # # Get list of unique scorers
    # scorers = tidy["variant_scorer"].unique()

    # for scorer in scorers:
    #     tidy_subset = tidy[tidy["variant_scorer"] == scorer]
        
    #     per_variant = (
    #         tidy_subset.groupby("variant_id")
    #         .agg(
    #             num_scores=("raw_score", "count"),
    #             num_biosamples=("biosample_name", lambda x: x.nunique() if x.notna().any() else None),
    #             num_genes=("gene_id", lambda x: x.nunique() if x.notna().any() else None),
    #         )
    #         .reset_index()
    #     )

    #     avg_scores = per_variant["num_scores"].mean()
    #     avg_biosamples = per_variant["num_biosamples"].mean()
    #     avg_genes = per_variant["num_genes"].mean()

    #     grouped_stats.append({
    #         "scorer": scorer,
    #         "avg_scores": round(avg_scores, 1),
    #         "avg_bio_samples": round(avg_biosamples, 1) if avg_biosamples is not None else "N/A",
    #         "avg_target_genes": round(avg_genes, 1) if avg_genes is not None else "N/A"
    #     })

    # # Pretty print
    # print(f"{'scorer':<90} | {'avg_scores':>10} | {'avg_bio_samples':>15} | {'avg_target_genes':>17}")
    # print("-" * 140)
    # for row in grouped_stats:
    #     print(f"{row['scorer']:<90} | {row['avg_scores']:>10} | {str(row['avg_bio_samples']):>15} | {str(row['avg_target_genes']):>17}")


    all_aggregated_rows = []

    for scorer_name, group in tidy.groupby("variant_scorer"):
        mode = AGGREGATION_POLICY.get(scorer_name, AggregationMode.AGGREGATE_ALL)

        if mode == AggregationMode.AGGREGATE_ALL:
            agg = (group.groupby(["variant_id", "output_type"], observed=True)
                    .raw_score.mean()
                    .reset_index())

        elif mode == AggregationMode.BY_BIO_SAMPLE:
            agg = (group.groupby(["variant_id", "output_type", "biosample_name"], observed=True)
                    .raw_score.mean()
                    .reset_index())

        elif mode == AggregationMode.BY_GENE:
            agg = (group.groupby(["variant_id", "output_type", "gene_id"], observed=True)
                    .raw_score.mean()
                    .reset_index())

        elif mode == AggregationMode.NONE:
            agg = group[["variant_id", "output_type", "raw_score"]].copy()

        agg["variant_scorer"] = scorer_name
        agg = agg.rename(columns={"raw_score": "score"})  # ensure score column name
        
        if "biosample_name" not in agg.columns:
            agg["biosample_name"] = "None"

        all_aggregated_rows.append(agg)

    scores = pd.concat(all_aggregated_rows, ignore_index=True)[
        ["variant_id", "output_type", "variant_scorer", "biosample_name", "score"]
    ]
 
    # wide matrix
    wide = scores.pivot(index="variant_id",
                        columns=["output_type","variant_scorer", "biosample_name"],
                        values="score")
    
    # some scores don't exist for all regions. Impute with mean value
    wide = wide.fillna(wide.mean())

    if "variant_id" in wide.columns:
        wide = wide.select_dtypes(exclude=['object'])


    wide.columns = [".".join(c) for c in wide.columns]

    # optional zero-shot single column
    zs = wide.mean(axis=1).rename("score").to_frame()

    # write only if paths were provided
    if out_features: 
        wide.reset_index().to_parquet(
            out_features,
            engine="fastparquet",
            append=(chunk_count != 0),
            index=False
        )
    
    if out_score:    
        zs.reset_index().to_parquet(
            out_score,
            engine="fastparquet",
            append=(chunk_count != 0),
            index=False
        )

def main():
    args = build_parser().parse_args()
    if not args.api_key:
        raise SystemExit("Set --api-key or ALPHAGENOME_API_KEY")

    if not args.out_features and not args.out_score:
         raise SystemExit("Nothing to write: provide --out-features and/or --out-score") 

    # files paths exist and are empty 
    for p in [args.out_features, args.out_score]:
        if p:
            Path(p).parent.mkdir(parents=True, exist_ok=True)

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

    for s in selected:
        print("scorer: {}".format(s))

    chunk_count = 0
    for start in tqdm(range(0, len(df), args.chunk), desc=f"Scoring variants (chunk={args.chunk})"):
        block_results = []    
        batch = df.iloc[start:start + args.chunk]
        variants, intervals = [], []
        for r in batch.itertuples(index=False):
            
            if r.variant_id in cached_variants:
                block_results.append(cached_variants[r.variant_id])
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

            block_results.extend(batch_scores)
            
        process_chunk(block_results, args.out_features, args.out_score, chunk_count)
        chunk_count += 1
        print ("processsed {} from cache and {} from API".format(args.chunk - len(variants), len(variants)))
    

if __name__ == "__main__":
    main()