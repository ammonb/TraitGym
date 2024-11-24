rule cadd_download_train_pos:
    output:
        "results/cadd/train_pos.vcf.gz",
    shell:
        "wget -O {output} https://krishna.gs.washington.edu/download/CADD-development/v1.7/training_data/GRCh38/simulation_SNVs.vcf.gz"



rule cadd_download_train_neg:
    output:
        "results/cadd/train_neg.vcf.gz",
    shell:
        "wget -O {output} https://krishna.gs.washington.edu/download/CADD-development/v1.7/training_data/GRCh38/humanDerived_SNVs.vcf.gz"


rule cadd_train_process:
    input:
        "results/cadd/train_pos.vcf.gz",
        "results/cadd/train_neg.vcf.gz",
    output:
        "results/cadd/train.parquet",
    run:
        def load_vcf(path):
            return (
                pl.read_csv(
                    path,
                    has_header=False,
                    separator="\t",
                    new_columns=["chrom", "pos", "id", "ref", "alt"],
                    schema_overrides=dict(chrom=str),
                )
                .drop("id")
            )

        cadd = pl.concat([
            load_vcf(input[0]).with_columns(cadd_label=pl.lit(True)),
            load_vcf(input[1]).with_columns(cadd_label=pl.lit(False)),
        ])
        print(cadd)
        cadd.write_parquet(output[0])


#rule dataset_to_vcf:
#    output:
#        "results/data/{dataset}.vcf.gz",
#    run:
#        V = load_dataset(wildcards.dataset, split="test").to_pandas()
#        V["id"] = "."
#        V[["chrom", "pos", "id", "ref", "alt"]].to_csv(
#            output[0], sep="\t", index=False, header=False,
#        )
#
#
## obtained from the website
#rule cadd_process:
#    input:
#        "results/data/cadd/{dataset}.tsv.gz",
#    output:
#        "results/features/{dataset}/CADD_RawScore.parquet",
#        "results/features/{dataset}/CADD_Annot.parquet",
#    run:
#        V = load_dataset(wildcards.dataset, split="test").to_pandas()
#        df = pd.read_csv(input[0], sep="\t", skiprows=1, dtype={"#Chrom": "str"})
#        df = df.rename(columns={"#Chrom": "chrom", "Pos": "pos", "Ref": "ref", "Alt": "alt"})
#        
#        # some variants have separate rows for AnnoType = Transcript vs. Intergenic
#        # we want features that are independent of the AnnoType
#        dup = df[df.duplicated(COORDINATES, keep=False)][COORDINATES].iloc[0]
#        df2 = df[df[COORDINATES].eq(dup).all(axis=1)]
#        independent_cols = [c for c in df2.columns if df2[c].nunique() == 1]
#        df = df.drop_duplicates(COORDINATES)[independent_cols]
#
#        df = df.loc[:, df.isna().mean() < 0.1]
#        df = df.loc[:, df.nunique() > 1]
#
#        df = df[[
#            c for c in df.columns
#            if c in COORDINATES or c in df.select_dtypes(include=["number", "bool"]).columns
#        ]]
#        print(df)
#
#        other_cols = [c for c in df.columns if c not in COORDINATES]
#        V = V.merge(df, how="left", on=COORDINATES)
#        V[["RawScore"]].to_parquet(output[0], index=False)
#        V[other_cols].to_parquet(output[1], index=False)



rule cadd_dataset_pos:
    input:
        "results/dataset/{dataset}/test.parquet",
    output:
        temp("results/dataset/{dataset}/pos.tsv.gz"),
    run:
        V = pd.read_parquet(input[0])
        V = V[["chrom", "pos"]].drop_duplicates().sort_values(["chrom", "pos"])
        V.to_csv(output[0], sep="\t", header=False, index=False)


rule cadd_tabix:
    input:
        "results/dataset/{dataset}/pos.tsv.gz",
    output:
        "results/dataset/{dataset}/cadd_tabix_res.tsv.gz",
    threads: workflow.cores
    shell:
        "tabix -h {config[cadd][url]} -R {input} | gzip > {output}"


rule cadd_process:
    input:
        "results/dataset/{dataset}/test.parquet",
        "results/dataset/{dataset}/cadd_tabix_res.tsv.gz",
    output:
        "results/dataset/{dataset}/features/CADD.parquet",
    run:
        V = pd.read_parquet(input[0])
        original_cols = V.columns
        df = pd.read_csv(
            input[1], sep="\t", skiprows=1, dtype={"#Chrom": "str"}, na_values=["."],
        )
        df = df.rename(columns={"#Chrom": "chrom", "Pos": "pos", "Ref": "ref", "Alt": "alt"})
        # some variants have separate rows for AnnoType = Transcript vs. Intergenic
        df = df.drop_duplicates(COORDINATES)
        features = config["cadd"]["features"]
        df = df[COORDINATES + features]
        V = V.merge(df, how="left", on=COORDINATES).drop(columns=original_cols)
        print(V)

        def get_default(feature):
            try:
                return trackData[feature.lower()]["na_value"]
            except:
                return None

        for feature in features:
            default = get_default(feature)
            if default is not None:
                V[feature] = V[feature].fillna(default)

        V = V.dropna(axis=1, how="all")
        V = V.loc[:, df.nunique() > 1]
        print(V)
        V.to_parquet(output[0], index=False)



##################################################################################
# copied from:
# https://github.com/kircherlab/CADD-scripts/blob/master/src/scripts/lib/tracks.py
##################################################################################

import math
import numpy as np
import statistics

NUCLEOTIDES = ["A", "C", "G", "T", "N"]
TRANSITIONS = set([('C', 'T'), ('T', 'C'), ('G', 'A'), ('A', 'G')])
TRANSVERSIONS = set([('A', 'C'), ('C', 'A'), ('T', 'A'), ('A', 'T'), ('C', 'G'), ('G', 'C'), ('G', 'T'), ('T', 'G')])

AMINOACIDS = ["*", "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
              "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "unknown"]
# we only train those AA exchanges that can be achieved with a single base exchange (reduces # of tracks from 420 to 170:
AMINOACID_EXCHANGES = \
    ['*-C', '*-E', '*-G', '*-K', '*-L', '*-Q', '*-R', '*-S', '*-W', '*-Y', 'A-D',
     'A-E', 'A-G', 'A-P', 'A-S', 'A-T', 'A-V', 'C-*', 'C-F', 'C-G', 'C-R', 'C-S',
     'C-W', 'C-Y', 'D-A', 'D-E', 'D-G', 'D-H', 'D-N', 'D-V', 'D-Y', 'E-*', 'E-A',
     'E-D', 'E-G', 'E-K', 'E-Q', 'E-V', 'F-C', 'F-I', 'F-L', 'F-S', 'F-V', 'F-Y',
     'G-*', 'G-A', 'G-C', 'G-D', 'G-E', 'G-R', 'G-S', 'G-V', 'G-W', 'H-D', 'H-L',
     'H-N', 'H-P', 'H-Q', 'H-R', 'H-Y', 'I-F', 'I-K', 'I-L', 'I-M', 'I-N', 'I-R',
     'I-S', 'I-T', 'I-V', 'K-*', 'K-E', 'K-I', 'K-M', 'K-N', 'K-Q', 'K-R', 'K-T',
     'L-*', 'L-F', 'L-H', 'L-I', 'L-M', 'L-P', 'L-Q', 'L-R', 'L-S', 'L-V', 'L-W',
     'M-I', 'M-K', 'M-L', 'M-R', 'M-T', 'M-V', 'N-D', 'N-H', 'N-I', 'N-K', 'N-S',
     'N-T', 'N-Y', 'P-A', 'P-H', 'P-L', 'P-Q', 'P-R', 'P-S', 'P-T', 'Q-*', 'Q-E',
     'Q-H', 'Q-K', 'Q-L', 'Q-P', 'Q-R', 'R-*', 'R-C', 'R-G', 'R-H', 'R-I', 'R-K',
     'R-L', 'R-M', 'R-P', 'R-Q', 'R-S', 'R-T', 'R-W', 'S-*', 'S-A', 'S-C', 'S-F',
     'S-G', 'S-I', 'S-L', 'S-N', 'S-P', 'S-R', 'S-T', 'S-W', 'S-Y', 'T-A', 'T-I',
     'T-K', 'T-M', 'T-N', 'T-P', 'T-R', 'T-S', 'V-A', 'V-D', 'V-E', 'V-F', 'V-G',
     'V-I', 'V-L', 'V-M', 'W-*', 'W-C', 'W-G', 'W-L', 'W-R', 'W-S', 'Y-*', 'Y-C',
     'Y-D', 'Y-F', 'Y-H', 'Y-N', 'Y-S',
     'A-A', 'C-C', 'D-D', 'E-E', 'F-F', 'G-G', 'H-H', 'I-I', 'K-K', 'L-L', 'N-N',
     'P-P', 'Q-Q', 'R-R', 'S-S', 'T-T', 'V-V', 'Y-Y',
     ]

trackData = {
    'y': {
        'description': 'ground truth value',
        'type': bool
    },
    'chrom': {
        'description': 'Chromosome',
        'type': str,
        'default': 'ignore'
    },
    'pos': {
        'description': 'Position (1-based)',
        'type': int,
        'default': 'ignore'
    },
    'ref': {
        'description': 'Reference allele',
        'type': list,
        'categories': NUCLEOTIDES,
        'dependencies': ['type', 'ref', 'alt'],
        'derive': lambda x: 'N' if x['type'] != 'SNV' else x['ref'],
        'na_value': 'N',
        'hcdiff_derive': lambda x: 'N' if x['type'] != 'SNV' else x['alt']
    },
    'alt': {
        'description': 'Observed allele',
        'type': list,
        'categories': NUCLEOTIDES,
        'dependencies': ['type', 'alt'],
        'derive': lambda x: 'N' if x['type'] != 'SNV' else x['alt'],
        'na_value': 'N',
        'hcdiff_derive': lambda x: 'N' if x['type'] != 'SNV' else x['ref']
    },
    'type': {
        'description': 'Event type (SNV, DEL, INS)',
        'type': list,
        'categories': ['SNV', 'INS', 'DEL'],
        'hcdiff_transformation': lambda x: 'SNV' if x == 'SNV' else {'INS': 'DEL', 'DEL': 'INS'}[x]
    },
    'length': {
        'description': 'Number of inserted/deleted bases',
        'type': int,
        'transformation': lambda x: min(x, 49)
    },
    'istv': {
        'description': 'Is transversion?',
        'type': float,
        'dependencies': ['ref', 'alt', 'type'],
        'derive': lambda x: 0.5 if x['type'] != 'SNV' else 0 if (x['ref'], x['alt']) in TRANSITIONS else 1 if (x['ref'], x['alt']) in TRANSVERSIONS else 0.5
    },
    'annotype': {
        'description': 'CodingTranscript, Intergenic, MotifFeature, NonCodingTranscript, RegulatoryFeature, Transcript',
        'type': list,
        'categories': ['CodingTranscript', 'Intergenic', 'MotifFeature', 'NonCodingTranscript', 'RegulatoryFeature', 'Transcript'],
        'default': 'ignore'
    },
    'consequence': {
        'description': 'VEP consequence, priority selected by potential impact',
        'type': list,
        'categories': ["3PRIME_UTR", "5PRIME_UTR", "DOWNSTREAM", "INTERGENIC", "INTRONIC", "NONCODING_CHANGE", "INFRAME", "FRAME_SHIFT", "NON_SYNONYMOUS", "REGULATORY", "CANONICAL_SPLICE", "SPLICE_SITE", "STOP_GAINED", "STOP_LOST", "SYNONYMOUS", "UPSTREAM"],
        'na_value': 'INTERGENIC',
        'hcdiff_transformation': lambda x: x if x not in ['STOP_LOST', 'STOP_GAINED'] else {'STOP_LOST': 'STOP_GAINED', 'STOP_GAINED': 'STOP_LOST'}[x]
    },
    'consscore': {
        'description': 'Custom deleterious score assigned to Consequence',
        'type': int,
        'default': 'ignore'
    },
    'consdetail': {
        'description': 'Trimmed VEP consequence prior to simplification',
        'type': str,
        'default': 'ignore'
    },
    'gc': {
        'description': 'Percent GC in a window of +/- 75bp',
        'type': float,
        'na_value': 0.42,
        'scaling': 'quantile'
    },
    'cpg': {
        'description': 'Percent CpG in a window of +/- 75bp',
        'type': float,
        'na_value': 0.02,
        'scaling': 'quantile'
    },
    'priphcons': {
        'description': 'Primate PhastCons conservation score (excl. human)',
        'type': float,
        'na_value': 0.115,
        'scaling': 'quantile'
    },
    'mamphcons': {
        'description': 'Mammalian PhastCons conservation score (excl. human)',
        'type': float,
        'na_value': 0.079,
        'scaling': 'quantile'
    },
    'verphcons': {
        'description': 'Vertebrate PhastCons conservation score (excl. human)',
        'type': float,
        'na_value': 0.094,
        'scaling': 'quantile'
    },
    'priphcons38': {
        'colname': 'priphcons',
        'copy': 'priphcons',
        'na_value': 0.,
    },
    'mamphcons38': {
        'colname': 'mamphcons',
        'copy': 'mamphcons',
        'na_value': 0.,
    },
    'verphcons38': {
        'colname': 'verphcons',
        'copy': 'verphcons',
        'na_value': 0.,
    },
    'priphylop': {
        'description': 'Primate PhyloP score (excl. human)',
        'type': float,
        'na_value': -0.033,
        'scaling': 'quantile'
    },
    'mamphylop': {
        'description': 'Mammalian PhyloP score (excl. human)',
        'type': float,
        'na_value': -0.038,
        'scaling': 'quantile'
    },
    'verphylop': {
        'description': 'Vertebrate PhyloP score (excl. human)',
        'type': float,
        'na_value': 0.017,
        'scaling': 'quantile'
    },
    'priphylop38': {
        'colname': 'priphylop',
        'copy': 'priphylop',
        'na_value': -0.029,
    },
    'mamphylop38': {
        'colname': 'mamphylop',
        'copy': 'mamphylop',
        'na_value': -0.005,
    },
    'verphylop38': {
        'colname': 'verphylop',
        'copy': 'verphylop',
        'na_value': 0.042,
    },
    'gerpn': {
        'description': 'Neutral evolution score defined by GERP++',
        'type': float,
        'na_value': 1.91,
        'scaling': 'quantile'
    },
    'gerpn38': {
        'colname': 'gerpn',
        'copy': 'gerpn',
        'na_value': 3.,
    },
    'gerps': {
        'description': 'Rejected Substitution score defined by GERP++',
        'type': float,
        'na_value': -0.200,
        'scaling': 'quantile'
    },
    'gerprs': {
        'description': 'Gerp element score',
        'type': float,
        'na_value': 0,
        'scaling': 'quantile'
    },
    'gerprspval': {
        'description': 'Gerp element p-Value',
        'type': float,
        'transformation': lambda x: math.log(4.94066e-324) if x == 0 else math.log(x),
        'na_value': 0,
        'scaling': 'quantile'
    },
    'bstatistic': {
        'description': 'Background selection score',
        'type': int,
        'na_value': 800,
        'scaling': 'quantile'
    },
    'mirsvrs': {
        'description': 'mirSVR-Score',
        'colname': 'mirsvr-score',
        'type': float,
        'na_value': 0,
        'indicator': True,
        'scaling': 'quantile'
    },
    'mirsvre': {
        'description': 'mirSVR-E',
        'colname': 'mirsvr-e',
        'type': float,
        'na_value': 0,
        'scaling': 'quantile'
    },
    'mirsvra': {
        'description': 'mirSVR-Aln',
        'colname': 'mirsvr-aln',
        'type': int,
        'na_value': 0,
        'scaling': 'quantile'
    },
    'targetscan': {
        'description': 'targetscan',
        'type': int,
        'na_value': 0,
        'indicator': True,
        'scaling': 'quantile'
    },
    'chmmtssa': {
        'description': 'Proportion of 127 cell types in cHmmTssA state',
        'type': float,
        'na_value': 0.0667,
        'indicator': True,
        'scaling': 'quantile'
    },
    'chmmtssaflnk': {
        'description': 'Proportion of 127 cell types in cHmmTssAFlnk state',
        'type': float,
        'na_value': 0.0667,
        'shared_indicator': 'chmmtssa',
        'scaling': 'quantile'
    },
    'chmmtxflnk': {
        'description': 'Proportion of 127 cell types in cHmmTxFlnk state',
        'copy': 'chmmtssaflnk'
    },
    'chmmtx': {
        'description': 'Proportion of 127 cell types in cHmmTx state',
        'copy': 'chmmtssaflnk'
    },
    'chmmtxwk': {
        'description': 'Proportion of 127 cell types in cHmmTxWk state',
        'copy': 'chmmtssaflnk'
    },
    'chmmenhg': {
        'description': 'Proportion of 127 cell types in cHmmEnhG state',
        'copy': 'chmmtssaflnk'
    },
    'chmmenh': {
        'description': 'Proportion of 127 cell types in cHmmEnh state',
        'copy': 'chmmtssaflnk'
    },
    'chmmznfrpts': {
        'description': 'Proportion of 127 cell types in cHmmZnfRpts state',
        'copy': 'chmmtssaflnk'
    },
    'chmmhet': {
        'description': 'Proportion of 127 cell types in cHmmHet state',
        'copy': 'chmmtssaflnk'
    },
    'chmmtssbiv': {
        'description': 'Proportion of 127 cell types in cHmmTssBiv state',
        'copy': 'chmmtssaflnk'
    },
    'chmmbivflnk': {
        'description': 'Proportion of 127 cell types in cHmmBivFlnk state',
        'copy': 'chmmtssaflnk'
    },
    'chmmenhbiv': {
        'description': 'Proportion of 127 cell types in cHmmEnhBiv state',
        'copy': 'chmmtssaflnk'
    },
    'chmmreprpc': {
        'description': 'Proportion of 127 cell types in cHmmReprPC state',
        'copy': 'chmmtssaflnk'
    },
    'chmmreprpcwk': {
        'description': 'Proportion of 127 cell types in cHmmReprPCWk state',
        'copy': 'chmmtssaflnk'
    },
    'chmmquies': {
        'description': 'Proportion of 127 cell types in cHmmQuies state',
        'copy': 'chmmtssaflnk'
    },
    'chmm_e1': {
        'description': 'Number of 48 cell types in chromHMM state E1_poised',
        'type': float,
        'na_value': 1.92,
        'indicator': True,
        'scaling': 'quantile'
    },
    'chmm_e2': {
        'description': 'Number of 48 cell types in chromHMM state E2_repressed',
        'type': float,
        'na_value': 1.92,
        'shared_indicator': 'chmm_e1',
        'scaling': 'quantile'
    },
    'chmm_e3': {
        'description': 'Number of 48 cell types in chromHMM state E3_dead',
        'copy': 'chmm_e2'
    },
    'chmm_e4': {
        'description': 'Number of 48 cell types in chromHMM state E4_dead',
        'copy': 'chmm_e2'
    },
    'chmm_e5': {
        'description': 'Number of 48 cell types in chromHMM state E5_repressed',
        'copy': 'chmm_e2'
    },
    'chmm_e6': {
        'description': 'Number of 48 cell types in chromHMM state E6_repressed',
        'copy': 'chmm_e2'
    },
    'chmm_e7': {
        'description': 'Number of 48 cell types in chromHMM state E7_weak',
        'copy': 'chmm_e2'
    },
    'chmm_e8': {
        'description': 'Number of 48 cell types in chromHMM state E8_gene',
        'copy': 'chmm_e2'
    },
    'chmm_e9': {
        'description': 'Number of 48 cell types in chromHMM state E9_gene',
        'copy': 'chmm_e2'
    },
    'chmm_e10': {
        'description': 'Number of 48 cell types in chromHMM state E10_gene',
        'copy': 'chmm_e2'
    },
    'chmm_e11': {
        'description': 'Number of 48 cell types in chromHMM state E11_gene',
        'copy': 'chmm_e2'
    },
    'chmm_e12': {
        'description': 'Number of 48 cell types in chromHMM state E12_distal',
        'copy': 'chmm_e2'
    },
    'chmm_e13': {
        'description': 'Number of 48 cell types in chromHMM state E13_distal',
        'copy': 'chmm_e2'
    },
    'chmm_e14': {
        'description': 'Number of 48 cell types in chromHMM state E14_distal',
        'copy': 'chmm_e2'
    },
    'chmm_e15': {
        'description': 'Number of 48 cell types in chromHMM state E15_weak',
        'copy': 'chmm_e2'
    },
    'chmm_e16': {
        'description': 'Number of 48 cell types in chromHMM state E16_tss',
        'copy': 'chmm_e2'
    },
    'chmm_e17': {
        'description': 'Number of 48 cell types in chromHMM state E17_proximal',
        'copy': 'chmm_e2'
    },
    'chmm_e18': {
        'description': 'Number of 48 cell types in chromHMM state E18_proximal',
        'copy': 'chmm_e2'
    },
    'chmm_e19': {
        'description': 'Number of 48 cell types in chromHMM state E19_tss',
        'copy': 'chmm_e2'
    },
    'chmm_e20': {
        'description': 'Number of 48 cell types in chromHMM state E20_poised',
        'copy': 'chmm_e2'
    },
    'chmm_e21': {
        'description': 'Number of 48 cell types in chromHMM state E21_dead',
        'copy': 'chmm_e2'
    },
    'chmm_e22': {
        'description': 'Number of 48 cell types in chromHMM state E22_repressed',
        'copy': 'chmm_e2'
    },
    'chmm_e23': {
        'description': 'Number of 48 cell types in chromHMM state E23_weak',
        'copy': 'chmm_e2'
    },
    'chmm_e24': {
        'description': 'Number of 48 cell types in chromHMM state E24_distal',
        'copy': 'chmm_e2'
    },
    'chmm_e25': {
        'description': 'Number of 48 cell types in chromHMM state E25_distal',
        'copy': 'chmm_e2'
    },
    'encexp': {
        'description': 'Maximum ENCODE expression value',
        'type': float,
        'na_value': 0,
        'transformation': lambda x: min(x, 593.26),
        'scaling': 'quantile'
    },
    'ench3k27ac': {
        'description': 'Maximum ENCODE H3K27 acetylation level',
        'transformation': lambda x: min(x, 84.24),
        'copy': 'encexp'
    },
    'ench3k4me1': {
        'description': 'Maximum ENCODE H3K4 methylation level',
        'transformation': lambda x: min(x, 40.4),
        'copy': 'encexp'
    },
    'ench3k4me3': {
        'description': 'Maximum ENCODE H3K4 trimethylation level',
        'transformation': lambda x: min(x, 72.88),
        'copy': 'encexp'
    },
    'encnucleo': {
        'description': 'Maximum of ENCODE Nucleosome position track score',
        'transformation': lambda x: min(x, 3.9),
        'copy': 'encexp'
    },
    'encocc': {
        'description': 'ENCODE open chromatin code',
        'type': int,
        'na_value': 5,
        'scaling': 'quantile'
    },
    'encoccombpval': {
        'description': 'ENCODE combined p-Value (PHRED-scale) of Faire, Dnase, polII, CTCF, Myc evidence for open chromatin',
        'type': float,
        'na_value': 0,
        'scaling': 'quantile'
    },
    'encocdnasepval': {
        'description': 'p-Value (PHRED-scale) of Dnase evidence for open chromatin',
        'copy': 'encoccombpval'
    },
    'encocfairepval': {
        'description': 'p-Value (PHRED-scale) of Faire evidence for open chromatin',
        'copy': 'encoccombpval'
    },
    'encocpoliipval': {
        'description': 'p-Value (PHRED-scale) of polII evidence for open chromatin',
        'copy': 'encoccombpval'
    },
    'encocctcfpval': {
        'description': 'p-Value (PHRED-scale) of CTCF evidence for open chromatin',
        'copy': 'encoccombpval'
    },
    'encocmycpval': {
        'description': 'p-Value (PHRED-scale) of Myc evidence for open chromatin',
        'copy': 'encoccombpval'
    },
    'encocdnasesig': {
        'description': 'Peak signal for Dnase evidence of open chromatin',
        'type': float,
        'na_value': 0,
        'transformation': lambda x: min(x, 0.51),
        'scaling': 'quantile'
    },
    'encocfairesig': {
        'description': 'Peak signal for Faire evidence of open chromatin',
        'transformation': lambda x: min(x, 0.04),
        'copy': 'encocdnasesig'
    },
    'encocpoliisig': {
        'description': 'Peak signal for polII evidence of open chromatin',
        'transformation': lambda x: min(x, 0.25),
        'copy': 'encocdnasesig'
    },
    'encocctcfsig': {
        'description': 'Peak signal for CTCF evidence of open chromatin',
        'transformation': lambda x: min(x, 3.09),
        'copy': 'encocdnasesig'
    },
    'encocmycsig': {
        'description': 'Peak signal for Myc evidence of open chromatin',
        'transformation': lambda x: min(x, 0.17),
        'copy': 'encocdnasesig'
    },
    'encodeh3k4me1-sum': {
        'description': 'Sum of ENCODE H3K4me1 levels (from 13 cell lines)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 1.74),
        'na_value': 0.76
    },
    'encodeh3k4me1-max': {
        'description': 'Maximum ENCODE H3K4me1 level (from 13 cell lines)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 1.24),
        'na_value': 0.37
    },
    'encodeh3k4me2-sum': {
        'description': 'Sum of ENCODE H3K4me2 levels (from 14 cell lines)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 2.16),
        'na_value': 0.73
    },
    'encodeh3k4me2-max': {
        'description': 'Maximum ENCODE H3K4me2 level (from 14 cell lines)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 1.53),
        'na_value': 0.37
    },
    'encodeh3k4me3-sum': {
        'description': 'Sum of ENCODE H3K4me3 levels (from 14 cell lines)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 2.24),
        'na_value': 0.81
    },
    'encodeh3k4me3-max': {
        'description': 'Maximum ENCODE H3K4me3 level (from 14 cell lines)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 1.56),
        'na_value': 0.38
    },
    'encodeh3k9ac-sum': {
        'description': 'Sum of ENCODE H3K9ac levels (from 13 cell lines)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 2.00),
        'na_value': 0.82
    },
    'encodeh3k9ac-max': {
        'description': 'Maximum ENCODE H3K9ac level (from 13 cell lines)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 1.44),
        'na_value': 0.41
    },
    'encodeh3k9me3-sum': {
        'description': 'Sum of ENCODE H3K9me3 levels (from 14 cell lines)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 2.24),
        'na_value': 0.81
    },
    'encodeh3k9me3-max': {
        'description': 'Maximum ENCODE H3K9me3 level (from 14 cell lines)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 1.56),
        'na_value': 0.38
    },
    'encodeh3k27ac-sum': {
        'description': 'Sum of ENCODE H3K27ac levels (from 14 cell lines)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 2.02),
        'na_value': 0.74
    },
    'encodeh3k27ac-max': {
        'description': 'Maximum ENCODE H3K27ac level (from 14 cell lines)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 1.53),
        'na_value': 0.36
    },
    'encodeh3k27me3-sum': {
        'description': 'Sum of ENCODE H3K27me3 levels (from 14 cell lines)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 1.67),
        'na_value': 0.93
    },
    'encodeh3k27me3-max': {
        'description': 'Maximum ENCODE H3K27me3 level (from 14 cell lines)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 1.15),
        'na_value': 0.47
    },
    'encodeh3k36me3-sum': {
        'description': 'Sum of ENCODE H3K36me3 levels (from 10 cell lines)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 1.78),
        'na_value': 0.71
    },
    'encodeh3k36me3-max': {
        'description': 'Maximum ENCODE H3K36me3 level (from 10 cell lines)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 1.12),
        'na_value': 0.39
    },
    'encodeh3k79me2-sum': {
        'description': 'Sum of ENCODE H3K79me2 levels (from 13 cell lines)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 2.02),
        'na_value': 0.64
    },
    'encodeh3k79me2-max': {
        'description': 'Maximum ENCODE H3K79me2 level (from 13 cell lines)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 1.34),
        'na_value': 0.34
    },
    'encodeh4k20me1-sum': {
        'description': 'Sum of ENCODE H4K20me1 levels (from 11 cell lines)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 1.55),
        'na_value': 0.88
    },
    'encodeh4k20me1-max': {
        'description': 'Maximum ENCODE H4K20me1 level (from 11 cell lines)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 1.08),
        'na_value': 0.47
    },
    'encodeh2afz-sum': {
        'description': 'Sum of ENCODE H2AFZ levels (from 13 cell lines)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 1.82),
        'na_value': 0.90
    },
    'encodeh2afz-max': {
        'description': 'Maximum ENCODE H2AFZ level (from 13 cell lines)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 1.17),
        'na_value': 0.42
    },
    'encodednase-sum': {
        'description': 'Sum of ENCODE DNase-seq levels (from 12 cell lines)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 2),
        'na_value': 0.
    },
    'encodednase-max': {
        'description': 'Maximum ENCODE DNase-seq level (from 12 cell lines)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 1.5),
        'na_value': 0.
    },
    'encodetotalrna-sum': {
        'description': 'Sum of ENCODE totalRNA-seq levels (from 10 cell lines always minus and plus strand)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 3),
        'na_value': 0.
    },
    'encodetotalrna-max': {
        'description': 'Maximum ENCODE totalRNA-seq level (from 10 cell lines, minus and plus strand separately)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+1), 2),
        'na_value': 0.
    },
    'encsegway': {
        'description': 'Result of genomic segmentation algorithm',
        'colname': 'segway',
        'type': list,
        'categories': ["C0", "C1", "D", "E/GM", "F0", "F1", "GE0", "GE1", "GE2", "GM0", "GM1", "GS", "H3K9me1", "L0", "L1", "R0", "R1", "R2", "R3", "R4", "R5", "TF0", "TF1", "TF2", "TSS", "unknown"],
        'na_value': 'unknown'
    },
    'toverlapmotifs': {
        'description': 'Number of overlapping predicted TF motifs',
        'type': float,
        'na_value': 0
    },
    'motifdist': {
        'description': 'Reference minus alternate allele difference in nucleotide frequency within an predicted overlapping motif',
        'type': float,
        'na_value': 0
    },
    'motifecount': {
        'description': 'Total number of overlapping motifs',
        'type': int,
        'na_value': 0
    },
    'motifename': {
        'description': 'Name of sequence motif the position overlaps',
        'type': str,
        'default': 'ignore'
    },
    'motifehipos': {
        'description': 'Is the position considered highly informative for an overlapping motif by VEP',
        'type': bool,
        'na_value': 0
    },
    'motifescorechng': {
        'description': 'VEP score change for the overlapping motif site',
        'type': float,
        'na_value': 0
    },
    'tfbs': {
        'description': 'Number of different overlapping ChIP transcription factor binding sites',
        'type': float,
        'na_value': 0,
        'transformation': lambda x: min(x, 48)
    },
    'tfbspeaks': {
        'description': 'Number of overlapping ChIP transcription factor binding site peaks summed over different cell types/tissue',
        'type': float,
        'na_value': 0,
        'transformation': lambda x: min(x, 81)
    },
    'tfbspeaksmax': {
        'description': 'Maximum value of overlapping ChIP transcription factor binding site peaks across cell types/tissue',
        'type': float,
        'na_value': 0,
        'transformation': lambda x: min(x, 542.0574)
    },
    'mindisttss': {
        'description': 'Distance to closest Transcribed Sequence Start (TSS)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+10), 5.5),
        'na_value': 5.5
    },
    'mindisttse': {
        'description': 'Distance to closest Transcribed Sequence End (TSE)',
        'type': float,
        'transformation': lambda x: min(math.log10(x+10), 5.5),
        'na_value': 5.5
    },
    'geneid': {
        'description': 'ENSEMBL GeneID',
        'type': str,
        'default': 'ignore'
    },
    'featureid': {
        'description': 'ENSEMBL feature ID (Transcript ID or regulatory feature ID)',
        'type': str,
        'default': 'ignore'
    },
    'ccds': {
        'description': 'Consensus Coding Sequence ID',
        'type': str,
        'default': 'ignore'
    },
    'genename': {
        'description': 'GeneName provided in ENSEMBL annotation',
        'type': str,
        'default': 'ignore'
    },
    'cdnapos': {
        'description': 'Base position from transcription start',
        'type': float,
        'na_value': 0,
        'indicator': True,
        'scaling': 'quantile'
    },
    'relcdnapos': {
        'description': 'Relative position in transcript',
        'type': float,
        'na_value': 0,
        'scaling': 'quantile'
    },
    'cdspos': {
        'description': 'Base position from coding start',
        'type': float,
        'na_value': 0,
        'indicator': True,
        'scaling': 'quantile'
    },
    'relcdspos': {
        'description': 'Relative position in coding sequence',
        'type': float,
        'na_value': 0,
        'scaling': 'quantile'
    },
    'protpos': {
        'description': 'Amino acid position from coding start',
        'type': float,
        'na_value': 0,
        'indicator': True,
        'scaling': 'quantile'
    },
    'relprotpos': {
        'description': 'Relative position in protein codon',
        'type': float,
        'na_value': 0,
        'scaling': 'quantile'
    },
    'domain': {
        'description': 'Domain annotation inferred from VEP annotation (ncoils, sigp, lcompl, hmmpanther, ndomain = "other named domain")',
        'type': list,
        'categories': ['ncoils', 'sigp', 'lcompl', 'ndomain', 'hmmpanther', 'other', 'UD'],
        'na_value': 'UD'
    },
    'dst2splice': {
        'description': 'Distance to splice site in 20bp; positive: exonic, negative: intronic',
        'type': float,
        'na_value': 0
    },
    'dst2spltype': {
        'description': 'Closest splice site is ACCEPTOR or DONOR',
        'type': list,
        'categories': ["DONOR", "ACCEPTOR", "unknown"],
        'na_value': 'unknown'
    },
    'exon': {
        'description': 'Exon number/Total number of exons',
        'type': str,
        'default': 'ignore'
    },
    'intron': {
        'description': 'Intron number/Total number of exons',
        'type': str,
        'default': 'ignore'
    },
    'oaa': {
        'description': 'Reference amino acid',
        'type': list,
        'categories': AMINOACIDS,
        'dependencies': ['type', 'oaa'],
        'derive': lambda x: 'unknown' if x['type'] != 'SNV' else x['oaa'],
        'na_value': 'unknown',
        'hcdiff_derive': lambda x: 'unknown' if x['type'] != 'SNV' else x['naa']
    },
    'naa': {
        'description': 'Amino acid of observed variant',
        'type': list,
        'categories': AMINOACIDS,
        'dependencies': ['type', 'naa'],
        'derive': lambda x: 'unknown' if x['type'] != 'SNV' else x['naa'],
        'na_value': 'unknown',
        'hcdiff_derive': lambda x: 'unknown' if x['type'] != 'SNV' else x['oaa']
    },
    'grantham': {
        'description': 'Grantham score: oAA,nAA',
        'type': float,
        'na_value': 0,
        'indicator': True,
        'scaling': 'quantile'
    },
    'polyphencat': {
        'description': 'PolyPhen category of change',
        'type': list,
        'categories': ["benign", "possibly_damaging", "probably_damaging", "unknown", "UD"],
        'na_value': 'UD',
        'dependencies': ['type', 'consequence', 'polyphencat'],
        'derive': lambda x: x['polyphencat'] if (x['type'] != 'SNV') or (x['consequence'] != 'NON_SYNONYMOUS') or (x['polyphencat'] != 'NA') else 'unknown'
    },
    'polyphenval': {
        'description': 'PolyPhen score',
        'type': float,
        'dependencies': ['type', 'consequence', 'polyphenval'],
        'derive': lambda x: x['polyphenval'] if (x['type'] != 'SNV') or (x['consequence'] != 'NON_SYNONYMOUS') or (x['polyphenval'] != 'NA') else 0.404,
        'na_value': 0,
        'indicator': True,
        'scaling': 'quantile'
    },
    'siftcat': {
        'description': 'SIFT category of change',
        'type': list,
        'categories': ["deleterious", "tolerated", "unknown", "UD"],
        'na_value': 'UD',
        'dependencies': ['type', 'consequence', 'siftcat'],
        'derive': lambda x: x['siftcat'] if (x['type'] != 'SNV') or (x['consequence'] != 'NON_SYNONYMOUS') or (x['siftcat'] != 'NA') else 'unknown'
    },
    'siftval': {
        'description': 'SIFT score',
        'type': float,
        'dependencies': ['type', 'consequence', 'siftval'],
        'derive': lambda x: x['siftval'] if (x['type'] != 'SNV') or (x['consequence'] != 'NON_SYNONYMOUS') or (x['siftval'] != 'NA') else 0.22,
        'na_value': 0,
        'indicator': True,
        'scaling': 'quantile'
    },
    'spliceai-acc-gain': {
        'description': 'SpliceAI acceptor gain score',
        'type': float,
        'na_value': 0,
        'indicator': True,
        'dependencies': ['spliceai-acc-gain', 'spliceai-acc-loss'],
        'derive': lambda x: x['spliceai-acc-gain'],
        'hcdiff_derive': lambda x: x['spliceai-acc-loss'],
    },
    'spliceai-acc-loss': {
        'description': 'SpliceAI acceptor loss score',
        'type': float,
        'na_value': 0,
        'shared_indicator': 'spliceai-acc-gain',
        'dependencies': ['spliceai-acc-loss', 'spliceai-acc-gain'],
        'derive': lambda x: x['spliceai-acc-loss'],
        'hcdiff_derive': lambda x: x['spliceai-acc-gain'],
    },
    'spliceai-don-gain': {
        'description': 'SpliceAI donor gain score',
        'copy': 'spliceai-acc-loss',
        'dependencies': ['spliceai-don-gain', 'spliceai-don-loss'],
        'derive': lambda x: x['spliceai-don-gain'],
        'hcdiff_derive': lambda x: x['spliceai-don-loss'],
    },
    'spliceai-don-loss': {
        'description': 'SpliceAI donor loss score',
        'copy': 'spliceai-acc-loss',
        'dependencies': ['spliceai-don-loss', 'spliceai-don-gain'],
        'derive': lambda x: x['spliceai-don-loss'],
        'hcdiff_derive': lambda x: x['spliceai-don-gain'],
    },
    'mmsplice-acceptorintron': {
        'description': 'MMSplice acceptor intron score',
        'type': float,
        'na_value': 0,
        'dependencies': ['mmsp_acceptorintron'],
        'derive': lambda x: min(float(x['mmsp_acceptorintron']), 0),
        'hcdiff_derive': lambda x: min(float(x['mmsp_acceptorintron']) * -1, 0),
    },
    'mmsplice-acceptor': {
        'description': 'MMSplice acceptor score',
        'type': float,
        'na_value': 0,
        'dependencies': ['mmsp_acceptor'],
        'derive': lambda x: min(float(x['mmsp_acceptor']), 0),
        'hcdiff_derive': lambda x: min(float(x['mmsp_acceptor']) * -1, 0),
    },
    'mmsplice-exon': {
        'description': 'MMSplice exon score',
        'type': float,
        'na_value': 0,
        'dependencies': ['mmsp_exon'],
        'derive': lambda x: min(float(x['mmsp_exon']), 0),
        'hcdiff_derive': lambda x: min(float(x['mmsp_exon']) * -1, 0),
    },
    'mmsplice-donorintron': {
        'description': 'MMSplice donor intron score',
        'type': float,
        'na_value': 0,
        'dependencies': ['mmsp_donorintron'],
        'derive': lambda x: min(float(x['mmsp_donorintron']), 0),
        'hcdiff_derive': lambda x: min(float(x['mmsp_donorintron']) * -1, 0),
    },
    'mmsplice-donor': {
        'description': 'MMSplice donor score',
        'type': float,
        'na_value': 0,
        'dependencies': ['mmsp_donor'],
        'derive': lambda x: min(float(x['mmsp_donor']), 0),
        'hcdiff_derive': lambda x: min(float(x['mmsp_donor']) * -1, 0),
    },
    'ensembleregulatoryfeature': {
        'description': 'Matches in the Ensemble Regualtory Built (similar to annotype)',
        'type': list,
        'categories': ['Promoter', 'TF binding site', 'Enhancer', 'Promoter Flanking Region', 'CTCF Binding Site', 'Open chromatin', 'NA'],
        'na_value': 'NA'
    },
    'dist2mutation': {
        'description': 'Distance between the closest known SNV up and downstream (position itself excluded)',
        'type': float,
        'transformation': math.log,
        'na_value': 0,
        'indicator': True,
        'scaling': 'quantile'
    },
    'freq100bp': {
        'description': 'Number of frequent (MAF > 0.05) SNV in 100 bp window nearby',
        'type': int,
        'dependencies': ['freq100bp', 'type'],
        'derive': lambda x: 0 if x['type'] != 'SNV' else x['freq100bp'],
        'na_value': 0,
        'scaling': 'quantile'
    },
    'rare100bp': {
        'description': 'Number of rare (MAF < 0.05) SNV in 100 bp window nearby',
        'dependencies': ['rare100bp', 'type'],
        'derive': lambda x: 0 if x['type'] != 'SNV' else x['rare100bp'],
        'copy': 'freq100bp'
    },
    'sngl100bp': {
        'description': 'Number of single occurrence SNV in 100 bp window nearby',
        'dependencies': ['sngl100bp', 'type'],
        'derive': lambda x: 0 if x['type'] != 'SNV' else x['sngl100bp'],
        'copy': 'freq100bp'
    },
    'freq1000bp': {
        'description': 'Number of frequent (MAF > 0.05) SNV in 1000 bp window nearby',
        'dependencies': ['freq1000bp', 'type'],
        'derive': lambda x: 0 if x['type'] != 'SNV' else x['freq1000bp'],
        'copy': 'freq100bp'
    },
    'rare1000bp': {
        'description': 'Number of rare (MAF < 0.05) SNV in 1000 bp window nearby',
        'dependencies': ['rare1000bp', 'type'],
        'derive': lambda x: 0 if x['type'] != 'SNV' else x['rare1000bp'],
        'copy': 'freq100bp'
    },
    'sngl1000bp': {
        'description': 'Number of single occurrence SNV in 1000 bp window nearby',
        'dependencies': ['sngl1000bp', 'type'],
        'derive': lambda x: 0 if x['type'] != 'SNV' else x['sngl1000bp'],
        'copy': 'freq100bp'
    },
    'freq10000bp': {
        'description': 'Number of frequent (MAF > 0.05) SNV in 10000 bp window nearby',
        'dependencies': ['freq10000bp', 'type'],
        'derive': lambda x: 0 if x['type'] != 'SNV' else x['freq10000bp'],
        'copy': 'freq100bp'
    },
    'rare10000bp': {
        'description': 'Number of rare (MAF < 0.05) SNV in 10000 bp window nearby',
        'dependencies': ['rare10000bp', 'type'],
        'derive': lambda x: 0 if x['type'] != 'SNV' else x['rare10000bp'],
        'copy': 'freq100bp'
    },
    'sngl10000bp': {
        'description': 'Number of single occurrence SNV in 10000 bp window nearby',
        'dependencies': ['sngl10000bp', 'type'],
        'derive': lambda x: 0 if x['type'] != 'SNV' else x['sngl10000bp'],
        'copy': 'freq100bp'
    },
    'mw-relfreq1': {
        'description': 'Log10 of relative frequency of frequent SNV in 100 and 1000 bp window',
        'type': float,
        'dependencies': ['freq100bp', 'freq1000bp', 'type'],
        'derive': lambda x: 0 if x['type'] != 'SNV' else math.log10(float(x['freq100bp']) / float(x['freq1000bp'])),
        'na_value': -0.5
    },
    'mw-relfreq2': {
        'description': 'Log10 of relative frequency of frequent SNV in 100 and 10000 bp window',
        'type': float,
        'dependencies': ['freq100bp', 'freq10000bp', 'type'],
        'derive': lambda x: 0 if x['type'] != 'SNV' else math.log10(float(x['freq100bp']) / float(x['freq10000bp'])),
        'na_value': -1.5
    },
    'mw-relfreq3': {
        'description': 'Log10 of relative frequency of frequent SNV in 1000 and 10000 bp window',
        'type': float,
        'dependencies': ['freq1000bp', 'freq10000bp', 'type'],
        'derive': lambda x: 0 if x['type'] != 'SNV' else math.log10(float(x['freq1000bp']) / float(x['freq10000bp'])),
        'na_value': -1
    },
    'mw-relrare1': {
        'description': 'Log10 of relative frequency of rare SNV in 100 and 1000 bp window',
        'type': float,
        'dependencies': ['rare100bp', 'rare1000bp', 'type'],
        'derive': lambda x: 0 if x['type'] != 'SNV' else math.log10(float(x['rare100bp']) / float(x['rare1000bp'])),
        'na_value': -1
    },
    'mw-relrare2': {
        'description': 'Log10 of relative frequency of rare SNV in 100 and 10000 bp window',
        'type': float,
        'dependencies': ['rare100bp', 'rare10000bp', 'type'],
        'derive': lambda x: 0 if x['type'] != 'SNV' else math.log10(float(x['rare100bp']) / float(x['rare10000bp'])),
        'na_value': -2
    },
    'mw-relrare3': {
        'description': 'Log10 of relative frequency of rare SNV in 1000 and 10000 bp window',
        'type': float,
        'dependencies': ['rare1000bp', 'rare10000bp', 'type'],
        'derive': lambda x: 0 if x['type'] != 'SNV' else math.log10(float(x['rare1000bp']) / float(x['rare10000bp'])),
        'na_value': -1
    },
    'mw-relsngl1': {
        'description': 'Log10 of relative frequency of single occurrence SNV in 100 and 1000 bp window',
        'type': float,
        'dependencies': ['sngl100bp', 'sngl1000bp', 'type'],
        'derive': lambda x: 0 if x['type'] != 'SNV' else math.log10(float(x['sngl100bp']) / float(x['sngl1000bp'])),
        'na_value': -1
    },
    'mw-relsngl2': {
        'description': 'Log10 of relative frequency of single occurrence SNV in 100 and 10000 bp window',
        'type': float,
        'dependencies': ['sngl100bp', 'sngl10000bp', 'type'],
        'derive': lambda x: 0 if x['type'] != 'SNV' else math.log10(float(x['sngl100bp']) / float(x['sngl10000bp'])),
        'na_value': -2
    },
    'mw-relsngl3': {
        'description': 'Log10 of relative frequency of single occurrence SNV in 1000 and 10000 bp window',
        'type': float,
        'dependencies': ['sngl1000bp', 'sngl10000bp', 'type'],
        'derive': lambda x: 0 if x['type'] != 'SNV' else math.log10(float(x['sngl1000bp']) / float(x['sngl10000bp'])),
        'na_value': -1
    },
    'dbscsnv-ada_score': {
        'description': 'Adaboost classifier score from dbscSNV',
        'type': float,
        'na_value': 0,
        'indicator': True
    },
    'dbscsnv-rf_score': {
        'description': 'Random forest classifier score from dbscSNV',
        'type': float,
        'na_value': 0,
        'indicator': True
    },
    'remapoverlaptf': {
        'description': 'Remap number of different transcription factors binding',
        'type': int,
        'transformation': lambda x: math.log10(x),
        'na_value': -0.5,
    },
    'remapoverlaptf-old': {
        'description': 'Old remap track that used the wrong library for imputation and thus was not used in CADDv1.4',
        'colname': 'remapoverlaptf',
        'type': int,
        'transformation': lambda x: np.log10(x),
        'na_value': -0.5,
    },
    'remapoverlapcl': {
        'description': 'Remap number of different transcription factor - cell line combinations binding',
        'type': int,
        'transformation': lambda x: math.log10(x),
        'na_value': -0.5,
    },
    'remapoverlapcl-old': {
        'description': 'Old remap track that used the wrong library for imputation and thus was not used in CADDv1.4',
        'colname': 'remapoverlapcl',
        'type': int,
        'transformation': lambda x: np.log10(x),
        'na_value': -0.5,
    },
    "esmscoremissense": {
        "desription": "score difference of alt and ref derived from esm language model esm1v_t33_650M_UR90S_1 of amino acid sequences",
        "type": float,
        "na_value": 0,
    },
    "esmscoreinframe": {
        "desription": "score difference of alt and ref derived from esm language model esm1v_t33_650M_UR90S_2 of amino acid sequences",
        "type": float,
        "na_value": 0,
    },
    "esmscoreframeshift": {
        "desription": "score difference of alt and ref derived from esm language model esm1v_t33_650M_UR90S_ of amino acid sequences",
        "type": float,
        "na_value": 0,
    },
    'regseq_positivemax': {
        'description': 'regulatory sequence model of 7 cell-lines with negatives positives tissues max',
        'dependencies': ['regseq%d' % i for i in range(7)],
        'derive': lambda x: max([float(i) if float(i) > 0 else 0 for i in x.values()]),
        'hcdiff_derive': lambda x: min([float(i) if float(i) < 0 else 0 for i in x.values()]),
        'transformation': abs,
        'type': float,
        'na_value': 0,
        'hcdiff_na_value': 1,
    },
    'regseq_positivemin': {
        'description': 'regulatory sequence model of 7 cell-lines with negatives positives tissues min',
        'copy': 'regseq_positivemax',
        'derive': lambda x: min([float(i) if float(i) > 0 else 0 for i in x.values()]),
        'hcdiff_derive': lambda x: max([float(i) if float(i) < 0 else 0 for i in x.values()]),
        'na_value': 1,
        'hcdiff_na_value': 0,
    },
    'regseq_negativemax': {
        'description': 'regulatory sequence model of 7 cell-lines with negatives negatives tissues max',
        'copy': 'regseq_positivemax',
        'derive': lambda x: max([float(i) if float(i) < 0 else 0 for i in x.values()]),
        'hcdiff_derive': lambda x: min([float(i) if float(i) > 0 else 0 for i in x.values()]),
        'hcdiff_na_value': 1,
    },
    'regseq_negativemin': {
        'description': 'regulatory sequence model of 7 cell-lines with negatives negatives tissues min',
        'copy': 'regseq_positivemax',
        'derive': lambda x: min([float(i) if float(i) < 0 else 0 for i in x.values()]),
        'hcdiff_derive': lambda x: max([float(i) if float(i) > 0 else 0 for i in x.values()]),
        'na_value': 1,
        'hcdiff_na_value': 0,
    },
    'regseq_positivemean': {
        'description': 'regulatory sequence model of 7 cell-lines with negatives positive tissues mean',
        'copy': 'regseq_positivemax',
        'derive': lambda x: statistics.mean([float(i) if float(i) > 0 else 0 for i in x.values()]),
        'hcdiff_derive': lambda x: statistics.mean([float(i) if float(i) < 0 else 0 for i in x.values()]),
        'hcdiff_na_value': 0,
    },
    'regseq_negativemean': {
        'description': 'regulatory sequence model of 7 cell-lines with negatives negatives tissues mean',
        'copy': 'regseq_positivemax',
        'derive': lambda x: statistics.mean([float(i) if float(i) < 0 else 0 for i in x.values()]),
        'hcdiff_derive': lambda x: statistics.mean([float(i) if float(i) > 0 else 0 for i in x.values()]),
        'hcdiff_na_value': 0,
    },
    'regseq_positivestd': {
        'description': 'regulatory sequence model of 7 cell-lines with negatives positive tissues std',
        'copy': 'regseq_positivemax',
        'derive': lambda x: statistics.stdev([float(i) if float(i) > 0 else 0 for i in x.values()]),
        'hcdiff_derive': lambda x: statistics.stdev([float(i) if float(i) < 0 else 0 for i in x.values()]),
        'hcdiff_na_value': 0,
    },
    'regseq_negativestd': {
        'description': 'regulatory sequence model of 7 cell-lines with negatives negatives tissues std',
        'copy': 'regseq_positivemax',
        'derive': lambda x: statistics.stdev([float(i) if float(i) < 0 else 0 for i in x.values()]),
        'hcdiff_derive': lambda x: statistics.stdev([float(i) if float(i) > 0 else 0 for i in x.values()]),
        'hcdiff_na_value': 0,
    },
    'regseq_positive': {
        'description': 'regulatory sequence model of 7 cell-lines with negatives positives negatives max',
        'dependencies': ['regseq7'],
        'derive': lambda x: float(x['regseq7']) if float(x['regseq7']) > 0 else 0,
        'hcdiff_derive': lambda x: float(x['regseq7']) if float(x['regseq7']) < 0 else 0,
        'transformation': abs,
        'type': float,
        'na_value': 0,
    },
    'regseq_negative': {
        'description': 'regulatory sequence model of 7 cell-lines with negatives negatives negatives max',
        'copy': 'regseq_positive',
        'derive': lambda x: float(x['regseq7']) if float(x['regseq7']) < 0 else 0,
        'hcdiff_derive': lambda x: float(x['regseq7']) if float(x['regseq7']) > 0 else 0,
    },
    "aparent2": {
        "description": "Aparent2 score",
        "type": float,
        "na_value": -0.0013,
        "scaling": "quantile",
    },
    "zoopriphylop": {
        "description": "ZooPriPhyloP score",
        "type": float,
        "na_value": 0.0050,
        "scaling": "quantile",
    },
    "zooverphylop": {
        "description": "ZooVerPhyloP score",
        "type": float,
        "na_value": -0.1460,
        "scaling": "quantile",
    },
    "zoorocc": {
        "description": "ZooRoCC score",
        "type": int,
        "na_value": 0,  # median 51,
        "scaling": "quantile",
    },
    "zoouce": {
        "description": "ZooUCE score",
        "type": int,
        "na_value": 0,
        "scaling": "quantile",
    },
    "roulette-filter": {
        "description": 'ID=low: Low quality regions as determined by gnomAD sequencing metrics. Mappability<0.5;overlap with>50nt simple repeat;ReadPosRankSum>1;0 SNVs in 100bp window. SFS_bump: Pentamer context with abnormal SFS. The fraction of high-frequency SNVS [0.0005<MAF<=0.2] is greater than 1.5x mutation rate controlled average. Tends to be repetitive contexts.TFBS:"Transcription factor binding site as determined by overlap with ChIP-seq peaks',
        "type": list,
        "categories": ["low", "SFS_bump", "TFBS", "N", "high"],
        "na_value": "N",
    },
    "roulette-mr": {
        "description": "Roulette mutation rate estimate",
        "type": float,
        "na_value": 0.0940,
        "scaling": "quantile",
    },
    "roulette-ar": {
        "description": "Adjusted Roulette mutation rate estimate",
        "type": float,
        "na_value": 0.1050,
        "scaling": "quantile",
    },
    # combination tracks that are built from other tracks
    'conseq-scores': {  # combination track of consequences and different scoring categories
        'description': 'combination track of consequences and different scoring categories',
        'type': 'combined',
        'base': 'consequence',
        'child': ['bstatistic', 'cdnapos', 'cdspos', 'dst2splice', 'gerpn',
                  'gerps', 'mamphcons', 'mamphylop', 'mindisttse', 'mindisttss',
                  'priphcons', 'priphylop', 'protpos', 'relcdnapos', 'relcdspos',
                  'relprotpos', 'verphcons', 'verphylop', 'dist2mutation',
                  'freq100bp', 'rare100bp', 'sngl100bp',
                  'freq1000bp', 'rare1000bp', 'sngl1000bp',
                  'freq10000bp', 'rare10000bp', 'sngl10000bp',
                  'spliceai-acc-gain', 'spliceai-acc-loss', 'spliceai-don-gain', 'spliceai-don-loss',
                  'mmsplice-acceptorintron', 'mmsplice-acceptor', 'mmsplice-exon', 'mmsplice-donor', 'mmsplice-donorintron',
                  "esmscoremissense", "esmscoreinframe", "esmscoreframeshift",
                  "aparent2",
                  "zoopriphylop", "zooverphylop", "zoorocc", "zoouce",
                  "roulette-mr", "roulette-ar",
                  ]
    },
    'splicetype2dst': {
        'description': 'combination track of Dst2SplType and Dst2Splice',
        'type': 'combined',
        'base': 'dst2spltype',
        'child': ['dst2splice']
    },
    'nuccomb': {
        'description': 'combinatorical of Ref and Alt',
        'type': list,
        'categories': ['%s-%s' % (a, b) for a in NUCLEOTIDES[:-1] for b in NUCLEOTIDES[:-1] if a != b] + ['N-N'],
        'dependencies': ['ref', 'alt', 'type'],
        'derive': lambda x: 'N-N' if x['type'] != 'SNV' else '%s-%s' % (x['ref'], x['alt']),
        'na_value': 'N-N',
        'hcdiff_derive': lambda x: 'N-N' if x['type'] != 'SNV' else '%s-%s' % (x['alt'], x['ref'])
    },
    'aacomb': {
        'description': 'combinatorical of oAA and nAA',
        'type': list,
        'categories': AMINOACID_EXCHANGES + ['NA'],
        'dependencies': ['oaa', 'naa', 'type'],
        'derive': lambda x: 'NA' if x['type'] != 'SNV' else '%s-%s' % (x['oaa'], x['naa']),
        'na_value': 'NA',
        'hcdiff_derive': lambda x: 'NA' if x['type'] != 'SNV' else '%s-%s' % (x['naa'], x['oaa'])
    },
}

# copy track:
# if a track is a copy of another track, all fields from the parent that are not in the child are copied
for childname in trackData.keys():
    child = trackData[childname]
    if 'copy' in child.keys():
        parent = trackData[child['copy']]
        for key in parent.keys():
            if not key in child.keys():
                trackData[childname][key] = parent[key]