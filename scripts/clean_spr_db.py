#!/usr/bin/env python3
import pandas as pd
import pickle, os

# path to your full‐cache keys
full_cache = 'data/SKEMPI2/SKEMPI2_cache/entries_cache'
keys = pickle.load(open(os.path.join(full_cache, 'keys.pkl'), 'rb'))

# extract the PDB‐codes for which we actually have both wt_ and mt_ in the DB
wt_codes = {k.split('_',1)[1] for k in keys if k.startswith('wt_')}
mt_codes = {k.split('_',1)[1] for k in keys if k.startswith('mt_')}
valid_codes = wt_codes & mt_codes

# load your SPR‐only CSV and filter
spr_csv   = 'data/SKEMPI2/SKEMPI2_SPR.csv'
clean_csv = 'data/SKEMPI2/SKEMPI2_SPR_clean.csv'
df = pd.read_csv(spr_csv, dtype=str)
df_clean = df[df['#Pdb'].isin(valid_codes)].copy()
df_clean.to_csv(clean_csv, index=False)

print(f"Dropped {len(df)-len(df_clean)} orphan entries; kept {len(df_clean)} rows in {clean_csv}")
