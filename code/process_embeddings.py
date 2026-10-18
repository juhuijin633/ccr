#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# in addition to requirements.txt, set up a venv that 
# has sklearn, polars, pandas, pathlib, pyarrow

from pathlib import Path
import pandas as pd
import numpy as np 
import polars as pl
import polars.selectors as cs
import pyarrow as pa 
import sklearn as sk
from sklearn.cross_decomposition import CCA
import re as re

# cd /Users/juhuijin/Desktop/ccr/demand-analysis-repro/code
from paths import PROJECT_DIR 

# there are 21,560 rows for each embedding file (1,540 products × 14 periods = 21,560 rows)

# image = 773 columns 
image = pl.read_parquet(PROJECT_DIR / "data/embeddings/image_embeddings.parquet")

# text = 773 columns 
text = pl.read_parquet(PROJECT_DIR / "data/embeddings/text_embeddings.parquet")

# tabular = 21 columns
tabular = pl.read_parquet(PROJECT_DIR / "data/embeddings/tabular_embeddings.parquet")

#------- Spectral Analysis (Principal Components)

#image = image.select(cs.starts_with("image_")).to_numpy()
#text = text.select(cs.starts_with("text_")).to_numpy()
#tabular = tabular.select(cs.starts_with("tabular_")).to_numpy()

post_process = {
    "none": lambda df: df, 
    "demean": lambda df: df.with_columns( embed_cols - embed_cols.mean() ), 
    "demean_scale": lambda df: df.with_columns( ( embed_cols - embed_cols.mean() ) / embed_cols.std() )
}

# save the singular values from SVD 
# and plot the L2 norm after each post-processing step 

rows_s = []
rows_l2 = []

embed_cols = cs.starts_with("image") | cs.starts_with("text")
image_cols = [ c for c in image.columns if re.fullmatch(r"image_\d+", c) ]
text_cols = [ c for c in text.columns if re.fullmatch(r"text_\d+", c)]

for p in ["none", "demean", "demean_scale"]: 
    
    # add post-processing to image and text embeddings 
    image_p = post_process[p](image).select(image_cols)
    text_p = post_process[p](text).select(text_cols)
    
    #--------- Extract singular values from SVD of image and text 
    image_s = np.linalg.svd(image_p.to_numpy(), compute_uv = False)
    text_s = np.linalg.svd(text_p.to_numpy(), compute_uv = False)
    
    rows_s.append(pd.DataFrame( { "value": image_s, "type": "image", "post_processing": p } ))
    rows_s.append(pd.DataFrame( { "value": text_s, "type": "text", "post_processing": p } ))
    
    #--------- Check the L2 norm (sample variance) of each embedding for image and text
    # this is to check normalization / units 

    image_l2 = image_p.group_by(pl.lit(1)).agg(pl.all().pow(2).mean()).to_numpy().ravel()
    text_l2 = text_p.group_by(pl.lit(1)).agg(pl.all().pow(2).mean()).to_numpy().ravel()
    
    rows_l2.append(pd.DataFrame( { "value": image_l2, "type": "image", "post_processing": p }))
    rows_l2.append(pd.DataFrame( { "value": text_l2, "type": "text", "post_processing": p }))
    

singular_values = pd.concat(rows_s, ignore_index = True).assign(statistic = "s_val")
l2_norms = pd.concat(rows_l2, ignore_index = True).assign(statistic = "l2_norm")

singular_values.to_csv(
    PROJECT_DIR / "data/spectral/singular_values.csv", 
    index = False
)

l2_norms.to_csv(
    PROJECT_DIR / "data/spectral/l2_norms.csv", 
    index = False
)

# plot the singular values and check whether they decay -- this is in the generate_figures.R script! 

#--------- Canonical Correlation Analysis 

# CCA is a model/estimator object from scikit-learn 
# .fit means learn from the data 
# .transform means apply the learned weights to data 

cca = CCA(n_components = 30, scale = True, max_iter = 10000)

image_c, text_c = cca.fit_transform(image, text)

canonical_corrs = [
    np.corrcoef(image_c[:, i], text_c[:, i])[0, 1]
    for i in range(10)
]







