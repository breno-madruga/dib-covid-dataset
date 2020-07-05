##############################################################################################
################### Merging the datasets of publications related to COVID-19 #################
##############################################################################################

########################################################################
# Importing the required libraries.
import csv, re, pandas as pd, numpy as np
from string import punctuation
########################################################################

########################################################################
# 1. Defining the required functions
########################################################################

# Defining the function "clean_title".
def clean_title(title):
    if len(title) == 1 and title in punctuation:
        return None
    if title[0] in punctuation:
        title = title[1:]
    if title[-1] in punctuation:
        title = title[:-1]
    return re.sub(r"\s+", " ", title).lower()

########################################################################
# 2. Getting and preprocessing the datasets
########################################################################

########################################################################
# 2.1. arXiv
########################################################################

# Getting the data.
df_arxiv = pd.read_csv("../data/prepared/arxiv_covid_19.csv", header=0,
    dtype={"id": "str"})

# Changing the type of features.
df_arxiv.loc[:, ["subject_areas", "authors"]] = df_arxiv.loc[:,
    ["subject_areas", "authors"]].apply(lambda x: x.apply(eval))
df_arxiv.publication_date = pd.to_datetime(df_arxiv.publication_date)

# Defining the feature "source".
df_arxiv["source"] = "arXiv"

# Normalizing the feature "title".
df_arxiv.title = df_arxiv.title.apply(clean_title)

########################################################################
# 2.2. bioRxiv
########################################################################

# Getting the data.
df_biorxiv = pd.read_csv("../data/prepared/biorxiv_covid_19.csv", header=0,
    dtype={"id": "str"})

# Changing the type of features.
df_biorxiv.author_affil.loc[df_biorxiv.author_affil.notnull()] = df_biorxiv.author_affil.loc[
    df_biorxiv.author_affil.notnull()].apply(eval)
df_biorxiv.publication_date = pd.to_datetime(df_biorxiv.publication_date)

# Normalizing the feature "title".
df_biorxiv.title = df_biorxiv.title.apply(clean_title)

########################################################################
# 2.3. PubMed
########################################################################

# Getting the data.
df_pubmed = pd.read_csv("../data/prepared/pubmed_covid_19.csv", header=0,
    dtype={"pubmed_id": "str"})

# Changing the type of features.
df_pubmed.auth_keywords.loc[df_pubmed.auth_keywords.notnull()] = df_pubmed.auth_keywords.loc[
    df_pubmed.auth_keywords.notnull()].apply(eval)
df_pubmed.author_affil.loc[df_pubmed.author_affil.notnull()] = df_pubmed.author_affil.loc[
    df_pubmed.author_affil.notnull()].apply(eval)
df_pubmed.publication_date = pd.to_datetime(df_pubmed.publication_date)

# Defining the feature "source".
df_pubmed["source"] = "PubMed"

# Normalizing the feature "title".
df_pubmed.title.loc[df_pubmed.title.notnull()] = df_pubmed.title.loc[
    df_pubmed.title.notnull()].apply(clean_title)


########################################################################
# 2.4. Scopus
########################################################################

# Getting the data.
df_scopus = pd.read_csv("../data/prepared/scopus_covid_19.csv", header=0,
    dtype={"id": "str", "eid": "str", "pii": "str", "pubmed_id": "str"})

# Changing the type of features.
df_scopus.auth_keywords.loc[df_scopus.auth_keywords.notnull()] = df_scopus.auth_keywords.loc[
    df_scopus.auth_keywords.notnull()].apply(eval)
df_scopus.index_terms.loc[df_scopus.index_terms.notnull()] = df_scopus.index_terms.loc[
    df_scopus.index_terms.notnull()].apply(eval)
df_scopus.affiliations.loc[df_scopus.affiliations.notnull()] = df_scopus.affiliations.loc[
    df_scopus.affiliations.notnull()].apply(eval)
df_scopus.subject_areas.loc[df_scopus.subject_areas.notnull()] = df_scopus.subject_areas.loc[
    df_scopus.subject_areas.notnull()].apply(eval)
df_scopus.authors.loc[df_scopus.authors.notnull()] = df_scopus.authors.loc[
    df_scopus.authors.notnull()].apply(eval)
df_scopus.author_affil.loc[df_scopus.author_affil.notnull()] = df_scopus.author_affil.loc[
    df_scopus.author_affil.notnull()].apply(eval)
df_scopus.references.loc[df_scopus.references.notnull()] = df_scopus.references.loc[
    df_scopus.references.notnull()].apply(eval)
df_scopus.publication_date = pd.to_datetime(df_scopus.publication_date)

# Defining the feature "source".
df_scopus["source"] = "Scopus"

# Normalizing the feature "title".
df_scopus.title = df_scopus.title.apply(clean_title)

########################################################################
# 3. Merging/Joining the datasets
########################################################################

# Removing the duplicated records between arXiv and Scopus.
df_arxiv = df_arxiv[~df_arxiv.title.isin(df_scopus.title)]

# Removing the duplicated records between bioRxiv and Scopus.
df_biorxiv = df_biorxiv[~df_biorxiv.title.isin(df_scopus.title)]

# Removing the duplicated records between PubMed and Scopus.
idx_removed = df_pubmed.pubmed_id[df_pubmed.pubmed_id.isin(df_scopus.pubmed_id) &
                                  df_pubmed.title.isin(df_scopus.title)].index.to_list()
idx_removed += df_pubmed.pubmed_id[~df_pubmed.pubmed_id.isin(df_scopus.pubmed_id) &
                                   df_pubmed.title.isin(df_scopus.title)].index.to_list()
idx_removed += df_pubmed.pubmed_id[df_pubmed.pubmed_id.isin(df_scopus.pubmed_id) &
                                   ~df_pubmed.title.isin(df_scopus.title)].index.to_list()
df_pubmed = df_pubmed[~df_pubmed.index.isin(list(set(idx_removed)))]

# Visualizing the final number of records for each dataset.
print("arXiv:", df_arxiv.id.size)
print("bioRxiv:", df_biorxiv.id.size)
print("PubMed:", df_pubmed.pubmed_id.size)
print("Scopus:", df_scopus.id.size)
print("Expected total number of records for the final dataset:",
    (df_arxiv.id.size + df_biorxiv.id.size + df_pubmed.pubmed_id.size + df_scopus.id.size))

# Merging/Joining the datasets.
df_final = pd.concat([df_arxiv, df_biorxiv, df_pubmed, df_scopus], ignore_index=True)

# Defining the "None" value for the "NaN" values.
df_final.replace({np.nan: None}, inplace=True)

# Renaming the feature "source".
df_final.rename(columns={"source": "data_source"}, inplace=True)

# Exporting the final dataset to CSV file.
df_final.to_csv("../data/raw/final_raw.csv", index=False, quoting=csv.QUOTE_ALL)