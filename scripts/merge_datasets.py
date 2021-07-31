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
    title = title.lower()
    title = title.replace("€", "").replace("…", "...").replace("τhe", "the").replace(
        "–", "-").replace("‘", "'").replace("“", "\"").replace("”", "\"").replace(
        "′", "'").replace("’", "'").replace("č", "c")
    while title[0] in punctuation or title[0] == " " or title[-1] in punctuation:
        if title[0] in punctuation:
            title = title[1:]
        if title[-1] in punctuation:
            title = title[:-1]
        title = title.strip()
    return re.sub(r"\"+", "", re.sub(r"\s+", " ", title))

########################################################################
# 2. Getting and preprocessing the datasets
########################################################################

########################################################################
# 2.1. arXiv
########################################################################

# Getting the data.
df_arxiv = pd.read_csv("data/prepared/arxiv_covid_19.csv", header=0,
    dtype={"id": "str"})

# Defining the "None" value for the "NaN" values.
df_arxiv.replace({np.nan: None}, inplace=True)

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
df_biorxiv = pd.read_csv("data/prepared/biorxiv_covid_19.csv", header=0,
    dtype={"id": "str"})

# Defining the "None" value for the "NaN" values.
df_biorxiv.replace({np.nan: None}, inplace=True)

# Changing the type of features.
df_biorxiv.author_affil.loc[df_biorxiv.author_affil.notnull()] = df_biorxiv.author_affil.loc[
    df_biorxiv.author_affil.notnull()].apply(eval)
df_biorxiv.subject_areas.loc[df_biorxiv.subject_areas.notnull()] = df_biorxiv.subject_areas.loc[
    df_biorxiv.subject_areas.notnull()].apply(eval)
df_biorxiv.publication_date = pd.to_datetime(df_biorxiv.publication_date)

# Normalizing the feature "title".
df_biorxiv.title = df_biorxiv.title.apply(clean_title)

########################################################################
# 2.3. PubMed
########################################################################

# Getting the data.
df_pubmed = pd.read_csv("data/prepared/pubmed_covid_19.csv", header=0,
    dtype={"pubmed_id": "str"})

# Defining the "None" value for the "NaN" values.
df_pubmed.replace({np.nan: None}, inplace=True)

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
df_scopus = pd.read_csv("data/prepared/scopus_covid_19.csv", header=0, dtype=object)

# Defining the "None" value for the "NaN" values.
df_scopus.replace({np.nan: None}, inplace=True)

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

# Filling the missing values of PubMed's features "title" and "doi" with data from Scopus.
df_pubmed.loc[df_pubmed.pubmed_id.isin(df_scopus.pubmed_id.values) & df_pubmed.title.isnull(), "title"] =     df_pubmed.pubmed_id[df_pubmed.pubmed_id.isin(df_scopus.pubmed_id.values) & df_pubmed.title.isnull()].apply(
        lambda x: df_scopus.title[df_scopus.pubmed_id == x].iloc[0])
df_pubmed.loc[df_pubmed.pubmed_id.isin(df_scopus.pubmed_id.values) & df_pubmed.doi.isnull(), "doi"] =     df_pubmed.pubmed_id[df_pubmed.pubmed_id.isin(df_scopus.pubmed_id.values) & df_pubmed.doi.isnull()].apply(
        lambda x: np.reshape(df_scopus.doi[df_scopus.pubmed_id == x].values, -1)[0] \
            if df_scopus.doi[df_scopus.pubmed_id == x].size > 0 else None)
df_pubmed.loc[df_pubmed.doi[df_pubmed.doi.notnull()].isin(df_scopus.doi[df_scopus.doi.notnull()].values) &
    df_pubmed.title.isnull(), "title"] = df_pubmed.doi[df_pubmed.doi[df_pubmed.doi.notnull()].isin(
        df_scopus.doi[df_scopus.doi.notnull()].values) & df_pubmed.title.isnull()].apply(
            lambda x: df_scopus.title[df_scopus.doi == x].item())

# Filling the missing values of PubMed's features "title", "abstract", "subject_areas" and "doi" with data from bioRxiv.
df_pubmed.loc[df_pubmed.doi.isin(df_biorxiv.doi.values) & df_pubmed.title.isnull(), "title"] =     df_pubmed.doi[df_pubmed.doi.isin(df_biorxiv.doi.values) & df_pubmed.title.isnull()].apply(
        lambda x: df_biorxiv.title[df_biorxiv.doi == x].item())
df_pubmed.loc[df_pubmed.title.isin(df_biorxiv.title.values) & df_pubmed.doi.isnull(), "doi"] =     df_pubmed.loc[df_pubmed.title.isin(df_biorxiv.title.values) & df_pubmed.doi.isnull(), ["doi", "title"]].apply(
        lambda x: df_biorxiv.doi[df_biorxiv.title == x.title].item() if not x.doi else x.doi, axis=1)
df_pubmed.loc[df_pubmed.doi.isin(df_biorxiv.doi.values) & df_pubmed.abstract.isnull(), "abstract"] =     df_pubmed.doi[df_pubmed.doi.isin(df_biorxiv.doi.values) & df_pubmed.abstract.isnull()].apply(
        lambda x: df_biorxiv.abstract[df_biorxiv.doi == x].item())
df_pubmed.loc[df_pubmed.doi.isin(df_biorxiv.doi.values), "subject_areas"] = df_pubmed.doi[
    df_pubmed.doi.isin(df_biorxiv.doi.values)].apply(lambda x: df_biorxiv.subject_areas[
        df_biorxiv.doi == x].item())

# Filling the missing values of PubMed's features "abstract" and "subject_areas" with data from arXiv.
df_pubmed.loc[df_pubmed.title.isin(df_arxiv.title.values) & df_pubmed.abstract.isnull(), "abstract"] =     df_pubmed.title[df_pubmed.title.isin(df_arxiv.title.values) & df_pubmed.abstract.isnull()].apply(
        lambda x: df_arxiv.abstract[df_arxiv.title == x].item())
df_pubmed.loc[df_pubmed.title.isin(df_arxiv.title.values), "subject_areas"] = df_pubmed.title[
    df_pubmed.title.isin(df_arxiv.title.values)].apply(
        lambda x: df_arxiv.subject_areas[df_arxiv.title == x].item())

# Filling the missing values of Scopus' features "abstract" and "subject_areas" with data from arXiv.
df_scopus.loc[df_scopus.title.isin(df_arxiv.title.values) & df_scopus.abstract.isnull(), "abstract"] =     df_scopus.title[df_scopus.title.isin(df_arxiv.title.values) & df_scopus.abstract.isnull()].apply(
        lambda x: df_arxiv.abstract[df_arxiv.title == x].item())
df_scopus.loc[df_scopus.title.isin(df_arxiv.title.values) & df_scopus.subject_areas.isnull(),
    "subject_areas"] = df_scopus.title[df_scopus.title.isin(df_arxiv.title.values) &
        df_scopus.subject_areas.isnull()].apply(lambda x: df_arxiv.subject_areas[df_arxiv.title == x].item())

# Filling the missing values of Scopus' features "doi" and "pubmed_id" with data from PubMed.
df_scopus.loc[df_scopus.pubmed_id.isin(df_pubmed.pubmed_id.values) & df_scopus.doi.isnull(), "doi"] =     df_scopus.loc[df_scopus.pubmed_id.isin(df_pubmed.pubmed_id.values) & df_scopus.doi.isnull(),
        ["doi", "pubmed_id"]].apply(lambda x: df_pubmed.doi[df_pubmed.pubmed_id == x.pubmed_id].item() \
            if not x.doi else x.doi, axis=1)
df_scopus.loc[df_scopus.title.isin(df_pubmed.title[df_pubmed.title.notnull()].values)
    & df_scopus.pubmed_id.isnull() & df_scopus.doi[
        df_scopus.doi.notnull()].isin(df_pubmed.doi[df_pubmed.doi.notnull()].values), "pubmed_id"] = \
df_scopus.loc[df_scopus.title.isin(df_pubmed.title[df_pubmed.title.notnull()].values)
    & df_scopus.pubmed_id.isnull() & df_scopus.doi[
        df_scopus.doi.notnull()].isin(df_pubmed.doi[df_pubmed.doi.notnull()].values),
    ["pubmed_id", "title", "doi"]].apply(lambda x: x.pubmed_id if x.pubmed_id else np.reshape(
        df_pubmed.pubmed_id[(df_pubmed.title == x.title) & (df_pubmed.doi == x.doi)].values, -1)[0] \
            if df_pubmed.pubmed_id[(df_pubmed.title == x.title) & (df_pubmed.doi == x.doi)].size > 0 \
                else None, axis=1)

# Filling the missing values of Scopus' feature "abstract" with data from PubMed.
df_scopus.loc[df_scopus.pubmed_id.isin(df_pubmed.pubmed_id.values) & df_scopus.abstract.isnull(), "abstract"] =     df_scopus.loc[df_scopus.pubmed_id.isin(df_pubmed.pubmed_id.values) & df_scopus.abstract.isnull(),
        ["abstract", "pubmed_id"]].apply(lambda x: df_pubmed.abstract[
            df_pubmed.pubmed_id == x.pubmed_id].item() if not x.abstract else x.abstract, axis=1)
df_scopus.loc[~df_scopus.pubmed_id.isin(df_pubmed.pubmed_id.values) &
    df_scopus.title.isin(df_pubmed.title[df_pubmed.title.notnull()].values) & df_scopus.abstract.isnull() &
    df_scopus.doi[df_scopus.doi.notnull()].isin(df_pubmed.doi[df_pubmed.doi.notnull()].values), "abstract"] = \
df_scopus.loc[~df_scopus.pubmed_id.isin(df_pubmed.pubmed_id.values) &
    df_scopus.title.isin(df_pubmed.title[df_pubmed.title.notnull()].values) & df_scopus.abstract.isnull() &
    df_scopus.doi[df_scopus.doi.notnull()].isin(df_pubmed.doi[df_pubmed.doi.notnull()].values),
    ["abstract", "title", "doi"]].apply(lambda x: x.abstract if not x.abstract else np.reshape(
        df_pubmed.abstract[(df_pubmed.title == x.title) & (df_pubmed.doi == x.doi)].values, -1)[0] \
            if df_pubmed.abstract[(df_pubmed.title == x.title) & (df_pubmed.doi == x.doi)].size > 0 \
                else None, axis=1)

# Filling the missing values of Scopus' feature "auth_keywords" with data from PubMed.
df_scopus.loc[df_scopus.pubmed_id.isin(df_pubmed.pubmed_id.values) & df_scopus.auth_keywords.isnull(),
    "auth_keywords"] = df_scopus.loc[df_scopus.pubmed_id.isin(df_pubmed.pubmed_id.values) &
        df_scopus.auth_keywords.isnull(), ["auth_keywords", "pubmed_id"]].apply(
            lambda x: df_pubmed.auth_keywords[df_pubmed.pubmed_id == x.pubmed_id].item() \
                if not x.auth_keywords else x.auth_keywords, axis=1)
df_scopus.loc[~df_scopus.pubmed_id.isin(df_pubmed.pubmed_id.values) &
    df_scopus.title.isin(df_pubmed.title[df_pubmed.title.notnull()].values) & df_scopus.auth_keywords.isnull() &
    df_scopus.doi[df_scopus.doi.notnull()].isin(df_pubmed.doi[df_pubmed.doi.notnull()].values), "auth_keywords"] = \
df_scopus.loc[~df_scopus.pubmed_id.isin(df_pubmed.pubmed_id.values) &
    df_scopus.title.isin(df_pubmed.title[df_pubmed.title.notnull()].values) & df_scopus.auth_keywords.isnull() &
    df_scopus.doi[df_scopus.doi.notnull()].isin(df_pubmed.doi[df_pubmed.doi.notnull()].values),
    ["auth_keywords", "title", "doi"]].apply(lambda x: x.auth_keywords if x.auth_keywords else np.reshape(
        df_pubmed.auth_keywords[(df_pubmed.title == x.title) & (df_pubmed.doi == x.doi)].values, -1)[0] \
            if df_pubmed.auth_keywords[(df_pubmed.title == x.title) & (df_pubmed.doi == x.doi)].size > 0 \
                else None, axis=1)

# Filling the missing values of Scopus' features "author_affil" and "subject_areas" with data from PubMed.
df_scopus.loc[df_scopus.pubmed_id.isin(df_pubmed.pubmed_id.values) & df_scopus.author_affil.isnull(),
    "author_affil"] = df_scopus.loc[df_scopus.pubmed_id.isin(df_pubmed.pubmed_id.values) &
        df_scopus.author_affil.isnull(), ["author_affil", "pubmed_id"]].apply(
            lambda x: df_pubmed.author_affil[df_pubmed.pubmed_id == x.pubmed_id].item() \
                if not x.author_affil else x.author_affil, axis=1)
df_scopus.loc[df_scopus.pubmed_id.isin(df_pubmed.pubmed_id.values) & df_scopus.subject_areas.isnull(),
    "subject_areas"] = df_scopus.loc[df_scopus.pubmed_id.isin(df_pubmed.pubmed_id.values) &
        df_scopus.subject_areas.isnull(), ["subject_areas", "pubmed_id"]].apply(
            lambda x: df_pubmed.subject_areas[df_pubmed.pubmed_id == x.pubmed_id].item() \
                if not x.subject_areas else x.subject_areas, axis=1)

# Removing the duplicated records between arXiv and bioRxiv.
df_arxiv = df_arxiv[~df_arxiv.title.isin(df_biorxiv.title.values)]

# Removing the duplicated records between arXiv and PubMed.
df_arxiv = df_arxiv[~df_arxiv.title.isin(df_pubmed.title[df_pubmed.title.notnull()].values)]

# Removing the duplicated records between arXiv and Scopus.
df_arxiv = df_arxiv[~df_arxiv.title.isin(df_scopus.title.values)]

# Removing the duplicated records between bioRxiv and PubMed.
df_biorxiv = df_biorxiv[~(df_biorxiv.title.isin(df_pubmed.title[df_pubmed.title.notnull()].values) &
    df_biorxiv.doi.isin(df_pubmed.doi[df_pubmed.doi.notnull()].values))]

# Removing the duplicated records between bioRxiv and Scopus.
df_biorxiv = df_biorxiv[~(df_biorxiv.title.isin(df_scopus.title.values) &
    df_biorxiv.doi.isin(df_scopus.doi[df_scopus.doi.notnull()].values))]

# Removing the duplicated records between PubMed and Scopus.
idx_removed = df_pubmed.pubmed_id[df_pubmed.pubmed_id.isin(df_scopus.pubmed_id[
    df_scopus.pubmed_id.notnull()].values)].index.to_list()
idx_removed += df_pubmed.pubmed_id[~df_pubmed.pubmed_id.isin(df_scopus.pubmed_id[
        df_scopus.pubmed_id.notnull()].values) &
    df_pubmed.title.isin(df_scopus.title.values) &
    df_pubmed.doi.isin(df_scopus.doi[df_scopus.doi.notnull()].values)].index.to_list()
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
df_final.to_csv("data/raw/final_raw.csv", index=False, quoting=csv.QUOTE_ALL)