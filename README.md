# DiB Covid-19 Production Dataset

## Description

This repository is responsible for versioning of the required source code to generate the COVID-19 manuscripts dataset, which it was published in the DiB (Data in Brief) journal.

## Input

The raw data collected by the Jupyter Notebooks, which are contained into the folder "notebooks/collect".

The data sources are:
- [arXiv](https://arxiv.org/covid19search)
- [bioRxiv/medRxiv](https://connect.biorxiv.org/relate/content/181)
- [PubMed](https://pubmed.ncbi.nlm.nih.gov)
- [Scopus](https://www.scopus.com/)

## Output

The final dataset is combination of the arXiv, bioRxiv, medRxiv, PubMed and Scopus datasets collected. It is generated by the use of DVC pipeline defined in this repository.

The features of the resulting dataset are:
* id (identifier): the identifier key of a manuscript.
* doi: the DOI of a manuscript.
* title: the title of a manuscript.
* abstract: the abstract of a manuscript.
* publication_date: the date of publication of a manuscript.
* citation_num: the number of citation of a manuscript.
* language: the language/idiom of a manuscript.
* production_type: the category/type/classification of source of a manuscript.
* source_type: the category/type/classification of source of a manuscript. It is a short version of feature *production_type*.
* auth_keywords: the list of keywords defined by the authors of a manuscript.
* index_terms: the list of indexed terms that defined by Scopus.
* issn: the ISSN/E-ISSN of a manuscript.
* vehicle_name: the name of source where a manuscript was published.
* publisher: the name of publisher that published a manuscript.
* affiliations: the list of affiliations (ID, country and name of affiliation) contained in a manuscript.
* subject_areas: the list of subject/study fields of a manuscript.
* authors: the list of authors (ID and name) contained in a manuscript.
* author_affil: the list of authors organized with their affiliations. The combination of the features *authors* and *affiliations*.
* ref_count: the number of references contained in a manuscript.
* references: the list of references data (authors, title, DOI and ID).
* data_source: the source database of a manuscript.
* period: the combination of the year and month, respectively, extracted from the feature *publication_date*.

## Steps for generating the dataset

For the execution of the following steps, I will consider that you already cloned/downloaded this repository, as well as the steps will be executed via shell/prompt within the folder of this repository. In addition, an essential prerequisite is that DVC is already installed on your machine.

For reusing the raw data that I already collected and the pipeline created, you can do the following steps:

1. Download the raw data, that is available on Google Drive, and put them in the ***data/raw*** folder. You can download these files from this [link](https://drive.google.com/drive/folders/14PzDoJI2YwvxNCowefLxbpEQl0FUY1nm?usp=sharing).

2. Execute the preprocessing pipeline. So, you can execute the following command:
    ```
    dvc repro
    ```

## Citation

[![DOI:10.1016/j.dib.2020.106178](https://zenodo.org/badge/DOI/10.1016/j.dib.2020.106178.svg)](https://doi.org/10.1016/j.dib.2020.106178)

### How does it cite?

Santos, Breno Santana; Silva, Ivanovitch; Ribeiro-Dantas, Marcel da Câmara; Alves, Gisliany; Endo, Patricia Takako; Lima, Luciana. **COVID-19: A scholarly production dataset report for research analysis**. *Data in Brief*, Volume 32, 2020, [DOI:10.1016/j.dib.2020.106178](https://doi.org/10.1016/j.dib.2020.106178).

### How does the article download?

You can download the article from this [link](https://www.sciencedirect.com/science/article/pii/S2352340920310726).