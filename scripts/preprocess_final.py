##############################################################################################
###### Cleaning and Preprocessing the final dataset of publications related to COVID-19 ######
##############################################################################################

########################################################################
# Importing the required libraries.
import pandas as pd, numpy as np
from preprocess import Preprocess
from datetime import date
########################################################################

class ProcessFinal(Preprocess):

    # Function to normalize the affiliations of the authors.
    @staticmethod
    def __normalize_affiliations(row):
        # Getting missing values within the feature "author_affil" from "affiliations" one.
        if row.affiliations and row.author_affil:
            for pos, author in enumerate(row.author_affil):
                for affil in row.affiliations:
                    if affil["id"] and author["affil_id"] and affil["id"] in [af.strip()
                            for af in author["affil_id"].split(",")]:
                        row.author_affil[pos]["affil_id"] = affil["id"]
                        row.author_affil[pos]["affiliation"] = affil["affiliation"]
                        if affil["country"] and not author["country"]:
                            row.author_affil[pos]["country"] = affil["country"]
                        elif affil["country"] != author["country"]:
                            row.author_affil[pos]["country"] = affil["country"]
        else:
            # Getting missing values within the feature "affiliations" from "author_affil" one.
            if row.author_affil:
                affils = set([(author["affil_id"], author["affiliation"], author["country"])
                            for author in row.author_affil
                            if author["affil_id"] or author["affiliation"]])
                if len(affils) > 0:
                    keys = ["id", "affiliation", "country"]
                    row.affiliations = tuple([dict(zip(keys, affil)) for affil in affils])
                else:
                    row.affiliations = None
        return row

    # Function to normalize the name of the authors.
    @staticmethod
    def __normalize_name_authors(row):
        if row.authors and row.author_affil:
            for pos, item in enumerate(row.authors):
                for author in list(row.author_affil):
                    if item["id"] == author["id"]:
                        row.authors[pos]["name"] = author["name"]
        elif row.author_affil:
            authors = set([(author["id"], author["name"]) for author in row.author_affil
                        if author["name"]])
            if len(authors) > 0:
                keys = ["id", "name"]
                row.authors = tuple([dict(zip(keys, author)) for author in authors])
            else:
                row.authors = None

        return row

    # Function to normalize the the authors and their affiliations.
    @staticmethod
    def __normalize_features(row):
        fields = {
            "authors": ["id", "name"],
            "affiliations": ["id", "affiliation", "country"],
            "affil": ["affil_id", "affiliation", "country"]
        }
        # Normalizing the authors.
        records = [tuple([item[f] for f in fields["authors"]]) for item in row.authors] \
            if row.authors else []
        if row.author_affil:
            records = set([*records, *[tuple([item[c] for c in fields["authors"]])
                                            for item in row.author_affil
                                            if item["id"] and item["name"]]])
        elif len(records) > 0 and not row.author_affil:
            row.author_affil = tuple([{**dict(zip(fields["authors"], auth)), "affil_id": None,
                                    "affiliation": None, "country": None} for auth in records])

        if len(records) > 0:
            row.authors = tuple([dict(zip(fields["authors"], auth)) for auth in records])

        # Normalizing the affiliations.
        if row.affiliations:
            records = [tuple([item[c] for c in fields["affiliations"]])
                            for item in row.affiliations]
            if row.author_affil:
                records = set([*records, *[tuple([item[c] for c in fields["affil"]])
                                                for item in row.author_affil
                                                if item["affil_id"] or item["affiliation"]]])
            row.affiliations = tuple([dict(zip(fields["affiliations"], affil))
                                    for affil in records])
        return row

    # Cleaning and preprocessing the final dataset.
    def _preprocess(self):
        # Defining the "None" value for the "NaN" values.
        self._dataframe.replace({np.nan: None}, inplace=True)

        # Changing the type of features.
        self._dataframe.loc[:, ["auth_keywords", "index_terms", "affiliations",
                "subject_areas", "authors", "author_affil", "references"]] = \
            self._dataframe.loc[:, ["auth_keywords", "index_terms", "affiliations",
                "subject_areas", "authors", "author_affil", "references"]].apply(
                    lambda x: x.apply(lambda y: eval(y) if y else None))
        self._dataframe.publication_date = pd.to_datetime(self._dataframe.publication_date)

        # Filtering the data from the start period of COVID-19 pandemic.
        self._dataframe = self._dataframe[
            self._dataframe.publication_date.dt.date >= date(2019, 12, 1)]

        # Creating the feature "period" from the feature "publication_date".
        self._dataframe.loc[self._dataframe.period.isnull(), "period"] = self._dataframe.loc[
            self._dataframe.period.isnull(), "publication_date"].apply(
                lambda x: x.strftime("%Y-%m"))

        # Defining the "zero" value for the articles without numbers of citation and references.
        self._dataframe.citation_num.loc[self._dataframe.citation_num.isnull()] = 0
        self._dataframe.ref_count.loc[self._dataframe.ref_count.isnull()] = 0

        # Applying the function "normalize_name_authors" to the data.
        self._dataframe[["authors", "author_affil"]] = self._dataframe[
            ["authors", "author_affil"]].apply(ProcessFinal.__normalize_name_authors, axis=1)

        # Applying the function "normalize_affiliations" to the data.
        self._dataframe[["affiliations", "author_affil"]] = self._dataframe[
            ["affiliations", "author_affil"]].apply(ProcessFinal.__normalize_affiliations, axis=1)

        # Applying the function "normalize_features" to the data.
        self._dataframe[["authors", "affiliations", "author_affil"]] = self._dataframe[
            ["authors", "affiliations", "author_affil"]].apply(
                ProcessFinal.__normalize_features, axis=1)

        # Normalizing the feature "id".
        self._dataframe.loc[
            self._dataframe.pubmed_id.notnull() & self._dataframe.id.isnull(), "id"] = \
        self._dataframe.pubmed_id[
            self._dataframe.pubmed_id.notnull() & self._dataframe.id.isnull()]

        # Removing the feature "pubmed_id".
        self._dataframe.drop(columns="pubmed_id", inplace=True)

        # Defining the "None" value for the "NaN" values.
        self._dataframe.replace({np.nan: None}, inplace=True)