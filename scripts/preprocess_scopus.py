##############################################################################################
############ Cleaning and Preprocessing the Scopus publications related to COVID-19 ##########
##############################################################################################

# For collecting the Scopus publications related to COVID-19, we used the "pybliometrics"
# library. It is avaliable on https://pypi.org/project/pybliometrics/.

########################################################################
# Importing the required libraries.
import re, pandas as pd, numpy as np
from preprocess import Preprocess
########################################################################

class ProcessScopus(Preprocess):
    # Cleaning and preprocessing the dataframe.
    def _preprocess(self):
        # Removing the invalid articles.
        self._dataframe = self._dataframe.loc[
            self._dataframe.id.notnull() & self._dataframe.eid.notnull()]

        # Defining the "None" value for the "NaN" values.
        self._dataframe.replace({np.nan: None}, inplace=True)

        # Defining the "zero" value for the articles without numbers of citation and references.
        self._dataframe.citation_num.loc[self._dataframe.citation_num.isnull()] = 0
        self._dataframe.ref_count.loc[self._dataframe.ref_count.isnull()] = 0

        # Normalizing the feature "abstract".
        self._dataframe.abstract.loc[
            self._dataframe.abstract.isnull() & self._dataframe.description.notnull()
        ] = self._dataframe.description.loc[
            self._dataframe.abstract.isnull() & self._dataframe.description.notnull()]

        # Normalizing the feature "vehicle_name".
        self._dataframe.vehicle_name.loc[
            self._dataframe.conference_name.notnull() & self._dataframe.vehicle_name.notnull()
        ] = self._dataframe.conference_name.loc[
            self._dataframe.conference_name.notnull() & self._dataframe.vehicle_name.notnull()]

        # Removing unnecessary columns.
        columns_drop = ["eid", "pii", "description", "isbn", "conf_location", "conference_name",
            "vehicle_address", "title_edition"]
        self._dataframe.drop(axis=1, columns=columns_drop, inplace=True)

        # Changing the type of features.
        self._dataframe.loc[:, ["citation_num", "ref_count"]] = self._dataframe.loc[:,
            ["citation_num", "ref_count"]].astype("int")
        self._dataframe.auth_keywords.loc[self._dataframe.auth_keywords.notnull()] = \
            self._dataframe.auth_keywords.loc[self._dataframe.auth_keywords.notnull()].apply(eval)
        self._dataframe.index_terms.loc[self._dataframe.index_terms.notnull()] = \
            self._dataframe.index_terms.loc[self._dataframe.index_terms.notnull()].apply(eval)
        self._dataframe.affiliations.loc[self._dataframe.affiliations.notnull()] = \
            self._dataframe.affiliations.loc[self._dataframe.affiliations.notnull()].apply(eval)
        self._dataframe.subject_areas.loc[self._dataframe.subject_areas.notnull()] = \
            self._dataframe.subject_areas.loc[self._dataframe.subject_areas.notnull()].apply(eval)
        self._dataframe.authors.loc[self._dataframe.authors.notnull()] = \
            self._dataframe.authors.loc[self._dataframe.authors.notnull()].apply(eval)
        self._dataframe.author_affil.loc[self._dataframe.author_affil.notnull()] = \
            self._dataframe.author_affil.loc[self._dataframe.author_affil.notnull()].apply(eval)
        self._dataframe.references.loc[self._dataframe.references.notnull()] = \
            self._dataframe.references.loc[self._dataframe.references.notnull()].apply(eval)
        self._dataframe.publication_date = pd.to_datetime(self._dataframe.publication_date)

        # Normalizing the feature "abstract".
        self._dataframe.abstract.loc[self._dataframe.abstract.notnull()] = \
            self._dataframe.abstract.loc[self._dataframe.abstract.notnull()].apply(
                lambda x: x.replace("\\u0019", "").replace("\\%", "%").replace("\\s", "s").strip())

        # Normalizing the itens contained in the features "auth_keywords" and "index_terms".
        self._dataframe.auth_keywords.loc[self._dataframe.auth_keywords.notnull()] = \
            self._dataframe.auth_keywords.loc[self._dataframe.auth_keywords.notnull()].apply(
                lambda x: tuple([item.replace("\ufeff", "").strip() for item in x]))
        self._dataframe.index_terms.loc[self._dataframe.index_terms.notnull()] = \
            self._dataframe.index_terms.loc[self._dataframe.index_terms.notnull()].apply(
                lambda x: tuple([item.replace("\ufeff", "").strip() for item in x]))

        # Normalizing the affiliations contained in the features "affiliations" and "author_affil".
        self._dataframe.affiliations.loc[self._dataframe.affiliations.notnull()] = \
            self._dataframe.affiliations.loc[self._dataframe.affiliations.notnull()].apply(
                lambda x: tuple([{"id": affil["id"],
                    "affiliation": affil["affiliation"].replace("\u200b", "").replace(
                        "\u202f", "").strip(),
                    "country": affil["country"]}
                for affil in x]))
        self._dataframe.author_affil.loc[self._dataframe.author_affil.notnull()] = \
            self._dataframe.author_affil.loc[self._dataframe.author_affil.notnull()].apply(
                lambda x: tuple([{"id": item["id"], "name": item["name"],
                    "affil_id": item["affil_id"], "affiliation": item["affiliation"].replace(
                        "\u200b", "").replace("\u202f", "").strip() if item["affiliation"] else None,
                    "country": item["country"]}
                for item in x]))