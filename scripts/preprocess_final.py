##############################################################################################
###### Cleaning and Preprocessing the final dataset of publications related to COVID-19 ######
##############################################################################################

########################################################################
# Importing the required libraries.
import csv, pandas as pd, numpy as np
from preprocess import Preprocess
########################################################################

class ProcessFinal(Preprocess):
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

        # Defining the "zero" value for the articles without numbers of citation and references.
        self._dataframe.citation_num.loc[self._dataframe.citation_num.isnull()] = 0
        self._dataframe.ref_count.loc[self._dataframe.ref_count.isnull()] = 0

        # Extracting the missing authors from the feature "author_affil".
        self._dataframe.authors.loc[
            self._dataframe.authors.isnull() & self._dataframe.author_affil.notnull()] = [
        tuple([{"name": author["name"]} for author in authors if author["name"]])
            for authors in self._dataframe.author_affil[
                self._dataframe.authors.isnull() & self._dataframe.author_affil.notnull()]]

        # Removing the empty lists of authors.
        self._dataframe.authors.loc[self._dataframe.authors == ()] = None

        # Extracting the missing affiliations from the feature "author_affil".
        self._dataframe.affiliations.loc[
            self._dataframe.affiliations.isnull() & self._dataframe.author_affil.notnull()] = [
        tuple([{"affiliation": affil["affiliation"]} for affil in affils if affil["affiliation"]])
            for affils in self._dataframe.author_affil[
                self._dataframe.affiliations.isnull() & self._dataframe.author_affil.notnull()]]

        # Removing the empty lists of affiliations.
        self._dataframe.affiliations.loc[self._dataframe.affiliations == ()] = None

        # Defining the "None" value for the "NaN" values.
        self._dataframe.replace({np.nan: None}, inplace=True)