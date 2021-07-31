##############################################################################################
############ Cleaning and Preprocessing the PubMed publications related to COVID-19 ##########
##############################################################################################

# For collecting the PubMed publications related to COVID-19, we used the "pymed" library.
# It is avaliable on https://pypi.org/project/pymed/.

########################################################################
# Uncomment to install the library.
# %pip install pylatexenc
########################################################################

########################################################################
# Importing the required libraries.
import re, pandas as pd, numpy as np
from pylatexenc.latex2text import LatexNodes2Text
from preprocess import Preprocess
########################################################################

class ProcessPubmed(Preprocess):

    # Defining the function "clean_text" for cleaning and preprocessing any text.
    @staticmethod
    def __clean_text(text):
        if text:
            return re.sub(r"\\", " ", re.sub(r"\s+", " ", re.sub(r"\-{2,}", "-",
            re.sub("[0-9]*\u200b", "", str(text)).replace("\xad", "-")).replace(
                "\u2009", " ").replace("\xa0", " ").replace("\n", " ").replace(
                "\ufeff", "").replace("\u202f", "").replace("\u2028", " ").replace(
                "\u200f", "").replace("\u200e", "").replace("()", "").replace(
                "[]", "").replace("\\'", "\'").replace("\uf06b", "").replace(
                "\x96", "").replace("\u200c", ""))).strip()
        else:
            return None

    # Cleaning and preprocessing the dataframe.
    def _preprocess(self):
        # Defining the "None" value for the "NaN" values.
        self._dataframe.replace({np.nan: None}, inplace=True)

        # Removing unnecessary columns.
        columns_drop = ["methods", "conclusions", "results", "copyrights", "xml", "isbn",
                        "language", "publication_type", "sections", "publisher", "publisher_location"]
        self._dataframe.drop(axis=1, columns=columns_drop, inplace=True)

        # Getting the PubMed ID for each paper.
        self._dataframe.pubmed_id = self._dataframe.pubmed_id.apply(lambda x: x.split()[0].strip())

        # Normalizing the doi for each paper.
        self._dataframe.loc[self._dataframe.doi.notnull(), "doi"] = self._dataframe.loc[
            self._dataframe.doi.notnull(), "doi"].apply(lambda x: x.split()[0].strip())

        # Normalizing the features "abstract", "title" and "journal".
        self._dataframe.abstract = self._dataframe.abstract.apply(
            lambda x: ProcessPubmed.__clean_text(LatexNodes2Text().latex_to_text(
                re.sub(r"\s+", " ", re.sub("%", "\\%", x)))) if x and len(x) > 0 else None)
        self._dataframe.title = self._dataframe.title.apply(
            lambda x: ProcessPubmed.__clean_text(x) if x and len(x) > 0 else None)
        self._dataframe.journal = self._dataframe.journal.apply(ProcessPubmed.__clean_text)

        # Setting the feature "keywords" as a tuple of keywords and
        # normalizing the keywords for each paper.
        self._dataframe.keywords.loc[self._dataframe.keywords.notnull()] = [
            tuple([ProcessPubmed.__clean_text(keyword) for keyword in eval(keywords)]) \
                if eval(keywords) else None
            for keywords in self._dataframe.keywords[self._dataframe.keywords.notnull()]]

        # Removing the invalid keywords.
        self._dataframe.keywords.loc[self._dataframe.keywords.notnull()] = [
            tuple([item for item in keywords if item])
            for keywords in self._dataframe.keywords[self._dataframe.keywords.notnull()]]
        self._dataframe.keywords.loc[self._dataframe.keywords.notnull()] = \
            self._dataframe.keywords.loc[self._dataframe.keywords.notnull()].apply(
                lambda x: x if len(x) > 0 else None)

        # Correcting the feature "authors".
        for idx, authors in enumerate(self._dataframe.authors):
            if not eval(authors):
                self._dataframe.authors[idx] = None
            else:
                list_authors = []
                for author in eval(authors):
                    auth = {}
                    if author["firstname"] and author["lastname"]:
                        auth["name"] = ProcessPubmed.__clean_text(
                            "{} {}".format(author["firstname"], author["lastname"]))
                    elif author["firstname"] and not author["lastname"]:
                        auth["name"] = ProcessPubmed.__clean_text(author["firstname"])
                    elif not author["firstname"] and author["lastname"]:
                        auth["name"] = ProcessPubmed.__clean_text(author["lastname"])
                    else:
                        auth["name"] = None

                    auth["id"] = str(hash("{} - {}".format(auth["name"], "PubMed"))) \
                        if auth["name"] else None
                    auth["affiliation"] = ProcessPubmed.__clean_text(author["affiliation"]) \
                        if "affiliation" in author else None
                    auth["affil_id"] = str(hash("{} - {}".format(auth["affiliation"], "PubMed")))                 if auth["affiliation"] else None
                    auth["country"] = None

                    if auth["affiliation"] or auth["name"]:
                        list_authors.append(auth)

                self._dataframe.authors[idx] = tuple(list_authors) if len(list_authors) > 0 else None

        # Renaming the features "authors", "keywords" and "journal".
        self._dataframe.rename(columns={"authors": "author_affil", "keywords": "auth_keywords",
            "journal": "vehicle_name"}, inplace=True)

        # Removing the duplicated records by features "title" and "doi".
        self._dataframe = pd.concat([
            self._dataframe[self._dataframe.title.isnull() | self._dataframe.doi.isnull()],
            self._dataframe[self._dataframe.title.notnull() & self._dataframe.doi.notnull()].sort_values(
                by=["title", "publication_date"]).drop_duplicates(
                    ["title", "doi"], "last")], ignore_index=True)