##############################################################################################
############ Cleaning and Preprocessing the PubMed publications related to COVID-19 ##########
##############################################################################################

# For collecting the PubMed publications related to COVID-19, we used the "pymed" library.
# It is avaliable on [https://pypi.org/project/pymed/].

########################################################################
# Uncomment to install the library.
# %pip install pylatexenc
########################################################################

########################################################################
# Importing the required libraries.
import re, numpy as np
from pylatexenc.latex2text import LatexNodes2Text
from preprocess import Preprocess
########################################################################

class ProcessPubmed(Preprocess):

    # Defining the function "clean_text" for cleaning and preprocessing any text.
    @staticmethod
    def __clean_text(text):
        if text:
            return re.sub(r"\s+", " ", re.sub("[0-9]*\u200b", "", str(text)).replace(
                "\u2009", " ").replace("\xa0", " ").replace("\n", " ").replace(
                "\ufeff", "").replace("\u202f", "").replace("\u2028", " ").replace(
                "\u200f", "")).strip()
        else:
            return None

    def _preprocess(self):
        # Defining the "None" value for the "NaN" values.
        self._dataframe.replace({np.nan: None}, inplace=True)

        # Removing unnecessary columns.
        columns_drop = ["methods", "conclusions", "results", "copyrights", "xml", "isbn",
                        "language", "publication_type", "sections", "publisher", "publisher_location"]
        self._dataframe.drop(axis=1, columns=columns_drop, inplace=True)

        # Getting the PubMed ID for each paper.
        self._dataframe.pubmed_id = self._dataframe.pubmed_id.apply(lambda x: x.split()[0].strip())

        # Normalizing the features "abstract" and "title".
        self._dataframe.abstract = self._dataframe.abstract.apply(
            lambda x: LatexNodes2Text().latex_to_text(
                re.sub(r"\s+", " ", re.sub("%", "\\%", x))) if x and len(x) > 0 else None)
        self._dataframe.title = self._dataframe.title.apply(
            lambda x: x.replace("\n", " ") if x and len(x) > 0 else None)

        # Setting the feature "keywords" as a tuple of keywords and
        # normalizing the keywords for each paper.
        self._dataframe.keywords.loc[self._dataframe.keywords.notnull()] = [
            tuple([ProcessPubmed.__clean_text(keyword) for keyword in eval(keywords)]) \
                if eval(keywords) else None
            for keywords in self._dataframe.keywords[self._dataframe.keywords.notnull()]]

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

                    if "affiliation" in author:
                        auth["affiliation"] = ProcessPubmed.__clean_text(author["affiliation"])
                    else:
                        auth["affiliation"] = None
                    
                    if "name" in auth:
                        list_authors.append(auth)
                if list_authors:
                    self._dataframe.authors[idx] = tuple(list_authors)
                else:
                    self._dataframe.authors[idx] = None

        # Renaming the features "authors", "keywords" and "journal".
        self._dataframe.rename(columns={"authors": "author_affil", "keywords": "auth_keywords",
            "journal": "vehicle_name"}, inplace=True)