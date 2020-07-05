
##############################################################################################
########### Cleaning and Preprocessing the bioRxiv publications related to COVID-19 ##########
##############################################################################################

# The publications' data were collected from bioRxiv webpage
# (https://connect.biorxiv.org/relate/content/181) related to COVID-19.

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

class ProcessBiorxiv(Preprocess):
    # Cleaning and preprocessing the dataframe.
    def _preprocess(self):
        # Removing unnecessary columns.
        self._dataframe.drop(axis=1, columns="rel_num_authors", inplace=True)

        # Renaming the columns.
        columns = {"rel_title": "title", "rel_doi": "doi", "rel_link": "id", "rel_abs": "abstract",
            "rel_authors": "author_affil", "rel_date": "publication_date", "rel_site": "source"}
        self._dataframe.rename(columns=columns, inplace=True)

        # Normalizing the feature "id".
        self._dataframe.id = self._dataframe.id.apply(lambda x: x.split("/")[-1])

        # Normalizing the features "title" and "abstract".
        self._dataframe.loc[:, ["title", "abstract"]] = self._dataframe.loc[:, ["title", "abstract"]
            ].apply(lambda x: x.apply(lambda y: re.sub("/r/", "",
                re.sub("@PER@CENT@", "%", re.sub(r"\^", "",
                    LatexNodes2Text().latex_to_text(re.sub(r"\s+", " ", re.sub("\\\\?%", "@PER@CENT@",
                        re.sub(r"\\href\{(.+)\}\{(.+)\}", "\g<2> \\url{\g<1>}", y))).strip()))))))

        # Changing the type of feature "author_affil".
        self._dataframe.author_affil = self._dataframe.author_affil.apply(eval)

        # Normalizing the feature "author_affil".
        self._dataframe.author_affil = [
            [{"name": re.sub(r"\s+", " ", LatexNodes2Text().latex_to_text(
                    re.sub(r"^\"(.+)\"$", "\g<1>", re.sub("^-\s", "", author["author_name"])))),
                "affiliation": re.sub(r"\s+", " ", LatexNodes2Text().latex_to_text(
                    re.sub(r"^\"(.+)\"$", "\g<1>", re.sub("Affiliation:", "",
                        re.sub(r"[0-9]+\.\s", "", author["author_inst"]), flags=re.IGNORECASE))))}
                for author in authors] if len(authors) > 0 else None
            for authors in self._dataframe.author_affil]

        # Removing the invalid authors and affiliations.
        invalid_authors = ["Revision Created", "Revision Converted", "Newly Submitted Revision",
                            "Final Decision"]
        for idx, authors in self._dataframe.author_affil.iteritems():
            if authors:
                for author in list(authors):
                    if author["name"].strip() in invalid_authors:
                        authors.remove(author)
                    elif not author["affiliation"] or author["affiliation"].lower().replace(
                                ".", "") == "none":
                        author["affiliation"] = None
                self._dataframe.author_affil[idx] = tuple(authors)

        # Defining the "None" value for the "NaN" values.
        self._dataframe.replace(
            {np.nan: None, "none": None, "none.": None, "None": None}, inplace=True)