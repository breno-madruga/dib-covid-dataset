##############################################################################################
############ Cleaning and Preprocessing the arXiv publications related to COVID-19 ###########
##############################################################################################

# The publications' data were collected from arXiv webpage (https://arxiv.org/covid19search)
# related to COVID-19.

########################################################################
# Uncomment to install the library.
# %pip install pylatexenc
########################################################################

########################################################################
# Importing the required libraries.
import re, numpy as np
from pylatexenc.latex2text import LatexNodes2Text
from datetime import datetime
from preprocess import Preprocess
########################################################################

class ProcessArxiv(Preprocess):

    # Cleaning and preprocessing the dataframe.
    def _preprocess(self):
        # Defining the "None" value for the "NaN" values.
        self._dataframe.replace({np.nan: None}, inplace=True)

        # Normalizing the feature "id".
        self._dataframe.id = self._dataframe.id.apply(lambda x: x.replace("arXiv:", "").strip())

        # Normalizing the feature "subject_areas".
        self._dataframe.subject_areas = self._dataframe.subject_areas.apply(lambda x: tuple(eval(x)))

        # Normalizing the features "title" and "abstract".
        self._dataframe.loc[:, ["title", "abstract"]] = self._dataframe.loc[:, ["title", "abstract"]
            ].apply(lambda x: x.apply(lambda y: re.sub("/r/", "",
                re.sub("@PER@CENT@", "%", re.sub(r"[\^_]", "",
                    LatexNodes2Text().latex_to_text(re.sub(r"\s+", " ",
                        re.sub(r"\\?%", "@PER@CENT@", y))).strip())))))

        # Normalizing the feature "authors".
        self._dataframe.authors = [tuple([{"id": str(hash("{} - {}".format(author, "arXiv"))),
                                           "name": author} for author in eval(authors)])
                                   for authors in self._dataframe.authors]

        # Normalizing the feature "date".
        self._dataframe.date = self._dataframe.date.apply(lambda x: re.sub(
            r"\s+", " ", x.split(".")[0]))
        self._dataframe.date = self._dataframe.date.apply(lambda x: x.replace("submitted ", ""))

        # Creating the feature "publication_date" from the feature "date".
        self._dataframe["publication_date"] = self._dataframe.date.apply(
            lambda x: datetime.strptime(x.split(";")[0].strip(), "%d %B, %Y").date())

        # Removing unnecessary columns.
        self._dataframe.drop(axis=1, columns="date", inplace=True)