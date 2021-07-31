##############################################################################################
############ Cleaning and Preprocessing the Scopus publications related to COVID-19 ##########
##############################################################################################

# For collecting the Scopus publications related to COVID-19, we used the "pybliometrics"
# library. It is avaliable on https://pypi.org/project/pybliometrics/.

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

class ProcessScopus(Preprocess):

    # Defining the function "clean_text" to clean and preprocess any text.
    @staticmethod
    def __clean_text(text, has_latex=False):
        if text:
            text = re.sub(r"\u2fff(s|\s)", r"'\1", re.sub(r"\s+", " ", re.sub(r"\ufeff\.?", "",
                re.sub(r"\\\\(\’\s)?", "", str(text))))).replace("\u200b", "").replace(
                "\ue001", "").replace("\ue061", "").replace("\u202f", "").replace(
                "\u2060", "").replace("\u200f", "").replace("\u200e", "").replace(
                "\u202c", "").replace("&#x2013;", "-").replace("&quot", "\"\"").replace(
                "\u200c", "").replace("\\u0019", "").replace("\\s", "s").replace(
                "\u202a", "").replace("\u202d", "-").replace("\u0383", "-").replace(
                "\u20f3", "ó").replace("\u20fa", "ú").replace("\u2fff", "-").strip()
            text = text.replace("TNF-alpha induced", "TNF-α induced").replace(
                "TNF-Alpha induced", "TNF-α induced").replace(
                "TNF- ␣ induced", "TNF-α induced").replace("TNF-αinduced", "TNF-α induced").replace(
                "via NF- \u242c B pathway", "via NF-κB pathway").replace(
                "via NF-kappaB pathway", "via NF-κB pathway").strip()
            if has_latex:
                text = LatexNodes2Text().latex_to_text(re.sub("\\?%", "@PER@CENT@", text)).replace(
                    "@PER@CENT@", "%")
            text = re.sub(r"\s+", " ", re.sub(r"\-{2,}", "-",
                re.sub(r"\s?\xad(\s|\-)?", "-", text))).replace("\\", "").replace(
                "\\%", "%").replace("()", "").replace("[]", "").strip()
            return text
        else:
            return None

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
        self._dataframe.abstract.loc[self._dataframe.abstract.notnull()] = \
            self._dataframe.abstract.loc[self._dataframe.abstract.notnull()].apply(
                lambda x: ProcessScopus.__clean_text(x, True))

        # Normalizing the feature "vehicle_name".
        self._dataframe.vehicle_name.loc[
            self._dataframe.conference_name.notnull() & self._dataframe.vehicle_name.notnull()
        ] = self._dataframe.conference_name.loc[
            self._dataframe.conference_name.notnull() & self._dataframe.vehicle_name.notnull()]
        self._dataframe.vehicle_name.loc[self._dataframe.vehicle_name.notnull()] = \
            self._dataframe.vehicle_name.loc[self._dataframe.vehicle_name.notnull()].apply(
                ProcessScopus.__clean_text)

        # Normalizing the feature "title".
        self._dataframe.title.loc[self._dataframe.title.notnull()] = self._dataframe.title.loc[
            self._dataframe.title.notnull()].apply(ProcessScopus.__clean_text)

        # Removing unnecessary columns.
        columns_drop = ["eid", "pii", "description", "isbn", "conf_location", "conference_name",
            "vehicle_address", "title_edition"]
        self._dataframe.drop(axis=1, columns=columns_drop, inplace=True)

        # Changing the type of some features.
        self._dataframe.loc[:, ["citation_num", "ref_count"]] = self._dataframe.loc[:,
            ["citation_num", "ref_count"]].astype(np.float32)
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

        # Creating the feature "period" from the feature "publication_date".
        if "period" not in self._dataframe:
            self._dataframe["period"] = self._dataframe.publication_date.apply(
                lambda x: "{}-{}".format(x.year, x.month))

        # Normalizing the itens contained in the features "auth_keywords" and "index_terms".
        self._dataframe.auth_keywords.loc[self._dataframe.auth_keywords.notnull()] = \
            self._dataframe.auth_keywords.loc[self._dataframe.auth_keywords.notnull()].apply(
                lambda x: tuple([ProcessScopus.__clean_text(item) for item in x]))
        self._dataframe.index_terms.loc[self._dataframe.index_terms.notnull()] = \
            self._dataframe.index_terms.loc[self._dataframe.index_terms.notnull()].apply(
                lambda x: tuple([ProcessScopus.__clean_text(item) for item in x]))

        # Checking there are invalid values in the features "auth_keywords", "index_terms" and "subject_areas".
        for column in ["auth_keywords", "index_terms", "subject_areas"]:
            count = self._dataframe.loc[self._dataframe[column].notnull(), column][
                    [np.any([item == None or item.lower() == "none" for item in items])
                    for items in self._dataframe.loc[self._dataframe[column].notnull(), column]]].size
            print("{}: {}".format(column, count))

        # Removing the invalid values in the features "auth_keywords", "index_terms" and "subject_areas".
        for column in ["auth_keywords", "index_terms", "subject_areas"]:
            self._dataframe.loc[self._dataframe[column].notnull(), column] = [
                tuple([item for item in items if item])
                for items in self._dataframe.loc[self._dataframe[column].notnull(), column]]
            self._dataframe.loc[self._dataframe[column].notnull(), column] = self._dataframe.loc[
                self._dataframe[column].notnull(), column].apply(lambda x: x if len(x) > 0 else None)

        # Normalizing the content contained in the features "authors", "affiliations" and "author_affil".
        self._dataframe.affiliations.loc[self._dataframe.affiliations.notnull()] = \
            self._dataframe.affiliations.loc[self._dataframe.affiliations.notnull()].apply(
                lambda x: tuple([{"id": item["id"],
                    "affiliation": ProcessScopus.__clean_text(item["affiliation"]),
                    "country": item["country"]} for item in x if item["id"]]))
        self._dataframe.author_affil.loc[self._dataframe.author_affil.notnull()] = \
            self._dataframe.author_affil.loc[self._dataframe.author_affil.notnull()].apply(
                lambda x: tuple([{"id": item["id"],
                    "name": ProcessScopus.__clean_text(item["name"]), "affil_id": item["affil_id"],
                    "affiliation": ProcessScopus.__clean_text(item["affiliation"]),
                    "country": item["country"]} for item in x if item["id"] or item["name"] or \
                        item["affil_id"] or item["affiliation"] or item["country"]]))
        self._dataframe.authors.loc[self._dataframe.authors.notnull()] = \
            self._dataframe.authors.loc[self._dataframe.authors.notnull()].apply(
                lambda x: tuple([{"id": item["id"],
                    "name": ProcessScopus.__clean_text(item["name"])} for item in x if item["id"]]))

        # Removing the invalid values in the features "authors", "affiliations" and "author_affil".
        for column in ["authors", "affiliations", "author_affil"]:
            self._dataframe.loc[self._dataframe[column].notnull(), column] = self._dataframe.loc[
                self._dataframe[column].notnull(), column].apply(lambda x: x if len(x) > 0 else None)

        # Creating the affiliations' and authors' IDs for those that have not a ID.
        self._dataframe.author_affil.loc[self._dataframe.author_affil.notnull()] = \
            self._dataframe.author_affil.loc[self._dataframe.author_affil.notnull()].apply(
                lambda x: tuple([{"id": item["id"] if item["id"] and item["name"] else \
                    str(hash("{} - {}".format(item["name"], "Scopus"))) if item["name"] else None,
                    "name": item["name"],
                    "affil_id": item["affil_id"] if item["affil_id"] and item["affiliation"] else \
                        str(hash("{} - {}".format(item["affiliation"], "Scopus"))) \
                            if item["affiliation"] else None,
                    "affiliation": item["affiliation"], "country": item["country"]}
                    for item in x]))

        # Removing duplicates within the list of affiliations and authors.
        self._dataframe.author_affil.loc[self._dataframe.author_affil.notnull()] = [
            set([(au["id"], au["name"], au["affil_id"],
                au["affiliation"], au["country"]) for au in row])
            for row in self._dataframe.author_affil[self._dataframe.author_affil.notnull()]]
        self._dataframe.author_affil.loc[self._dataframe.author_affil.notnull()] = [tuple([dict(zip(
                ["id", "name", "affil_id", "affiliation", "country"], au)) for au in row])
            for row in self._dataframe.author_affil[self._dataframe.author_affil.notnull()]]

        # Removing the duplicated records by feature "id".
        self._dataframe = self._dataframe.sort_values(by=["id", "period"]).drop_duplicates(
            "id", keep="first")

        # Removing the duplicated records by features "title" and "doi".
        self._dataframe = pd.concat([
            self._dataframe[self._dataframe.title.isnull() | self._dataframe.doi.isnull()],
            self._dataframe[self._dataframe.title.notnull() & self._dataframe.doi.notnull()].sort_values(
                by=["title", "citation_num", "publication_date"]).drop_duplicates(
                    ["title", "doi"], "last")], ignore_index=True)

        # Normalizing the feature "references".
        self._dataframe.references.loc[self._dataframe.references.notnull()] = \
            self._dataframe.references.loc[self._dataframe.references.notnull()].apply(
                lambda x: tuple([
                    {"id": ref["id"], "title": ProcessScopus.__clean_text(ref["title"], True),
                    "doi": ProcessScopus.__clean_text(ref["doi"]),
                    "authors": ProcessScopus.__clean_text(ref["authors"], True)} for ref in x]))