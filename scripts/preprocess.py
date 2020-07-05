##############################################################################################
################ Cleaning and Preprocessing the publications related to COVID-19 #############
##############################################################################################

########################################################################
# Importing the required libraries.
import csv, pandas as pd, sys
from abc import ABC, abstractmethod
########################################################################

class Preprocess(ABC):
    # Datasets enumeration.
    ARXIV = "arxiv"
    BIORXIV = "biorxiv"
    PUBMED = "pubmed"
    SCOPUS = "scopus"
    FINAL = "final"

    # Method that perform the specific process of cleaning and preprocessing.
    @abstractmethod
    def _preprocess(self):
        pass

    # Method that perform the generic process of cleaning and preprocessing.
    def process_raw_data(self, raw_data, preprocessed_data, dtypes_raw_data = None):
        # Creating a dataframe from the raw data.
        self._dataframe = pd.read_csv(raw_data, header=0, dtype=dtypes_raw_data)

        # Cleaning and preprocessing the raw data.
        self._preprocess()

        # Exporting the data to CSV file.
        self._dataframe.to_csv(preprocessed_data, index=False, quoting=csv.QUOTE_ALL)

    # Method that generate a specific process of cleaning and preprocessing. 
    @staticmethod
    def factory_process(dataset):
        if dataset == Preprocess.ARXIV:
            from preprocess_arxiv import ProcessArxiv
            return ProcessArxiv()
        elif dataset == Preprocess.BIORXIV:
            from preprocess_biorxiv import ProcessBiorxiv
            return ProcessBiorxiv()
        elif dataset == Preprocess.PUBMED:
            from preprocess_pubmed import ProcessPubmed
            return ProcessPubmed()
        elif dataset == Preprocess.SCOPUS:
            from preprocess_scopus import ProcessScopus
            return ProcessScopus()
        elif dataset == Preprocess.FINAL:
            from preprocess_final import ProcessFinal
            return ProcessFinal()
        else:
            raise FileNotFoundError("This dataset does not exist.")

# Executing the cleaning and preprocessing process of raw data.
if __name__ == "__main__":
    if len(sys.argv) == 1:
        Preprocess.factory_process(Preprocess.ARXIV).process_raw_data(
            "data/raw/arxiv_raw.csv", "data/prepared/arxiv_covid_19.csv")
        Preprocess.factory_process(Preprocess.BIORXIV).process_raw_data(
            "data/raw/biorxiv_raw.csv", "data/prepared/biorxiv_covid_19.csv")
        Preprocess.factory_process(Preprocess.PUBMED).process_raw_data(
            "data/raw/pubmed_raw.csv", "data/prepared/pubmed_covid_19.csv",
            {"pubmed_id": "str"})
        Preprocess.factory_process(Preprocess.SCOPUS).process_raw_data(
            "data/raw/scopus_raw.csv", "data/prepared/scopus_covid_19.csv",
            {"id": "str", "eid": "str", "pii": "str", "pubmed_id": "str"})
    elif sys.argv[1] == "final":
        Preprocess.factory_process(Preprocess.FINAL).process_raw_data(
            "data/raw/final_raw.csv", "data/prepared/final_covid_19.csv",
            {"id": "str", "pubmed_id": "str"})
