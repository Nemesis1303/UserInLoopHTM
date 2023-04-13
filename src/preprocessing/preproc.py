import json
import pathlib
from typing import List
import pandas as pd
import re
import pandas as pd
import contractions
import spacy
import argparse
from tqdm import tqdm
from gensim.models.phrases import Phrases
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from acronyms import acronyms_list
from preproc_utils import det
import logging
from preproc_utils import check_nltk_packages


# TO BE EXECUTED ONLY ONCE
#!python3 -m spacy download spacy_model (where spacy_model is one of en_core_web_sm | en_core_web_md | en_core_web_lg )
#!python3 -m spacy download xx_sent_ud_sm
#!pip install --upgrade spacy_langdetect

class nlpPipeline():
    """
    Class to carry out text preprocessing tasks that are needed by topic modeling
    - Basic stopword removal
    - Acronyms substitution
    - NLP preprocessing
    - Ngrams detection
    """
    
    def __init__(self, 
                 stw_files: List[pathlib.Path], 
                 logger=None):
        """
        Initilization Method
        Stopwords files will be loaded during initialization

        Parameters
        ----------
        stw_files: list of str
            List of paths to stopwords files
        logger: Logger object
            To log object activity
        """
       
        if logger:
            self._logger = logger
        else:
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('nlpPipeline')
        
        self._loadSTW(stw_files)
        self._loadACR()
        self._nlp = spacy.load('en_core_web_sm', exclude=['parser', 'ner'])
            
        return
            
    def _loadSTW(self, stw_files: List[pathlib.Path]):
        """
        Loads stopwords as list from files provided in the argument

        Parameters
        ----------
        stw_files: list of str
            List of paths to stopwords files
        """
        
        stw_list = [pd.read_csv(stw_file, names=['stopwords'], header=None, skiprows=3) for stw_file in stw_files]
        stw_list = [stopword for stw_df in stw_list for stopword in stw_df['stopwords']]
        self._stw_list = list(dict.fromkeys(stw_list)) # remove duplicates
        self._logger.info(f"Stopwords list created with {len(stw_list)} items.")

        return 
    
    def _loadACR(self):
        """
        Loads list of acronyms
        """
        
        self._acr_list = acronyms_list
        
        return
    
    def _replace(self, text, patterns) -> str:
        """
        Auxiliary function to replace patterns in strings.
        
        Parameters
        ----------
        text: str
            Text in which the patterns are going to be replaced
        patterns: List of tuples
            Replacement to be carried out
            
        Returns
        -------
        text: str
            Replaced text
        """
         
        for(raw,rep) in patterns:
            regex = re.compile(raw)
            text = regex.sub(rep,text)
        return text

    def do_pipeline(self, rawtext) -> str:
        """
        Implements the preprocessing pipeline, by carrying out:
        - Lemmatization according to POS
        - Removal of non-alphanumerical tokens
        - Removal of basic English stopwords and additional ones provided       
          within stw_files
        - Acronyms replacement
        - Expansion of English contractions
        - Word tokenization
        - Lowercase conversion
        
        Parameters
        ----------
        rawtext: str
            Text to preprocess
            
        Returns
        -------
        final_tokenized: List[str]
            List of tokens (strings) with the preprocessed text
        """
        
        # Change acronyms by their meaning
        text = self._replace(rawtext, self._acr_list)
      
        # Expand contractions
        try:
            text = contractions.fix(text) 
        except:
            text = text # this is only for SS
        
        valid_POS = set(['VERB', 'NOUN', 'ADJ', 'PROPN'])

        doc = self._nlp(text)
        lemmatized = [token.lemma_ for token in doc
                            if token.is_alpha
                            and token.pos_ in valid_POS
                            and not token.is_stop
                            and token.lemma_ not in self._stw_list]
        
        # Convert to lowercase
        final_tokenized = [token.lower() for token in lemmatized] 
        
        return final_tokenized

    def preproc(self, corpus_df: dd.DataFrame) -> dd.DataFrame:
        """
        Invokes NLP pipeline and carries out, in addition, n-gram detection.
        
        Parameters
        ----------
        corpus_df: dd.DataFrame
            Dataframe representation of the corpus to be preprocessed. 
            It needs to contain (at least) the following columns:
            - raw_text
            
        Returns
        -------
        corpus_df: dd.DataFrame
            Preprocessed DataFrame
            It needs to contain (at least) the following columns:
            - raw_text
            - lemmas
            - lemmas_with_grams
        """
        
        # Lemmatize text
        self._logger.info("-- INFO: Lemmatizing text")        
        lemmas = corpus_df["raw_text"].apply(self.do_pipeline, 
                                            meta=('lemmas', 'object'))
        # this does the same but with batch preprocessing
        #def apply_pipeline_to_partition(partition):
        #    return partition['raw_text'].apply(self.do_pipeline)
        #lemmas = corpus_df.map_partitions(apply_pipeline_to_partition, meta=('lemmas', 'object'))
        
        # Create corpus from tokenized lemmas
        self._logger.info(
            "-- INFO: Creating corpus from lemmas for n-grams detection")
        with ProgressBar():
            lemmas = lemmas.compute(scheduler='processes')   
        corpus = lemmas.values.tolist()

        # Create Phrase model for n-grams detection
        self._logger.info("-- INFO: Creating Phrase model")
        phrase_model = Phrases(corpus, min_count=2, threshold=20)

        # Carry out n-grams substitution
        self._logger.info("-- INFO: Carrying out n-grams substitution")
        corpus = (phrase_model[doc] for doc in corpus)
        corpus = list((" ".join(doc) for doc in corpus))

        # Save n-grams in new column in the dataFrame
        self._logger.info(
            "-- INFO: Saving n-grams in new column in the dataFrame")
        def get_ngram(row):
            return corpus.pop(0)
        corpus_df["lemmas_with_grams"] =  corpus_df.apply(get_ngram, meta=('lemmas_with_grams', 'object'), axis=1)
        #corpus_df["lemmas_with_grams"] =  corpus_df.apply(lambda row: corpus[row.name], meta=('lemmas_with_grams', 'object'), axis=1)
        #import pdb; pdb.set_trace()
        #corpus_df["lemmas_with_grams"] =  corpus_df.apply(corpus.pop(0), meta=('lemmas_with_grams', 'object'), axis=1)
        
        return corpus_df
                
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Scripts for Embeddings Service")
    parser.add_argument("--source_path", type=str, default="/Users/lbartolome/Documents/GitHub/hierarchical-topic-models/data/CORDIS.parquet",
                        required=False, metavar=("path_to_parquet"),
                        help="Path to file with source file/s")
    parser.add_argument("--source_type", type=str, default='parquet',
                        required=False, metavar=("source_type"),
                        help="Type of file in which the source is contained")
    parser.add_argument("--source", type=str, default='cordis',
                        required=False, metavar=("source_type"),
                        help="Name of the dataset to be preprocessed (e.g., cordis)")
    parser.add_argument("--destination_path", type=str, default='/Users/lbartolome/Documents/GitHub/hierarchical-topic-models/data',
                        required=False, metavar=("destination_path"),
                        help="Path to save the new preprocessed files")
    parser.add_argument("--nw", type=int, default=0,
                    required=False, help="Number of workers to use with Dask")
    
    # Create logger object
    logging.basicConfig(level='INFO')
    logger = logging.getLogger('nlpPipeline')
    
    logger.info(
                f'-- -- Installing necessary packages...')
    #check_nltk_packages()
    
    args = parser.parse_args()
    source_path = pathlib.Path(args.source_path)
    destination_path = pathlib.Path(args.destination_path)
    
    # Get stopword lists
    stw_lsts = []
    for entry in pathlib.Path("stw_lists").iterdir():
        # check if it is a file
        if entry.as_posix().endswith("txt"):
            stw_lsts.append(entry)
    
    nlpPipeline = nlpPipeline(stw_files=stw_lsts, logger=logger)
    
    with open('config.json') as f:
        field_mappings = json.load(f)

    if args.source in field_mappings:
        mapping = field_mappings[args.source]
        logger.info(f"Reading from {args.source}...")
        id_fld = mapping["id"]
        raw_text_fld = mapping["raw_text"]
        title_fld = mapping["title"]
    else:
        logger.error(f"Unknown source: {args.source}")
            
    readers = {
        "xlsx": lambda path: dd.from_pandas(pd.read_excel(path), npartitions=10).fillna(""),
        "parquet": lambda path: dd.read_parquet(path).fillna("")
    }

    if args.source_type in readers:
        reader = readers[args.source_type]
        df = reader(source_path)
    else:
        logger.error(f"Unsupported source type: {args.source_type}")
        
    corpus_df = df.sample(frac=0.00001, replace=True, random_state=1)
    corpus_df = df[[id_fld, raw_text_fld, title_fld]]
        
    # Detect abstracts' language and filter out those non-English ones
    corpus_df = \
        corpus_df[corpus_df[raw_text_fld].apply(
            det,
            meta=('langue', 'object')) == 'en']

    # Concatenate title + abstract/summary
    corpus_df["raw_text"] = \
        corpus_df[[title_fld, raw_text_fld]].apply(
            " ".join, axis=1, meta=('raw_text', 'object'))
    # Filter out rows with no raw_text
    corpus_df = corpus_df.dropna(subset=["raw_text"], how="any")
    
    logger.info(f'-- -- NLP preprocessing starts...')
    import time
    start_time = time.time()
    corpus_df = nlpPipeline.preproc(corpus_df)
    print("--- %s seconds ---" % (time.time() - start_time))
                    
    # Save new df in parquet file
    outFile = destination_path.joinpath("preproc_" + args.source + ".parquet")
    if outFile.is_file():
        outFile.unlink()
    
    with ProgressBar():
        if args.nw > 0:
            corpus_df.to_parquet(outFile, write_index=False, schema="infer", compute_kwargs={
                'scheduler': 'processes', 'num_workers': args.nw})
        else:
            # Use Dask default number of workers (i.e., number of cores)
            corpus_df.to_parquet(outFile, write_index=False, schema="infer", compute_kwargs={
                'scheduler': 'processes'})
            