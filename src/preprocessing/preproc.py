import pathlib
import sys
from typing import List
import pandas as pd
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import pandas as pd
import contractions
import spacy
import argparse
from langdetect import detect
from tqdm import tqdm
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from acronyms import acronyms_list
from preproc_utils import det
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import logging
from preproc_utils import check_nltk_packages

#!python3 -m spacy download en_core_web_lg
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
        
        self._stopwords = self._loadSTW(stw_files)
        self._loadACR()
            
        return
            
    def _loadSTW(self, stw_files: List[pathlib.Path]):
        """
        Loads all stopwords as list from all files provided in the argument

        Parameters
        ----------
        stw_files: list of str
            List of paths to stopwords files
        """
        
        stw_list = []
        for stw_file in stw_files:
            stw_df = pd.read_csv(stw_file, names=['stopwords'], header=None, skiprows=3)
            stw_list.extend(list(stw_df['stopwords']))
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
        Carries out NLP pipeline. In particular, the following steps:
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
        
        valid_POS = set(['VERB', 'NOUN', 'ADJ', 'PROPN'])

        doc = self.nlp(rawtext)
        lemmatized = ' '.join([token.lemma_ for token in doc
                            if token.is_alpha
                            and token.pos_ in valid_POS
                            and not token.is_stop
                            and token.lemma_ not in self._stw_list])
        
        lemmatized2 = ''
        for lemma in lemmatized.split(' '):
            # Change acronyms by their meaning
            text = self._replace(lemma,self._acr_list) 
            # Expand contractions 
            text2 = contractions.fix(text) 
            lemmatized2 = lemmatized2 + ' ' + text2
        # To build the dictionary afterwards
        tokenized2 = word_tokenize(lemmatized2) 
        # Convert to lowercase
        final_tokenized = [token.lower() for token in tokenized2] 
        return ' '.join(el for el in final_tokenized)


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
        
        # Create nlp pipeline
        self.nlp = spacy.load('en_core_web_lg')

        # Disable unnecessary components
        self.nlp.disable_pipe('parser')
        self.nlp.disable_pipe('ner')
        
        # Lemmatize text
        #corpus_df['lemmas'] = corpus_df["raw_text"].apply(self.do_pipeline, meta=('lemmas', 'object'))
        corpus_df['lemmas'] = corpus_df["raw_text"].apply(self.do_pipeline)
        
        # Create corpus from tokenized lemmas
        #dbag = corpus_df['lemmas'].to_dask_array().compute()
        #dbag_list = list(dbag)
        #print(dbag)
        #print(len(dbag_list))
        
        #corpus = corpus_df['lemmas'].compute().values
        corpus = corpus_df['lemmas'].values
        print(len(corpus))
        print(len(corpus_df))

        corpus2 = [el.split() for el in corpus]
        
        # Create Phrase model for n-grams detection
        phrase_model = Phrases(corpus2, min_count=2, threshold=20)
        
        # Carry out n-grams substitution
        corpus2 = [el for el in phrase_model[corpus2]] 

        corpus3 = [" ".join(el) for el in corpus2]

        print(len(corpus3))
        #print(len(corpus_df.compute()))

        def get_ngram(row):
            return corpus3.pop(0)

        # Save n-grams in new column in the dataFrame
        #corpus_df["lemmas_with_grams"] =  corpus_df.apply(get_ngram, meta=('lemmas_with_grams', 'object'), axis=1)
        corpus_df["lemmas_with_grams"] =  corpus_df.apply(get_ngram, axis=1)
        
        return corpus_df
                
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Scripts for Embeddings Service")
    parser.add_argument("--source_path", type=str, default=None,
                        required=True, metavar=("path_to_parquet"),
                        help="Path to file with source file/s")
    parser.add_argument("--source_type", type=str, default=None,
                        required=True, metavar=("source_type"),
                        help="Type of file in which the source is contained")
    parser.add_argument("--source", type=str, default=None,
                        required=True, metavar=("source_type"),
                        help="Name of the dataset to be preprocessed (e.g., cordis)")
    parser.add_argument("--destination_path", type=str, default=None,
                        required=True, metavar=("destination_path"),
                        help="Path to save the new preprocessed files")
    parser.add_argument("--nw", type=int, default=0,
                    required=False, help="Number of workers to use with Dask")
    
    # Create logger object
    logging.basicConfig(level='INFO')
    logger = logging.getLogger('nlpPipeline')
    
    logger.info(
                f'-- -- Installing necessary packages...')
    check_nltk_packages()
    
    args = parser.parse_args()
    
    source_path = pathlib.Path(args.source_path)
    destination_path = pathlib.Path(args.destination_path)
    
    # Create corpus_df
    if args.source_type == "xlsx":
        if args.source == "cordis":
            logger.info(
                f'-- -- Reading from Cordis...')
            id_fld = "projectID"
            raw_text_fld = "objective"#summary
            title_fld = "title"
            
        df = pd.read_excel(source_path)
        #corpus_df = dd.from_pandas(df, npartitions=3)
        
        #corpus_df = df.sample(frac=0.001, replace=True, random_state=1)
        corpus_df = df[[id_fld, raw_text_fld, title_fld]]
        
        # Detect abstracts' language and filter out those non-English ones
        #corpus_df['langue'] = corpus_df[raw_text_fld].apply(det, meta=('langue', 'object'))
        corpus_df['langue'] = corpus_df[raw_text_fld].apply(det)
        
        corpus_df = corpus_df[corpus_df['langue'] == 'en']
        
        # Filter out abstracts with no text   
        corpus_df = corpus_df[corpus_df[raw_text_fld] != ""]
        
        # Concatenate title + abstract/summary
        #corpus_df["raw_text"] = corpus_df[[title_fld, raw_text_fld]].apply(" ".join, axis=1, meta=('raw_text', 'object'))
        corpus_df["raw_text"] = corpus_df[[title_fld, raw_text_fld]].apply(" ".join, axis=1)
        
    elif args.source_type == "parquet":
        if args.source == "scholar":
            logger.info(
                f'-- -- Reading from Scholar...')
            id_fld = ""
            raw_text_fld = "paperAbstract"
            title_fld = "title"
            
        res = []
        for entry in source_path.iterdir():
            # check if it is a file
            if entry.as_posix().endswith("parquet"):
                res.append(entry)
        
        logger.info(
                f'-- -- Reading of parquet files starts...')
        for idx, f in enumerate(tqdm(res)):
            df = dd.read_parquet(f)
            
            # Filter out abstracts with no text   
            df = df[df[raw_text_fld] != ""]
            
            # Detect abstracts' language and filter out those non-English ones
            #df['langue'] = df[raw_text_fld].apply(det, meta=('langue', 'object'))
            df['langue'] = df[raw_text_fld].apply(det)
            df = df[df['langue'] == 'en']
            
            # Filter out abstracts with no text   
            df = df[df[raw_text_fld] != ""]
            
            # Concatenate title + abstract/summary
            #df["raw_text"] = df[[title_fld, raw_text_fld]].apply(" ".join, axis=1, meta=('raw_text', 'object'))
            df["raw_text"] = df[[title_fld, raw_text_fld]].apply(" ".join, axis=1)
            
            # Concatenate dataframes
            if idx == 0:
                corpus_df = df
            else:
                corpus_df = dd.concat([corpus_df, df])
    
    # Get stopword lists
    stw_lsts = []
    for entry in pathlib.Path("/export/usuarios_ml4ds/lbartolome/hierarchical-topic-models/data/stw_lists").iterdir():
        #/export/usuarios_ml4ds/lbartolome/hierarchical-topic-models/data/stw_lists
        #/workspaces/hierarchical-topic-models/data/stw_lists
        # check if it is a file
        if entry.as_posix().endswith("txt"):
            stw_lsts.append(entry)
    
    logger.info(
                f'-- -- NLP preprocessing starts...')
    nlpPipeline = nlpPipeline(stw_files=stw_lsts,
                              logger=logger)
    
    corpus_df = nlpPipeline.preproc(corpus_df)
    
    print(corpus_df)
            
    # Save new df in parquet file
    outFile = destination_path.joinpath("preproc_" + args.source + ".parquet")
    if outFile.is_file():
        outFile.unlink()
    corpus_df.to_parquet(outFile.as_posix())
    
    """
    with ProgressBar():
        if args.nw > 0:
            corpus_df.to_parquet(outFile, write_index=False, schema="infer", compute_kwargs={
                'scheduler': 'processes', 'num_workers': args.nw})
        else:
            # Use Dask default number of workers (i.e., number of cores)
            corpus_df.to_parquet(outFile, write_index=False, schema="infer", compute_kwargs={
                'scheduler': 'processes'})
    """


    