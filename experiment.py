import sys
import os
import subprocess
import argparse
import skopt
import numpy as np
import json


subword_nmt = os.path.join("/home/ualelm", "subword_fork")
if subword_nmt not in sys.path:
    sys.path.append(subword_nmt)
import apply_bpe
from get_vocab import write_vocab

import models #Simply contains lists of Marian parameters

def working_directory():
    return os.path.abspath(".")

def absolute_file(path):
    absolute = os.path.abspath(path)
    if not os.path.isfile(absolute):
        raise ValueError("%s is not a file." % absolute)
    return absolute

def absolute_dir(path):
    absolute = os.path.abspath(path)
    if not os.path.isdir(absolute):
        raise ValueError("%s is not a directory." % absolute)
    return absolute

max_sequences_DEFAULT=(1000, 1000)
vocab_threshold_DEFAULT=0

def create_parser():
    parser = argparse.ArgumentParser(description="Find optimal number of BPE codes for a translation task")

    parser.add_argument("--train", required=True, nargs=2, metavar=("<src_path>", "<ref_path>"), type=absolute_file, help="Source and destination training corpora in plain text")

    #CORPORA
    parser.add_argument("--dev", nargs="+", metavar=("<dev_corpus>"), type=absolute_file, help="Development corpora in plain text (only need source if using BLEU, both source and dest otherwise")
    parser.add_argument("--dev-sgml", nargs=2, metavar=("<src_plain>", "<ref_plain>"), type=absolute_file, help="Development corpora in SGML (required for BLEU")

    #BPE Merges
    parser.add_argument("--codes", required=True, nargs="+", metavar="<codes_path>", type=os.path.abspath, help="BPE codes file(s) (pass in only one if using joint codes)")
    parser.add_argument("--max-sequences", "-s", default=max_sequences_DEFAULT, nargs="+", metavar="n", type=int, help="Maximum number of codes to use from BPE code file(s) (default %(default)s); include two numbers for different limits on sequences in the source and destination languages")
    parser.add_argument("--vocabulary-threshold", "--thresh", default=vocab_threshold_DEFAULT, metavar="n", type=int, help="--vocbaulary threshold parameter for apply_bpe.py")

    #OUTPUT PATHS
    parser.add_argument("--translation-dir", "--trans", default=working_directory(), metavar="<dir>", type=absolute_dir, help="Directory to write translated text (default current directory")
    parser.add_argument("--vocab-dir", "--vocab", default=working_directory(), metavar="<dir>", type=absolute_dir, help="Directory to write vocabulary files (default current directory")
    parser.add_argument("--train-log-prefix", metavar="<dir>", type=str, help="Prefix for Marian training logs")
    parser.add_argument("--model-prefix", metavar="<dir>", type=str, help="Prefix for Marian models")

    #SCORING
    parser.add_argument("--metric", "-m", default="bleu", nargs=1, help="Validation metric to use (\"bleu\"/\"ce-mean-words\")")
    parser.add_argument("--results", "-r", default="output.json", metavar="<json_path>", type=os.path.abspath, help="Path to write experimental results in JSON format")

    #MISCELLANEOUS
    parser.add_argument("--dest-lang", "-l", default="en", metavar="xx", type=str, help="ISO 2-char language code for target language")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show additional messages")

    return parser

def convert_value(value):
   """
   Hack used to make the most important values in an OptimizeResult object JSON-writeable"
   """
   if type(value) == np.int32 or type(value) == np.int64: return int(value)
   if type(value) == np.float32 or type(value) == np.float64: return float(value)
   if type(value) == np.ndarray: return value.tolist()
   return str(value)

class Experiment:

    @staticmethod
    def __log_extension():
        return ".log"

    @staticmethod
    def __model_extension():
        return ".npz"

    @staticmethod
    def __vocab_extension():
        return ".vocab"

    @staticmethod
    def __process_prefix(user_prefix, default_basename, extension):
        if not(user_prefix):
           return default_basename + extension
        elif not user_prefix.endswith(extension):
           return user_prefix + extension
        else:
           return user_prefix

    @staticmethod
    def __detailed_path(path, merges, extension):
        index = path.rfind(extension)
        return os.path.abspath( path[:index] + merges + extension )

    @staticmethod
    def __merges_string(num_merges):
        if type(num_merges) != list: return ".s" + str(num_merges)
        if len(num_merges) > 1:      return ".s" + str(num_merges[0]) + "-" + str(num_merges[1])
        return ".s" + str(num_merges[0]) + "-" + str(num_merges[0])

    @staticmethod
    def parse_nist(report):
        scoreLabel = "NIST score = "
        maxRawScore = 100

        if type(report) != str:
            lines = report.readlines()
        else:
            lines = report.split("\n")
         
        for line in lines:
            if line.startswith(scoreLabel):
                startIndex = line.index(scoreLabel)
                scoreIndex = startIndex + len(scoreLabel)
                scoreString = line[scoreIndex : ].split(" ")[0]
                print("Extracting the NIST score")
                return maxRawScore - float(scoreString)

        return float("inf")

    @staticmethod
    def parse_marian(report):
        if ("nan" in report) or ("inf" in report):
            return float("inf")
        else:
            return float(report)   #For a summary score, Marian returns a negative mean/sum of the log probabilities
            #return -float(report) #Marian returns log probabilities, not negative log probabilities

    def __init__(self, codes, train, dev, dev_sgml = None,
                       max_sequences = max_sequences_DEFAULT, vocab_threshold = vocab_threshold_DEFAULT, 
                       metric = "bleu", results = os.path.abspath("output.json"),
                       model_prefix = None, train_log_prefix = None, vocab_dir = working_directory(), translation_dir = working_directory(),
                       dest_lang = "en", verbose = True):
        """
        dev - Tuple of either just the source development corpus, or both the source and development corpora, in plain text
        codes - Tuple of paths to one or more BPE codes files
        dest_lang specifies the target language when using Moses's wrap_xml.perl
        """

        self.verbose = verbose
        self.train_source = absolute_file(train[0])
        self.train_dest = absolute_file(train[1])

        self.metric = metric
        self.dev_source = absolute_file(dev[0])
        self.dev_dest   = None
        if self.metric == "bleu":
            if not dev_sgml: raise ValueError("Must provide SGML development corpora to use BLEU metric")
            self.sgml_dev_source = dev_sgml[0]
            self.sgml_dev_dest = dev_sgml[1]
        elif len(dev) < 2: raise ValueError("Must provide source and destination development corpora in plain text if not using BLEU metric")
        else:
            self.dev_dest = absolute_file(dev[1])


        self.results = os.path.abspath(results)
        if self.verbose: print("Will write results to %s" % str(self.results))

        if len(codes) > 1:
            self.joint_codes = False
            self.source_codes = absolute_file(codes[0])
            self.dest_codes = absolute_file(codes[1])
        else:
            self.joint_codes = True
            self.source_codes = absolute_file(codes[0])
            self.dest_codes = self.source_codes
            if self.verbose: print("Using joint codes for source and target text")

        self.max_sequences = max_sequences
        self.vocab_threshold = vocab_threshold if vocab_threshold > vocab_threshold_DEFAULT else vocab_threshold_DEFAULT

        self.model_prefix = Experiment.__process_prefix(model_prefix, "model", Experiment.__model_extension())
        if self.verbose: print("Set model prefix as %s" % str(self.model_prefix))

        self.train_log_prefix = Experiment.__process_prefix(train_log_prefix, "train", Experiment.__log_extension())
        if self.verbose: print("Set training log prefix as %s" % str(self.train_log_prefix))

        self.vocab_dir = absolute_dir(vocab_dir)
        if self.verbose: print("Set vocab directory as %s" % self.vocab_dir)

        self.translation_dir = absolute_dir(translation_dir)
        if self.verbose: print("Set translation directory as %s" % self.translation_dir)

        self.seed = 1 #FIXME: Make this customizable by the user
        self.dest_lang = dest_lang
        self.score_table = []


    def __already_used(self, num_merges):
       i = 0
       for record in self.score_table:
           if num_merges == record[0]: return i
           i += 1
       return -1

    def score_nist(self, model, source_vocab, dest_vocab, bpe_dev_source):
        sgml_translation = os.path.abspath( os.path.join(self.translation_dir, self.dest_lang + Experiment.__merges_string(num_merges) + ".dev.sgm") )
        translate_command = ["marian-decoder",
                             "--models", str(model),
                             "--input", str(bpe_dev_source),
                             "--vocabs", str(source_vocab), str(dest_vocab),
                             "--devices", str(0)
                             #,"--quiet"
                            ]
        translating =  subprocess.Popen( translate_command,                   universal_newlines=True,                            stdout=subprocess.PIPE)
        desegmenting = subprocess.Popen(["sed", "-r", "s/(@@ )|(@@ ?$)//g"],  universal_newlines=True, stdin=translating.stdout,  stdout=subprocess.PIPE)
        detruecasing = subprocess.Popen(["detruecase.perl"],                  universal_newlines=True, stdin=desegmenting.stdout, stdout=subprocess.PIPE)
        detokenizing = subprocess.Popen(["detokenizer.perl"],                 universal_newlines=True, stdin=detruecasing.stdout, stdout=subprocess.PIPE)
        with open(sgml_translation, "w") as out:
            wrapping = subprocess.Popen(["wrap-xml.perl", self.dest_lang, str(self.sgml_dev_source), "TheUniversityOfAlabama"],
                                         universal_newlines=True, stdin=detokenizing.stdout, stdout=out)
            status = wrapping.wait() 
            if status:
                raise RuntimeError("Translating " + str(bpe_dev_source) + " failed with exit code " + str(status)) 

        if self.verbose: print("Wrote translation %s" % (sgml_translation))

        score_command = ["mteval-v14.pl", "-s", str(self.sgml_dev_source), "-t", str(sgml_translation), "-r", str(self.sgml_dev_dest)]
        scoring = subprocess.Popen(score_command, universal_newlines=True, stdout=subprocess.PIPE)
        output = scoring.communicate()[0]
        status = scoring.wait()
        if status:
            raise RuntimeError("Scoring translation of " + str(bpe_dev_source) + " failed with exit code " + str(status)) 
        return Experiment.parse_nist(output)

    @staticmethod
    def score_marian(model_path, source_vocab, dest_vocab, bpe_dev_source, bpe_dev_dest):
        command = [
          "marian-scorer",
          "--model", str(model_path),
          "--vocab", source_vocab, dest_vocab,
          "--train-sets", bpe_dev_source, bpe_dev_dest,
          "--summary", "ce-mean-words"
         ]
        scoring = subprocess.Popen(command, stdout=subprocess.PIPE, universal_newlines=True)
        (results_text, err) = scoring.communicate()
        status = scoring.wait()
        if status:
           raise RuntimeError("Scoring translation of " + str(bpe_dev_source) + " failed with exit code " + str(status))
        return Experiment.parse_marian(results_text)

    @staticmethod
    def __segment_corpus(raw, processed, bp_encoder):
        with open(raw, "r") as r, open(processed, "w") as p:
            for line in r.readlines():
                p.write(bp_encoder.segment(line).strip())
                p.write("\n")

    @staticmethod
    def __generate_vocab(bpe_corpus, vocab):
           with open(bpe_corpus, "r", encoding="utf-8") as r, open(vocab, "w", encoding="utf-8") as w:
               vocab_proc = subprocess.Popen(["marian-vocab"], universal_newlines=True, stdin=r, stdout=w)
               status = vocab_proc.wait()
           if status: raise RuntimeError("Generating vocabulary file " + str(vocab) + " merges failed with exit code " + str(status))


    def __segment_corpora(self, num_merges, codes, train, dev=None, side="source"):
       bpe_train =  os.path.join( str(train) + Experiment.__merges_string(num_merges) )
       bpe_dev = os.path.join( str(dev) + Experiment.__merges_string(num_merges) ) if dev else None

       with open(codes, "r", encoding="utf-8") as cds:
           encoder = apply_bpe.BPE(cds, num_merges)
           Experiment.__segment_corpus(train, bpe_train, encoder)
           if self.vocab_threshold:
               filter_file = os.path.join(self.vocab_dir, side + Experiment.__merges_string(num_merges) + ".filter")
               write_vocab(bpe_train, filter_file)
               if self.verbose: print("Wrote vocabulary filter %s" % filter_file)

               with open(filter_file, "r", encoding="utf-8") as ff:
                   filter_dict = apply_bpe.read_vocabulary(ff, self.vocab_threshold)
               encoder = apply_bpe.BPE(cds, num_merges, vocab=filter_dict)
               Experiment.__segment_corpus(train, bpe_train, encoder)
           if self.verbose: print("Wrote segmented training corpus %s" % bpe_train)
           if dev:
               Experiment.__segment_corpus(dev, bpe_dev, encoder)
               if self.verbose: print("Wrote segmented development corpus %s" % bpe_dev)

       return (bpe_train, bpe_dev)

    def __train_brief(self, bpe_train_source, bpe_train_dest, source_vocab, dest_vocab, num_merges):
        """
        Returns the path of the model trained.
        """
        merges_string = Experiment.__merges_string(num_merges)
        model_path = Experiment.__detailed_path(self.model_prefix, merges_string, Experiment.__model_extension())
        log_path = Experiment.__detailed_path(self.train_log_prefix, merges_string, Experiment.__log_extension())

        model_params = models.best_deep
        local_params = [
                       #General options
                       "--log", str(log_path), "--quiet",
                       "--seed", str(self.seed),
                       #Model options
                       "--model", str(model_path),
                       #Training options
                       "--train-sets", str(bpe_train_source), str(bpe_train_dest),
                       "--vocabs", str(source_vocab), str(dest_vocab)
                       ]
        command = ["marian"] + model_params + local_params
        training = subprocess.Popen(command, universal_newlines=True)
        status = training.wait()
        if status: raise RuntimeError("Training process with " + str(num_merges) + " merges failed with exit code " + str(status))

        return model_path

    def __vocab_rating(self, num_merges):
         index = self.__already_used(num_merges)
         if index != -1:
             score = self.score_table[index][-1]
             if self.verbose: print("Returning cached score of %f" % score)
             return score

         source_merges = num_merges[0]
         dest_merges = num_merges[1] if len(num_merges) > 1 else source_merges

         (bpe_train_source, bpe_dev_source) = self.__segment_corpora(source_merges, self.source_codes, self.train_source, dev=self.dev_source, side="source")
         (bpe_train_dest, bpe_dev_dest)     =  self.__segment_corpora(dest_merges, self.dest_codes,   self.train_dest,   dev=self.dev_dest, side="dest")

         source_vocab = os.path.join(self.vocab_dir, "source" + Experiment.__merges_string(source_merges) + Experiment.__vocab_extension())
         dest_vocab = os.path.join(self.vocab_dir, "dest" + Experiment.__merges_string(dest_merges) + Experiment.__vocab_extension())

         Experiment.__generate_vocab(bpe_train_source, source_vocab)
         if self.verbose: print("Wrote vocabulary file %s" % source_vocab)
         Experiment.__generate_vocab(bpe_train_dest, dest_vocab)
         if self.verbose: print("Wrote vocabulary file %s" % dest_vocab)

         model_path = self.__train_brief(bpe_train_source, bpe_train_dest, source_vocab, dest_vocab, num_merges)

         if self.metric == "bleu": score =   self.score_nist(model_path, source_vocab, dest_vocab, bpe_dev_source)
         else:                     score = Experiment.score_marian(model_path, source_vocab, dest_vocab, bpe_dev_source, bpe_dev_dest)

         self.score_table += [ (num_merges, score) ]
         print("score_table =", self.score_table)

         return score
   

    def optimize_merges(self):
        if self.joint_codes:              space = [ (0, self.max_sequences[0]) ]
        elif len(self.max_sequences) > 1: space = [ (0, self.max_sequences[0]), (0, self.max_sequences[1]) ]
        else:                             space = [ (0, self.max_sequences[0]), (0, self.max_sequences[0]) ]
        res = skopt.gp_minimize( self.__vocab_rating, space, n_calls = 10, random_state = self.seed, verbose = self.verbose)
        return res

    def run_experiment(self):
        res = self.optimize_merges()
        if self.verbose: print(res)
        with open(self.results, "w") as out:
            json.dump(res, out, default=convert_value, indent = "    ")
        return res 
#End Experiment class

if __name__ == "__main__":
   parser = create_parser()
   args = parser.parse_args()

   exp = Experiment(  args.codes, args.train, args.dev, dev_sgml = args.dev_sgml, max_sequences=args.max_sequences, vocab_threshold=args.vocabulary_threshold,
                      metric = args.metric, results = args.results,
                      model_prefix = args.model_prefix, train_log_prefix = args.train_log_prefix, vocab_dir = args.vocab_dir, translation_dir = args.translation_dir,
                      dest_lang = args.dest_lang, verbose = args.verbose
   )
   exp.run_experiment()
