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
from apply_bpe import BPE


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

def create_parser():
    parser = argparse.ArgumentParser(description="Find optimal number of BPE codes for a translation task")

    parser.add_argument("--train", required=True, nargs=2, metavar=("<src_path>", "<ref_path>"), type=absolute_file, help="Source and destination training corpora in plain text")

    #CORPORA
    parser.add_argument("--dev", nargs="+", metavar=("<dev_corpus>"), type=absolute_file, help="Development corpora in plain text (only need source if using BLEU, both source and dest otherwise")
    parser.add_argument("--dev-sgml", nargs=2, metavar=("<src_plain>", "<ref_plain>"), type=absolute_file, help="Development corpora in SGML (required for BLEU")

    #BPE Merges
    parser.add_argument("--codes", required=True, nargs="+", metavar="<codes_path>", type=os.path.abspath, help="BPE codes file(s) (pass in only one if using joint codes)")
    parser.add_argument("--max-sequences", "-s", default=max_sequences_DEFAULT, nargs="+", metavar="n", type=int, help="Maximum number of codes to use from BPE code file(s) (default %(default)s); include two numbers for different limits on sequences in the source and destination languages")


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


    def __init__(self, codes, train, dev, dev_sgml = None, max_sequences = max_sequences_DEFAULT, #Corpora, BPE codes
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
        if self.metric == "bleu":
            if not dev_sgml: raise ValueError("Must provide SGML development corpora to use BLEU metric")
            self.dev_sgml_source = dev_sgml[0]
            self.dev_sgml_dest = dev_sgml[1]
        elif len(dev) < 2: raise ValueError("Must provied source and destination development corpora in plain text if not using BLEU metric")
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
            self.source_codes = absolute_file(source_codes)
            self.dest_codes = self.source_codes
            if self.verbose: print("Using joint codes for source and target text")
        self.max_sequences = max_sequences

        self.model_prefix = Experiment.__process_prefix(model_prefix, "model", Experiment.__model_extension())
        if self.verbose: print("Set model prefix as %s" % str(self.model_prefix))

        self.train_log_prefix = Experiment.__process_prefix(train_log_prefix, "train", Experiment.__log_extension())
        if self.verbose: print("Set training log prefix as %s" % str(self.train_log_prefix))

        self.vocab_dir = absolute_dir(vocab_dir)
        if self.verbose: print("Set working directory as vocab directory.")

        self.translation_dir = absolute_dir(translation_dir)
        if self.verbose: print("Set working directory as translation directory.")

        self.seed = 1 #FIXME: Make this customizable by the user
        self.dest_lang = dest_lang
        self.score_table = []

        #Temporary variables that change with each iteration of the experiment
        self.source_vocab = None
        self.dest_vocab = None
        self.bpe_train_source = None
        self.bpe_train_dest = None
        self.bpe_dev_source = None
        self.bpe_dev_dest = None


    def __already_used(self, num_merges):
       i = 0
       for record in self.score_table:
           if num_merges == record[0]: return i
           i += 1
       return -1
                         

    def score_nist(self, model, source_vocab, dest_vocab, bpe_dev_source, num_merges):
        sgm_translation = os.path.abspath( os.path.join(self.translation_dir, self.dest_lang + Experiment.__merges_string(num_merges) + ".dev.sgm") )

        translate_command = ["marian-decoder",
                             "--models", str(model),
                             "--input", str(bpe_dev_source),
                             "--vocabs", str(source_vocab), str(dest_vocab),
                             "--devices", str(0),
                             "--quiet"
                            ]

        translating = subprocess.Popen( translate_command,                    universal_newlines=True,                            stdout=subprocess.PIPE)
        desegmenting = subprocess.Popen(["sed", "-r", "s/(@@ )|(@@ ?$)//g"],  universal_newlines=True, stdin=translating.stdout,  stdout=subprocess.PIPE)
        detruecasing = subprocess.Popen(["detruecase.perl"],                  universal_newlines=True, stdin=desegmenting.stdout, stdout=subprocess.PIPE)
        detokenizing = subprocess.Popen(["detokenizer.perl"],                 universal_newlines=True, stdin=detruecasing.stdout, stdout=subprocess.PIPE)
        with open(sgm_translation, "w") as out:
            wrapping = subprocess.Popen(["wrap-xml.perl", self.dest_lang, str(self.dev_source_sgml), "TheUniversityOfAlabama"],
                                         universal_newlines=True, stdin=detokenizing.stdout, stdout=out)
            status = wrapping.wait() 
            if status:
                raise RuntimeError("Translation process with " + str(num_merges) + " merges failed with exit code " + str(status)) 

        if self.verbose: print("Wrote translation %s" % (sgm_translation))
        sys.stdout.flush()

        score_command = ["mteval-v14.pl", "-s", str(self.dev_source_sgml), "-t", str(sgm_translation), "-r", str(self.dev_dest_sgml)]
        scoring = subprocess.Popen(score_command, universal_newlines=True, stdout=subprocess.PIPE)
        output = scoring.communicate()[0]
        status = scoring.wait()
        if status:
            raise RuntimeError("Scoring process with " + str(num_merges) + " merges failed with exit code " + str(status)) 
        sys.stdout.flush()

        return Experiment.parse_nist(output)

    @staticmethod
    def __segment_corpus(raw, processed, bp_encoder):
        with open(raw, "r") as r, open(processed, "w") as p:
            for line in r.readlines():
                p.write(bp_encoder.segment(line).strip())
                p.write("\n")

    def __generate_vocabs(self, bpe_train_source, bpe_train_dest, num_merges):
       if self.joint_codes:
           source_vocab = os.path.join(self.vocab_dir, "joint" + Experiment.__merges_string(num_merges) + Experiment.__vocab_extension())
           dest_vocab = source_vocab

           cat_command = ["cat", str(bpe_train_source), str(bpe_train_dest)]
           cat_proc = subprocess.Popen(cat_command, stdout = subprocess.PIPE, universal_newlines=True)

           with open(source_vocab, "w", encoding="utf-8") as out:
               vocab_command = ["marian-vocab"]
               vocab_proc = subprocess.Popen(vocab_command, stdin = cat_proc.stdout, stdout=out, universal_newlines=True)

           status = vocab_proc.wait()
           if status: raise RuntimeError("Generating joint vocabulary with " + str(num_merges) + " merges failed with exit code " + str(status))
           if self.verbose: print("Wrote joint vocabulary file %s" % source_vocab)


       else:
           source_vocab = os.path.join(self.vocab_dir, "source" + Experiment.__merges_string(num_merges[0]) + Experiment.__vocab_extension())
           dest_vocab = os.path.join(self.vocab_dir, "dest" + Experiment.__merges_string(num_merges[1]) + Experiment.__vocab_extension())

           with open(bpe_train_source, "r", encoding="utf-8") as r, open(source_vocab, "w", encoding="utf-8") as w:
               vocab_proc = subprocess.Popen(["marian-vocab"], universal_newlines=True, stdin=r, stdout=w)
               status = vocab_proc.wait()
           if status: raise RuntimeError("Generating source vocabulary with " + str(num_merges) + " merges failed with exit code " + str(status))
           if self.verbose: print("Wrote vocabulary file %s" % source_vocab)

           with open(bpe_train_dest, "r", encoding="utf-8") as r, open(dest_vocab, "w", encoding="utf-8") as w:
               vocab_proc = subprocess.Popen(["marian-vocab"], universal_newlines=True, stdin=r, stdout=w)
               status = vocab_proc.wait()
           if status: raise RuntimeError("Generating dest vocabulary with " + str(num_merges) + " merges failed with exit code " + str(status))
           if self.verbose: print("Wrote vocabulary files %s" % dest_vocab)

       sys.stdout.flush()


       return (source_vocab, dest_vocab)


    def __preprocess_corpora(self, num_merges):
       source_merges = num_merges[0]
       dest_merges = source_merges if len(num_merges) == 1 else num_merges[1]

       bpe_train_source = os.path.join( str(self.train_source) + Experiment.__merges_string(source_merges))
       bpe_train_dest = os.path.join( str(self.train_dest) + Experiment.__merges_string(dest_merges))

       bpe_dev_source = os.path.join( str(self.dev_source) + Experiment.__merges_string(source_merges))

       with open(self.source_codes, "r", encoding="utf-8") as src_codes:
           source_encoder = BPE(src_codes, source_merges)

           Experiment.__segment_corpus(self.train_source, bpe_train_source, source_encoder)
           if self.verbose: print("Wrote segmented training source corpus to %s" % str(bpe_train_source))

           Experiment.__segment_corpus(self.dev_source, bpe_dev_source, source_encoder)
           if self.verbose: print("Wrote segmented development corpus to %s" % str(bpe_dev_source))

       with open(self.dest_codes, "r", encoding="utf-8") as dst_codes:
           dest_encoder = BPE(dst_codes, dest_merges)

           Experiment.__segment_corpus(self.train_dest, bpe_train_dest, dest_encoder)
           if self.verbose: print("Wrote segmented training destination corpus to %s" % str(bpe_train_dest))
       sys.stdout.flush()

       return (bpe_train_source, bpe_train_dest, bpe_dev_source)


    def __train_brief(self, bpe_train_source, bpe_train_dest, source_vocab, dest_vocab, num_merges):
        """
        Returns the path of the model trained.
        """
        merges_string = Experiment.__merges_string(num_merges)
        model_path = Experiment.__detailed_path(self.model_prefix, merges_string, Experiment.__model_extension())
        log_path = Experiment.__detailed_path(self.train_log_prefix, merges_string, Experiment.__log_extension())

        epoch_size = 10000
        #epoch_size = 100000
        minibatch_size = 2**6
        disp_freq = int(epoch_size / minibatch_size // 100)

        #print(bpe_train_source, bpe_train_dest, source_vocab, dest_vocab)

        marian_command = ["marian", 
                          #General options
                          "--workspace", str(8192),         #8192MB = 8GB 
                          "--log", str(log_path), "--quiet",
                          "--seed", str(self.seed),
                          #Model options
                          "--model", str(model_path),
                          "--type", "s2s",
                          "--dim-emb", str(512), "--dim-rnn", str(1024),
                          "--enc-cell", "lstm", "--enc-cell-depth",      str(2), "--enc-depth", str(4),
                          "--dec-cell", "lstm", "--dec-cell-base-depth", str(4), "--dec-cell-high-depth", str(2), "--dec-depth", str(4),
                          "--skip", "--layer-normalization", "--tied-embeddings",
                          "--dropout-rnn", str(0.2), "--dropout-src", str(0.1), "--dropout-trg", str(0.1),
                          #Training options
                          "--train-sets", str(bpe_train_source), str(bpe_train_dest),
                          "--vocabs", str(source_vocab), str(dest_vocab),
                          "--max-length", str(50), "--max-length-crop",
                          "--after-epochs", str(6),
                          "--disp-freq", str(disp_freq),
                          "--device", str(0),
                          "--mini-batch-fit", #str(32),
                          "--label-smoothing", str(0.1), "--exponential-smoothing"
                          ]

        training = subprocess.Popen(marian_command, universal_newlines=True)
        status = training.wait()
        if status: raise RuntimeError("Training process with " + str(num_merges) + " merges failed with exit code " + str(status))

        return model_path

    def __vocab_rating(self, num_merges):
         sys.stdout.flush()

         index = self.__already_used(num_merges)
         if index != -1:
             score = self.score_table[index][-1]
             if self.verbose: print("Returning cached score of %f" % score)
             return score

         (bpe_train_source, bpe_train_dest, bpe_dev_source) = self.__preprocess_corpora(num_merges)
         self.bpe_train_source = bpe_train_source
         self.bpe_train_dest = bpe_train_dest
         self.bpe_dev_source = bpe_dev_source

         (source_vocab, dest_vocab) = self.__generate_vocabs(bpe_train_source, bpe_train_dest, num_merges)
         self.source_vocab = source_vocab
         self.dest_vocab = dest_vocab


         model_path = self.__train_brief(bpe_train_source, bpe_train_dest, source_vocab, dest_vocab, num_merges)

         if self.metric == "bleu": score = self.score_nist(model_path, source_vocab, dest_vocab, bpe_dev_source, num_merges) 
         else:                     score = 7.0 #Placeholder

         self.score_table = [ (num_merges, score) ]

         print("First score: ", score)
         exit(0)

         return score
   

    def optimize_merges(self):
        if self.joint_codes:              space = [ (0, self.max_sequences[0]) ]
        elif len(self.max_sequences) > 1: space = [ (0, self.max_sequences[0]), (0, self.max_sequences[1]) ]
        else:                             space = [ (0, self.max_sequences[0]), (0, self.max_sequences[0]) ]
        res = skopt.gp_minimize( self.__vocab_rating, space, n_calls = 20, random_state = self.seed, verbose = self.verbose)
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

   exp = Experiment(  args.codes, args.train, args.dev, dev_sgml = args.dev_sgml, max_sequences=args.max_sequences,
                      metric = args.metric, results = args.results,
                      model_prefix = args.model_prefix, train_log_prefix = args.train_log_prefix, vocab_dir = args.vocab_dir, translation_dir = args.translation_dir,
                      dest_lang = args.dest_lang, verbose = args.verbose
   )
   exp.run_experiment()
