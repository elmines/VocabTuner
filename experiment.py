import sys
import os
import subprocess
import argparse
import skopt

subword_nmt = os.path.join("/home/ualelm", "subword_fork")
if subword_nmt not in sys.path:
    sys.path.append(subword_nmt)
from apply_bpe import BPE


def working_directory():
    return os.path.abspath(".")


max_sequences_DEFAULT=(1000, 1000)

def create_parser():
    parser = argparse.ArgumentParser(description="Find optimal number of BPE codes for a translation task")

    parser.add_argument("--train", required=True, nargs=2, metavar=("<src_path>", "<ref_path>"), type=os.path.abspath, help="Source and destination training corpora")

    parser.add_argument("--dev", required=True, nargs=3, metavar=("<src_sgml>", "<ref_sgml>", "<src_plain>"), type=os.path.abspath, help="Development corpora, with the source in both SGML and plain text.")
    parser.add_argument("--test", required=True, nargs=3, metavar=("<src_sgml>", "<ref_sgml>", "<src_plain>"), type=os.path.abspath, help="Test corpora, with the source in both SGML and plain text.")

    parser.add_argument("--dest-lang", "-l", required=True, metavar="xx", type=str, help="ISO 2-char language code for target language")

    parser.add_argument("--codes", required=True, nargs="+", metavar="<codes_path>", type=os.path.abspath, help="BPE codes file(s) (pass in only one if using joint codes)")

    parser.add_argument("--max-sequences", "-s", default=max_sequences_DEFAULT, nargs="+", metavar="n", type=int, help="Maximum number of codes to use from BPE code file(s) (default %(default)s); include two numbers for different limits on source and dest codes")


    parser.add_argument("--translation-dir", "--trans", default=working_directory(), metavar="<dir>", type=os.path.abspath, help="Directory to write translated text (default current directory")
    parser.add_argument("--vocab-dir", "--vocab", default=working_directory(), metavar="<dir>", type=os.path.abspath, help="Directory to write vocabulary files (default current directory")

    parser.add_argument("--train-log-prefix", metavar="<dir>", type=str, help="Prefix for Marian traning logs")
    parser.add_argument("--model-prefix", metavar="<dir>", type=str, help="Prefix for Marian models")

    parser.add_argument("--verbose", "-v", action="store_true", help="Show additional messages")

    return parser



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

    def __init__(self, codes,
                       train_source, train_dest,
                       dev_source, dev_dest, dev_source_preproc,
                       max_sequences = max_sequences_DEFAULT,
                       dest_lang = "en",
                       model_prefix = None, train_log_prefix = None, vocab_dir = working_directory(), translation_dir = working_directory(),
                       verbose = True):
        """
        dest_lang specifies the target language when using Moses's wrap_xml.perl
        Either joint_vocab must be specified, or both source_vocab and dest_vocab must be specified.
        """

        self.verbose = verbose
        self.train_source = os.path.abspath(train_source)
        self.train_dest = os.path.abspath(train_dest)

        self.dev_source = os.path.abspath(dev_source)
        self.dev_dest = os.path.abspath(dev_dest)
        self.dev_source_preproc = os.path.abspath(dev_source_preproc)

        self.dest_lang = dest_lang

        if len(codes) > 1:
            self.joint_codes = False
            self.source_codes = os.path.abspath(codes[0])
            self.dest_codes = os.path.abspath(codes[1])
        else:
            self.joint_codes = True
            self.source_codes = os.path.abspath(source_codes)
            self.dest_codes = self.source_codes

        self.max_sequences = max_sequences

        self.model_prefix = Experiment.__process_prefix(model_prefix, "model", Experiment.__model_extension())
        if self.verbose: print("Set model prefix as %s" % str(self.model_prefix))

        self.train_log_prefix = Experiment.__process_prefix(train_log_prefix, "train", Experiment.__log_extension())
        if self.verbose: print("Set training log prefix as %s" % str(self.train_log_prefix))

        self.vocab_dir = os.path.abspath(vocab_dir)
        if self.verbose: print("Set working directory as vocab directory.")

        self.translation_dir = os.path.abspath(translation_dir)
        if self.verbose: print("Set working directory as translation directory.")

        self.seed = 1 #FIXME: Make this customizable by the user

        self.score_table = []

    def __already_used(self, num_merges):
       i = 0
       for record in self.score_table:
           if num_merges == record[0]: return i
           i += 1
       return -1

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
                         

    def score_nist(self, model, source_vocab, dest_vocab, bpe_dev_source, num_merges):
        sgm_translation = os.path.abspath( os.path.join(self.translation_dir, self.dest_lang + Experiment.__merges_string(num_merges) + ".dev.sgm") )

        translate_command = ["marian-decoder",
                             "--models", str(model),
                             "--input", str(bpe_dev_source),
                             "--vocabs", str(source_vocab), str(dest_vocab),
                             "--devices", str(0),
                             "--quiet"
                            ]

        translating = subprocess.Popen( translate_command, universal_newlines=True, stdout=subprocess.PIPE)
        desegmenting = subprocess.Popen( ["sed", "-r", "s/(@@ )|(@@ ?$)//g"], stdin=translating.stdout, stdout=subprocess.PIPE)
        detruecasing = subprocess.Popen(["detruecase.perl"], universal_newlines=True, stdin=desegmenting.stdout, stdout=subprocess.PIPE)
        detokenizing = subprocess.Popen(["detokenizer.perl"], universal_newlines=True, stdin=detruecasing.stdout, stdout=subprocess.PIPE)
        with open(sgm_translation, "w") as out:
            wrapping = subprocess.Popen(["wrap-xml.perl", self.dest_lang, str(self.dev_source), "TheUniversityOfAlabama"], universal_newlines=True, stdin=detokenizing.stdout, stdout=out)
            status = wrapping.wait() 
            if status:
                raise RuntimeError("Translation process with " + str(num_merges) + " merges failed with exit code " + str(status)) 

        if self.verbose: print("Wrote translation %s" % (sgm_translation))
        sys.stdout.flush()

        score_command = ["mteval-v14.pl", "-s", str(self.dev_source), "-t", str(sgm_translation), "-r", str(self.dev_dest)]
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

       bpe_dev_source = os.path.join( str(self.dev_source_preproc) + Experiment.__merges_string(source_merges))

       with open(self.source_codes, "r", encoding="utf-8") as src_codes:
           source_encoder = BPE(src_codes, source_merges)

           Experiment.__segment_corpus(self.train_source, bpe_train_source, source_encoder)
           if self.verbose: print("Wrote segmented training source corpus to %s" % str(bpe_train_source))

           Experiment.__segment_corpus(self.dev_source_preproc, bpe_dev_source, source_encoder)
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

        epoch_size = 100000
        minibatch_size = 2**6
        disp_freq = epoch_size // minibatch_size

        #print(bpe_train_source, bpe_train_dest, source_vocab, dest_vocab)

        marian_command = ["marian", 
                          #General options
                          "--workspace", str(8192),         #8192MB = 8GB (on the Pascal GPU, half the memory)
                          "--log", str(log_path),
                          "--quiet",
                          "--seed", str(self.seed),
                          #Model options
                          "--model", str(model_path),
                          "--type", "s2s",
                          "--dim-emb", str(512),
                          "--dim-rnn", str(1024),
                          "--skip",
                          "--layer-normalization",
                          "--tied-embeddings",
                          #Training options
                          "--train-sets", str(bpe_train_source), str(bpe_train_dest),
                          "--vocabs", str(source_vocab), str(dest_vocab),
                          "--after-epochs", str(3),
                          "--disp-freq", str(disp_freq),
                          "--device", str(0),
                          "--mini-batch-fit", str(minibatch_size)
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
         (source_vocab, dest_vocab) = self.__generate_vocabs(bpe_train_source, bpe_train_dest, num_merges)
         model_path = self.__train_brief(bpe_train_source, bpe_train_dest, source_vocab, dest_vocab, num_merges)

         score = self.score_nist(model_path, source_vocab, dest_vocab, bpe_dev_source, num_merges) 
         self.score_table = [ (num_merges, score) ]
         return score
   

    def optimize_merges(self):
        if self.joint_codes: 
            space = [ (0, self.max_sequences[0]) ]
        elif len(self.max_sequences) > 1:
            space = [ (0, self.max_sequences[0]), (0, self.max_sequences[1]) ]
        else:
            space = [ (0, self.max_sequences[0]), (0, self.max_sequences[0]) ]

        res = skopt.gp_minimize( self.__vocab_rating,
                                 space,
                                 n_calls = 10,
                                 random_state = self.seed,
                                 verbose = self.verbose)
        return res

    def run_experiment(self):
        res = self.optimize_merges()
        print(res)

if __name__ == "__main__":
   parser = create_parser()
   args = parser.parse_args()

   exp = Experiment(  args.codes, 
                      args.train[0], args.train[1],
                      args.dev[0], args.dev[1], args.dev[2],
                      dest_lang = args.dest_lang,
                      max_sequences = args.max_sequences,
                      model_prefix = args.model_prefix, train_log_prefix = args.train_log_prefix, vocab_dir = args.vocab_dir, translation_dir = args.translation_dir,
                      verbose = args.verbose
   )
   exp.run_experiment()
