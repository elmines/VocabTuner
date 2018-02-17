
#Command line arguments
import sys

#File paths
import os

#Calling shell scripts
import subprocess

#For Bayessian optimization
import skopt

subword_nmt = os.path.join("/home/ualelm", "subword_fork")
if subword_nmt not in sys.path:
    sys.path.append(subword_nmt)
#print("sys.path = %s" % sys.path)

from apply_bpe import BPE


SKIP_PREPROC = True
SKIP_TRAIN = True

def f():
    print("Hello, world!")

def preprocess(source_lang, dest_lang):
    experiment_dir = os.path.abspath( os.path.join("experiments", source_lang + "-" + dest_lang) )
    data_dir = os.path.join(experiment_dir, "data")
    Preprocessing = subprocess.Popen(["shell/preprocess.sh", data_dir, source_lang, dest_lang], universal_newlines=True)
    Preprocessing.wait()


class Experiment:

    dest_lang = ""

    train_source = None
    train_dest = None

    dev_source_preproc = None  #Preproccessed development set aligned with dev_source
    dev_source = None          #Source SGM file for development set
    dev_dest = None            #Destination SGM file for development set

    joint_codes = None
    source_codes = None
    dest_codes = None
    vocab_prefix = None

    max_merges = 100000 #Constant for now

    best_model = None

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
    def detailed_path(path, num_merges, extension):
        index = path.rfind(extension)
        return os.path.abspath( path[:index] + ".s" + str(num_merges) + extension )

    def __init__(self, train_source, train_dest,
                       dev_source, dev_dest, dev_source_preproc,
                       dest_lang = "en",
                       joint_codes = None,
                       source_codes = None, dest_codes = None,
                       model_prefix = None, train_log_prefix = None, vocab_dir = None, translation_dir = None):
        """
        dest_lang specifies the target language when using Moses's wrap_xml.perl
        Either joint_vocab must be specified, or both source_vocab and dest_vocab must be specified.
        """

        self.train_source = os.path.abspath(train_source)
        self.train_dest = os.path.abspath(train_dest)
        self.dev_source = os.path.abspath(dev_source)
        self.dev_dest = os.path.abspath(dev_dest)
        self.dev_source_preproc = os.path.abspath(dev_source_preproc)
        self.dest_lang = dest_lang

        if joint_codes:
            self.joint_codes = os.path.abspath(joint_codes)
            self.source_codes = self.joint_codes
            self.dest_codes = self.joint_codes
        elif source_codes and dest_codes:
            self.source_codes = os.path.abspath(source_codes)
            self.dest_codes = os.path.abspath(dest_codes)
        else:
            raise ValueError("Must specify either joint_codes or both source_codes and dest_codes")


        self.model_prefix = Experiment.__process_prefix(model_prefix, "model", Experiment.__model_extension())
        self.train_log_prefix = Experiment.__process_prefix(model_prefix, "train", Experiment.__log_extension())

        if not(vocab_dir):
            self.vocab_dir = os.path.abspath( os.path.join(".") )
        else:
            self.vocab_dir = os.path.abspath( vocab_dir )

        if not(translation_dir):
            self.translation_dir = os.path.abspath( os.path.join(".") )
        else:
            self.translation_dir = os.path.abspath( translation_dir )

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
        sgm_translation = os.path.abspath( os.path.join(self.translation_dir, self.dest_lang + ".s" + str(num_merges) + ".tst.sgm") )

        print("Translating development set with %d merges and writing to %s" % (num_merges, sgm_translation))

        #Good
        translate_command = ["shell/translate.sh", str(model), str(source_vocab), str(dest_vocab), str(bpe_dev_source)]
        translating = subprocess.Popen( translate_command, universal_newlines=True, stdout=subprocess.PIPE)

        #status = translating.wait()
        #if status:
        #    raise RuntimeError("Translation process with " + str(num_merges) + " merges failed with exit code " + str(status)) 

        #FIXME: Need to get the proper path for dev_source_sgm
        with open(sgm_translation, "w") as out:
            postprocess_command = ["shell/postprocess.sh", self.dest_lang, str(self.dev_source)]
            postprocessing = subprocess.Popen( postprocess_command, universal_newlines=True, stdin=translating.stdout, stdout=out)
            status = postprocessing.wait()
        if status:
            raise RuntimeError("Translation process with " + str(num_merges) + " merges failed with exit code " + str(status)) 

        print("Finished translating successfully")

        exit(0)

        score_command = ["perl", "/home/ualelm/scripts/mteval-v14.pl", "-s", str(self.dev_source), "-t", str(sgm_translation), "-r", str(self.dev_dest)]
        scoring = subprocess.Popen(score_command, universal_newlines=True, stdout=subprocess.PIPE)
        output = scoring.communicate()[0]
        status = scoring.wait()
        if status:
            raise RuntimeError("Scoring process with " + str(num_merges) + " merges failed with exit code " + str(status)) 


        print("Finished scoring process successfully")
        print(output)
        exit(0)

        return parse_nist(output)

    @staticmethod
    def __preprocess_corpus(raw, processed, bp_encoder):
        with open(raw, "r") as r, open(processed, "w") as p:
            for line in r.readlines():
                p.write(bp_encoder.segment(line).strip())
                p.write("\n")

    def __generate_vocabs(self, bpe_train_source, bpe_train_dest, num_merges):
       if self.joint_codes:
           source_vocab = os.path.join(self.vocab_dir, "joint.s%d" % num_merges)
           dest_vocab = source_vocab

           if SKIP_PREPROC: return (source_vocab, dest_vocab)

           #print("source_vocab = %s" % source_vocab)

           cat_command = ["cat", str(bpe_train_source), str(bpe_train_dest)]
           cat_proc = subprocess.Popen(cat_command, stdout = subprocess.PIPE, universal_newlines=True)

           with open(source_vocab, "w") as out:
               vocab_command = ["/home/ualelm/marian/build/marian-vocab"]
               vocab_proc = subprocess.Popen(vocab_command, stdin = cat_proc.stdout, stdout=out, universal_newlines=True)

           status = vocab_proc.wait()
           if status:
               raise RuntimeError("Generating joint vocabulary with " + str(num_merges) + " merges failed with exit code " + str(status))

       else:
           source_vocab = os.path.join(self.vocab_dir, "source.s%d.vocab" % num_merges)
           dest_vocab = os.path.join(self.vocab_dir, "dest.s%d.vocab" % num_merges)

           if SKIP_PREPROC: return (source_vocab, dest_vocab)

           vocab_proc = subprocess.Popen(["marian/build/marian-vocab"], universal_newlines=True, stdin=bpe_train_source, stdout=source_vocab)
           status = vocab_proc.wait()
           if status:
              raise RuntimeError("Generating source vocabulary with " + str(num_merges) + " merges failed with exit code " + str(status))

           vocab_proc = subprocess.Popen(["marian/build/marian-vocab"], universal_newlines=True, stdin=bpe_train_dest, stdout=dest_vocab)
           status = vocab_proc.wait()
           if status:
              raise RuntimeError("Generating dest vocabulary with " + str(num_merges) + " merges failed with exit code " + str(status))

       return (source_vocab, dest_vocab)


    def __preprocess_corpora(self, num_merges):
       bpe_train_source = os.path.join( str(self.train_source) + ".s" + str(num_merges))
       bpe_train_dest = os.path.join( str(self.train_dest) + ".s" + str(num_merges))

       bpe_dev_source = os.path.join( str(self.dev_source_preproc) + ".s" + str(num_merges))
       bpe_dev_dest = os.path.join( str(self.dev_dest) + ".s" + str(num_merges))

       toReturn = (bpe_train_source, bpe_train_dest, bpe_dev_source, bpe_dev_dest)
       #print("File paths for %d merges:" %  num_merges, str(toReturn))

       if not SKIP_PREPROC:
           #print("source_codes =", self.source_codes)
           with open(self.source_codes, "r") as src_codes:
               source_encoder = BPE(src_codes, num_merges)
               Experiment.__preprocess_corpus(self.train_source, bpe_train_source, source_encoder)
               print("Wrote %s" % str(bpe_train_source))
               Experiment.__preprocess_corpus(self.dev_source, bpe_dev_source, source_encoder)
               print("Wrote %s" % str(bpe_dev_source))


           with open(self.dest_codes, "r") as dst_codes:
               dest_encoder = BPE(dst_codes, num_merges)
               Experiment.__preprocess_corpus(self.train_dest, bpe_train_dest, dest_encoder)
               print("Wrote %s" % str(bpe_train_dest))
               #Experiment.__preprocess_corpus(self.dev_dest, bpe_dev_dest, dest_encoder)
               #print("Wrote %s" % str(bpe_dev_dest))


       #(source_vocab, dest_vocab) = self.__generate_vocabs(bpe_dev_source, bpe_dev_dest, num_merges)
       (source_vocab, dest_vocab) = self.__generate_vocabs(bpe_train_source, bpe_train_dest, num_merges)


       toReturn = (bpe_train_source, bpe_train_dest, bpe_dev_source, bpe_dev_dest, source_vocab, dest_vocab)

       return toReturn

    def vocab_rating(self, num_merges):
         model_path = Experiment.detailed_path(self.model_prefix, num_merges[0], Experiment.__model_extension())
         log_path = Experiment.detailed_path(self.train_log_prefix, num_merges[0], Experiment.__log_extension())

         #print("num_merges = %s" % num_merges[0])
         #print("model_path = %s" % model_path)
         #print("log_path = %s" % log_path)


         #print("About to call preprocess_corpora")
         (bpe_train_source, bpe_train_dest, bpe_dev_source, bpe_dev_dest, source_vocab, dest_vocab) = self.__preprocess_corpora(num_merges[0])

         print(bpe_train_source, bpe_train_dest, bpe_dev_source, bpe_dev_dest, source_vocab, dest_vocab)

         #exit(0)

         if SKIP_TRAIN: return self.score_nist(model_path, source_vocab, dest_vocab, bpe_dev_source, num_merges[0]) 

         train_command = [ "shell/train_brief.sh",
                          str(self.train_source),
                          str(self.train_dest),
                          str(source_vocab),
                          str(dest_vocab),
                          str(model_path),
                          str(log_path)
                         ]

         #print("Starting training on %d merges" % num_merges[0])

         training = subprocess.Popen(train_command, universal_newlines=True)
         (output, err) = training.communicate()
         status = training.wait()

         if status:
             raise RuntimeError("Training process with " + str(num_merges[0]) + " merges failed with exit code " + str(status))

         #print("Finished training on %d merges" % num_merges[0])

         return self.score_nist(model_path, source_vocab, dest_vocab, bpe_dev_source, num_merges[0]) 
   

    def optimize_merges(self):
        res = skopt.gp_minimize( self.vocab_rating,
                                 [ (0, self.max_merges) ],
                                 n_calls = 10,
                                 random_state = 1,
                                  verbose = True)
        return res

    def run_experiment(self):
        res = self.optimize_merges()
        print(res)
    
if __name__ == "__main__":
    source_lang = sys.argv[1]
    dest_lang = sys.argv[2]

    exp = Experiment(source_lang, dest_lang)
    exp.run_experiment()
