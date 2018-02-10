
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
    dev_source = None
    dev_dest = None
    joint_codes = None
    source_codes = None
    dest_codes = None

    max_merges = 100000 #Constant for now

    best_model = None

    @staticmethod
    def __log_extension():
        return ".log"

    @staticmethod
    def __model_extension():
        return ".npz"

    @staticmethod
    def detailed_path(path, num_merges, extension):
        index = path.rfind(extension)
        return os.path.abspath( path[:index] + ".s" + str(num_merges) + extension )

    def __init__(self, train_source, train_dest, dev_source, dev_dest,
                       dest_lang = "en",
                       joint_codes = None,
                       source_codes = None, dest_codes = None,
                       model_prefix = None, train_log_prefix = None, translation_dir = None):
        """
        dest_lang specifies the target language when using Moses's wrap_xml.perl
        Either joint_vocab must be specified, or both source_vocab and dest_vocab must be specified.
        """

        self.train_source = os.path.abspath(train_source)
        self.train_dest = os.path.abspath(train_dest)
        self.dev_source = os.path.abspath(dev_source)
        self.dev_dest = os.path.abspath(dev_dest)
        self.dest_lang = dest_lang

        print("train_source = %s" % train_source)
        print("train_dest = %s" % train_dest)
        #print("dev_source = %s" % dev_source)
        #print("dev_dest = %s" % dev_dest)
        #print("dest_lang = %s" % dest_lang)

        if joint_codes:
            self.joint_codes = os.path.abspath(joint_codes)
            self.source_codes = self.joint_codes
            self.dest_codes = self.joint_codes
        elif source_codes and dest_codes:
            self.source_codes = os.path.abspath(source_codes)
            self.dest_codes = os.path.abspath(dest_codes)
        else:
            raise ValueError("Must specify either joint_codes or both source_codes and dest_codes")

        print("joint_codes = %s, source_codes = %s, dest_codes = %s" % (self.joint_codes, self.source_codes, self.dest_codes))


        if not(model_prefix):
           self.model_prefix = "model" + Experiment.__model_extension()
        elif not model_prefix.endswith(Experiment.__model_extension()) :
           self.model_prefix = model_prefix + Experiment.__model_extension()
        else:
           self.model_prefix = model_prefix

        #print("model_prefix = %s" % self.model_prefix) 

        if not(train_log_prefix):
           self.train_log_prefix = "train" + Experiment.__log_extension()
        elif not train_log_prefix.endswith(Experiment.__log_extension()):
           self.train_log_prefix = train_log_prefix + Experiment.__log_extension()
        else:
            self.train_log_prefix = train_log_prefix

        #print("train_log_prefix = %s" % self.train_log_prefix)

        if not(translation_dir):
            self.translation_dir = os.path.abspath( os.path.join(".") )
        else:
            self.translation_dir = os.path.abspath( translation_dir )

        #print("translation_dir = %s" % self.translation_dir)

    def parse_nist(report):
        scoreLabel = "NIST score = "
        maxRawScore = 100

        if type(report) != str:
            lines = report.readlines()
        else:
            lines = report.split("\n")
         
        while line and (not line.startswith(scoreLabel)):
            line = fp.readline()

        for line in lines:
            if line.startswith(scoreLabel):
                startIndex = line.index(scoreLabel)
                scoreIndex = startIndex + len(scoreLabel)
                scoreString = line[scoreIndex : ].split(" ")[0]
                print("Extracting the NIST score")
                return maxRawScore - float(scoreString)

        return float("inf")
                         

    def score_nist(self, model_path, num_merges):
        #Translate validation set text
        #Postprocess translation (including XML)
        #Score translation using mteval-v14.pl
    

        sgm_translation = os.path.abspath( os.path.join(self.translation_dir, dest_lang + ".s" + str(num_merges) + ".tst.sgm") )

        print("Translating development set with %d merges and writing to %s" % (num_merges, sgm_translation))

        translate_command = [      "shell/translate.sh", str(self.source_vocab), str(self.dest_vocab), str(self.dev_source),
                              "|", "shell/postprocess.sh", self.dest_lang, str(self.dev_source),
                              ">", str(sgm_translation)
                            ]

        score_command = [ "~/scripts/mteval-v14.pl", "-s", str(self.dev_source), "-t", str(sgm_translation), "-r", str(self.dev_dest)]

        translating = subprocess.Popen( translate_command, universal_newlines=True, stdout=subprocess.PIPE)
        output = translating.communicate()[0] 
        status = translating.wait()
        if status:
            raise RuntimeError("Translation process with " + str(num_merges) + " merges failed with exit code " + str(status)) 

        print("Finished translating successfully")

        scoring = subprocess.Popen(score_command, universal_newlines=True, stdout=subprocess.PIPE)
        output = scoring.communicate()[0]
        status = scoring.wait()
        if status:
            raise RuntimeError("Scoring process with " + str(num_merges) + " merges failed with exit code " + str(status)) 

        print("Finished scoring process successfully")

        return parse_nist(output)

    @staticmethod
    def __preprocess_corpus(raw, processed, bp_encoder):
        with (open(raw, "r"), open(processed, "w")) as r, p:
            for line in r.readlines():
                p.write(bp_encoder.segment(line).strip())
                p.write("\n")

    def __generate_vocabs(self, bpe_train_source, bpe_train_dest, num_merges):
       if self.joint_codes:
           source_vocab = os.path.join( "joint" + ".s" + num_merges + ".vocab")
           dest_vocab = source_vocab
           vocab_command = ["cat", str(bpe_train_source), str(bpe_train_dest), "|",
                            "marian/build/marian-vocab", ">", str(source_vocab)
                           ]

           vocab_proc = subprocess.Popen(vocab_command, universal_newlines=True)
           status = vocab_proc.wait()
           if status:
               raise RuntimeError("Generating joint vocabulary with " + str(num_merges[0]) + " merges failed with exit code " + str(status))

       else:
           source_vocab = os.path.join( "source.s%d.vocab" % num_merges)
           dest_vocab = os.path.join( "dest.s%d.vocab" % num_merges)

           vocab_proc = subprocess.Popen(["marian/build/marian-vocab"], universal_newlines=True, stdin=bpe_train_source, stdout=source_vocab)
           status = vocab_proc.wait()
           if status:
              raise RuntimeError("Generating source vocabulary with " + str(num_merges[0]) + " merges failed with exit code " + str(status))

           vocab_proc = subprocess.Popen(["marian/build/marian-vocab"], universal_newlines=True, stdin=bpe_train_dest, stdout=dest_vocab)
           status = vocab_proc.wait()
           if status:
              raise RuntimeError("Generating dest vocabulary with " + str(num_merges[0]) + " merges failed with exit code " + str(status))

       return (source_vocab, dest_vocab)


    def __preprocess_corpora(self, num_merges):
       #FIXME
       bpe_train_source = os.path.join( str(self.train_source) + ".s" + num_merges)
       bpe_train_dest = os.path.join( str(self.train_dest) + ".s" + num_merges)

       bpe_dev_source = os.path.join( str(self.dev_source) + ".s" + num_merges)
       bpe_dev_dest = os.path.join( str(self.dev_dest) + ".s" + num_merges)

       toReturn = (bpe_train_source, bpe_train_dest, bpe_dev_source, bpe_dev_dest)
       print("File paths for %d merges:" % num_merges, str(toReturn))
       exit(0)

       with open(self.source_codes, "r") as src_codes:
           source_encoder = BPE(src_codes, num_merges)
           Experiment.__preprocess_corpus(self.train_source, bpe_train_source, source_encoder)
           Experiment.__preprocess_corpus(self.dev_source, bpe_dev_source, source_encoder)


       with open(self.dest_codes, "r") as dst_codes:
           dest_encoder = BPE(dst_codes, num_merges)
           Experiment.__preprocess_corpus(self.train_dest, bpe_train_dest, dest_encoder)
           Experiment.__preprocess_corpus(self.dev_dest, bpe_dev_dest, source_dest)

       (source_vocab, dest_vocab) = self.__generate_vocabs(bpe_train_source, bpe_train_dest, num_merges)


       toReturn = (bpe_train_source, bpe_train_dest, bpe_dev_source, bpe_dev_dest, source_vocab, dest_vocab)
       print("File paths for %d merges:" % num_merges, str(toReturn))

       return toReturn

    def vocab_rating(self, num_merges):
         model_path = Experiment.detailed_path(self.model_prefix, num_merges[0], Experiment.__model_extension())
         log_path = Experiment.detailed_path(self.train_log_prefix, num_merges[0], Experiment.__log_extension())

         #print("num_merges = %s" % num_merges[0])
         #print("model_path = %s" % model_path)
         #print("log_path = %s" % log_path)


         (bpe_train_source, bpe_train_dest, bpe_dev_source, bpe_dev_dest, source_vocab, dest_vocab) = self.__preprocess_corpora(num_merges)
         #Assign to source_vocab
         #Assign to dest_vocab

         train_command = [ "shell/train_brief.sh",
                          str(self.train_source),
                          str(self.train_dest),
                          str(source_vocab),
                          str(dest_vocab),
                          str(model_path),
                          str(log_path)
                         ]

         print("Starting training on %d merges" % num_merges)

         training = subprocess.Popen(train_command, universal_newlines=True)
         (output, err) = training.communicate()
         status = training.wait()

         if status:
             raise RuntimeError("Training process with " + str(num_merges[0]) + " merges failed with exit code " + str(status))

         print("Finished training on %d merges" % num_merges[0])

         return self.score_nist(model_path, num_merges[0]) 
   

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
