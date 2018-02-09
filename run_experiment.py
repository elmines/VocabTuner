
#Command line arguments
import sys

#File paths
import os

#Calling shell scripts
import subprocess

#For Bayessian optimization
import skopt

import numpy as np

def preprocess(source_lang, dest_lang):
    experiment_dir = os.path.abspath( os.path.join("experiments", source_lang + "-" + dest_lang) )
    data_dir = os.path.join(experiment_dir, "data")
    Preprocessing = subprocess.Popen(["shell/preprocess.sh", data_dir, source_lang, dest_lang], universal_newlines=True)
    Preprocessing.wait()


class Experiment:

    source_vocab = None
    dest_vocab = None
    dest_lang = ""

    max_codes = 100000 #Constant for now

    best_model = None

    

    @staticmethod
    def __log_extension():
        return ".log"

    @staticmethod
    def __model_extension():
        return ".npz"

    @staticmethod
    def detailed_path(path, num_codes, extension):
        index = path.rfind(extension)
        return os.path.abspath( path[:index] + ".s" + str(num_codes) + extension )

    def __init__(self, training_source, training_dest, dev_source, dev_dest, dest_lang = "en", joint_vocab = None, source_vocab = None, dest_vocab = None, model_prefix = None, train_log_prefix = None, translation_dir = None):
        """
        dest_lang specifies the target language when using Moses's wrap_xml.perl
        Either joint_vocab must be specified, or both source_vocab and dest_vocab must be specified.
        """

        self.training_source = os.path.abspath(training_source)
        self.training_dest = os.path.abspath(training_dest)
        self.dev_source = os.path.abspath(dev_source)
        self.dev_dest = os.path.abspath(dev_dest)
        self.dest_lang = dest_lang

        if joint_vocab:
            self.source_vocab = os.path.abspath(joint_vocab)
            self.dest_vocab = self.source_vocab
        elif source_vocab and dest_vocab
            self.source_vocab = source_vocab
            self.dest_vocab = dest_vocab
        else:
            raise ValueError("Must specify either joint_vocab or both source_vocab and dest_vocab")


        if not(model_prefix):
           self.model_prefix = "model" + Experiment.__model_extension()
        elif not model_prefix.endswith(Experiment.__model_extension()) :
           self.model_prefix = model_prefix + Experiment.__model_extension()
        else:
           self.model_prefix = model_prefix


        if not(train_log_prefix):
           self.train_log_prefix = "train" + Experiment.__log_extension()
        elif not train_log_prefix.endswith(Experiment.__log_extension()):
           self.train_log_prefix = train_log_prefix + Experiment.__log_extension()
        else:
            self.train_log_prefix = train_log_prefix


        if not(translation_dir):
            self.translation_dir = os.path.abspath( os.path(".") )
        else:
            self.translation_dir = translation_dir

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
                return maxRawScore = float(scoreString)

        return float("inf")
                         

    def score_nist(self, model_path, num_codes):
        #Translate validation set text
        #Postprocess translation (including XML)
        #Score translation using mteval-v14.pl
    

        sgm_translation = os.path.abspath( os.path.join(self.translation_dir, dest_lang + ".s" + str(num_codes) + ".tst.sgm") )


        translate_command = [      "shell/translate.sh", str(self.source_vocab), str(self.dest_vocab), str(self.dev_source),
                              "|", "shell/postprocess.sh", self.dest_lang, str(self.dev_source),
                              ">", str(sgm_translation)
                            ]

        score_command = [ "~/scripts/mteval-v14.pl", "-s" str(self.dev_source), "-t", str(sgm_translation), "-r", str(self.dev_dest)]

        translating = subprocess.Popen( translate_command, universal_newlines=True, stdout=subprocess.PIPE)
        output = translating.communicate()[0] 
        status = translating.wait()
        if status:
            raise RuntimeError("Translation process with " + str(num_codes) + " merges failed with exit code " + str(status)) 

        print("Finished translating successfully")

        scoring = subprocess.Popen(score_command, universal_newlines=True, stdout=subprocess.PIPE)
        output = scoring.communicate()[0]
        status = scoring.wait()
        if status:
            raise RuntimeError("Scoring process with " + str(num_codes) + " merges failed with exit code " + str(status)) 

        print("Finished scoring process successfully")

        return parse_nist(output)


    def vocab_rating(self, num_codes):
         model_path = Experiment.detailed_path(self.model_prefix, num_codes, Experiment.__model_extension())
         log_path = Experiment.detailed_path(self.train_log_prefix, num_codes, Experiment.__log_extension())


         train_command = [ "shell/train_brief.sh",
                          str(self.training_source),
                          str(self.training_dest),
                          str(source_vocab),
                          str(dest_vocab),
                          str(model_path)
                          str(log_path)
                         ]

         training = subprocess.Popen(train_command, universal_newlines=True)
         (output, err) = training.communicate()
         status = training.wait()

         if status:
             raise RuntimeError("Training process with " + str(num_codes) + " merges failed with exit code " + str(status))

         print("Finished training successfully")

         return self.scoreNist(model_path, num_codes) 
   

    def optimize_merges(self):
        res = skopt.gp_minimize( self.vocab_rating,
                                 [ (0, self.max_codes) ],
                                 n_calls = 10,
                                  verbose = True)
        return res

    def run_experiment(self):
        res = self.optimize_merges()
        print(res)
    
if __name__ == "__main__":
    source_lang = sys.argv[1]
    dest_lang = sys.argv[2]
    command = sys.argv[3] if len(sys.argv) > 3 else ""

    if "train" in command.lower():
        preprocess(source_lang, dest_lang)
    else:
        exp = Experiment(source_lang, dest_lang)
        exp.run_experiment()
