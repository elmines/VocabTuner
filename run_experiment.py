
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

    source_lang = ""
    dest_lang = ""
    joint_vocab = True
    experiment_dir = ""

    max_codes = 100000 #Constant for now

    def __init__(self, source_lang, dest_lang, joint_vocab = True):
        self.source_lang = source_lang
        self.dest_lang = dest_lang
        self.experiment_dir = os.path.abspath( os.path.join("experiments", source_lang + "-" + dest_lang) )
        self.joint_vocab = joint_vocab


        if joint_vocab:
            source_vocab = os.path.join(self.experiment_dir, "data", source_lang + dest_lang + ".yml") 
            dest_vocab = source_vocab
        else: #FIMXE: not the exact file paths I want to use for non-joint vocabs
            source_vocab = os.path.join(self.experiment_dir, "data", source_lang + ".vocab.yml") 
            dest_vocab = os.path.join(self.experiment_dir, "data", dest_lang + ".vocab.yml") 

                          

    def optimize_merges(self):
        res = skopt.gp_minimize( self.f,
                                 [ (0, self.max_codes) ],
                                 n_calls = 10,
                                  verbose = True)

        return res
    

    def vocab_rating(num_codes):

         model_path = os.path.join(self.experiment_dir,
                                              "models",
                                              source_lang + "-" + dest_lang + ".s" + str(num_codes) + ".npz"
                                  )

         train_command = [ str(os.path.join(self.experiment_dir, "train_brief.sh")),
                          self.source_lang,
                          self.dest_lang,
                          str(source_vocab),
                          str(dest_vocab),
   
    #Function to be minimized by skopt
    def f(self, x):
        return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) * (np.random.randn() * 0.1)

    
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
