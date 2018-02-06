import sys
import os
import subprocess

class Experiment:

    source_lang = ""
    dest_lang = ""

    def __init__(self, source_lang, dest_lang):
        self.source_lang = source_lang
        self.dest_lang = dest_lang
        self.run_experiment()

    def preprocess(source_lang, dest_lang):
        experiment_dir = os.path.abspath( os.path.join("experiments", source_lang + "-" + dest_lang) )
        data_dir = os.path.join(experiment_dir, "data")
    
        #print("experiment_dir = ", experiment_dir, ", data_dir = ", data_dir, sep="")
    
        Preprocessing = subprocess.Popen(["./preprocess.sh", data_dir, source_lang, dest_lang], universal_newlines=True)
    
        Preprocessing.wait()
    
        #print("Finished subprocess")
    
    #Function to be minimized by skopt
    def vocab_rating(vocab_size):
        
    
    def run_experiment(self):
    
if __name__ == "__main__":
    source_lang = sys.argv[1]
    dest_lang = sys.argv[2]
    Experiment(source_lang, dest_lang)
