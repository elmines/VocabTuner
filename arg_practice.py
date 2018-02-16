import argparse
parser = argparse.ArgumentParser(description="Obtain optimal number of BPE sequences for a given translation task.")

parser.add_argument("--train", nargs=2, metavar=("<source_corpus>", "<dest_corpus>"), required=True, help="Training corpora in tokenized, truecased form.")

parser.add_argument("-m", "--max-codes", type=int, help="Maximum number of BPE codes that can be used to segment a text")
args = parser.parse_args()

