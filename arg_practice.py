import argparse

def create_parser():
    parser = argparse.ArgumentParser(description="Obtain optimal number of BPE sequences for a given translation task.")

    parser.add_argument("--train", required=True, nargs=2, metavar=("<source_corpus>", "<dest_corpus>"), help="Training corpora in tokenized, truecased form.")

    parser.add_argument("--dev", required=True, nargs=2, metavar=("<source_corpus>", "<dest_corpus>"), help="Development corpora in SGML")

    parser.add_argument("--proc-dev", required=True, metavar=("<corpus>"), help="Source development corpus that has been cleansed of XML, tokenized, and truecased.")

    parser.add_argument("-l", "--dest-lang", default="en", help="Destination language ISO code")

    parser.add_argument("-c", "--codes", nargs=2, metavar=("<source_codes>", "<dest_codes>"), help="BPE codes file")

    parser.add_argument("-j", "--joint-codes", metavar=("<joint_codes>"), help="BPE codes file learned from both corpora (overrides --codes)")

    parser.add_argument("-m", "--max-codes", type=int, help="Maximum number of BPE codes that can be used to segment a text (<= symbols in --codes)")

    parser.add_argument("--train-log", metavar=("<train-log-prefix>"), help="Marian-style path prefix to be used for training logs")

    parser.add_argument("-v", "--vocab-dir", default=".", metavar=("<vocab_dir>"), help="Path to write vocab files generatred for each number of merges")

    parser.add_argument("-t", "-trans-dir", default=".", metavar=("<trans_dir>"), help="Path to write translated text during validation")

    return parser

parser = create_parser()
args = parser.parse_args()

