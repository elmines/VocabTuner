import os
import argparse
import subprocess


tok_suffix = ".tok"
tc_suffix = ".tc"
tc_model_suffix = tc_suffix + "_model"

write_dir_DEFAULT = os.path.abspath(".")
suffix_DEFAULT = tok_suffix + tc_suffix

def create_parser():
    parser = argparse.ArgumentParser(description="Tokenize, truecase, and learn BPE codes of plain-text parallel corpora")

    parser.add_argument("--train", required=True, nargs=2, metavar=("<src_path>", "<ref_path>"), type = argparse.FileType("r"))

    parser.add_argument("--codes", "-c", required=True, nargs="+", metavar=("<codes_path>"), type = argparse.FileType("w"), help="Paths for one or two BPE codes files")
    parser.add_argument("--joint", "-j", action="store_true", help="Generate joint BPE codes file (uses only 1st path of --codes)")


    parser.add_argument("--langs", "-l", required=True, nargs=2, metavar=("xx", "xx"), type=str, help="ISO 2-char language codes for source and target languages")


    parser.add_argument("--suffix", default=suffix_DEFAULT, metavar=".xxxx", type=str, help="suffix to append to output file names (default %(default)s)")

    parser.add_argument("--write-dir", "-d", default=write_dir_DEFAULT, metavar="<dir>", type=os.path.abspath, help="Directory to write processed corpora (default current directory")

    parser.add_argument("--source", nargs="+", metavar=("<src_path>"), type = argparse.FileType("r"), help="Additional source corpora to tokenize and truecase")
    parser.add_argument("--dest", nargs="+", metavar=("<dest_path>"), type = argparse.FileType("r"), help="Additional destination corpora to tokenize and truecase")


    parser.add_argument("--verbose", "-v", action="store_true", help="Show paths of all files written")

    return parser


def process_training():
    return 0

def tokenize(raw, tok, lang):
    """
    Tokenize plain text corpus.
    """
    with open(raw, "r", encoding = "utf-8") as r, open(tok, "w", encoding = "utf-8") as w:
        tokenizing = subprocess.Popen(["tokenizer.perl", "-l", lang], stdin=r, stdout=w, universal_newlines=True)
        status = tokenizing.wait()
    if status:
        raise RuntimeError("Tokenization of %s failed with exit code %d" % (raw, status))

def learn_tc(tok, tc_model):
    """
    Learn truecase model from a tokenized corpus.
    """
    learning = subprocess.Popen(["train-truecaser.perl", "--model", tc_model, "--corpus", tok], universal_newlines=True)
    status = learning.wait()
    if status:
        raise RuntimeError("Learning truecase model from %s failed with exit code %d" % (tok, status))

def truecase(tok, tc, tc_model):
    """
    Truecase a tokenized corpus.
    """
    with open(tok, "r", encoding = "utf-8") as r, open(tc, "w", encoding = "utf-8") as w:
        truecasing = subprocess.Popen(["truecase.perl", "--model", tc_model], stdin=r, stdout=w, universal_newlines=True)
        status = truecasing.wait()
    if status:
        raise RuntimeError("Truecasing of %s failed with exit code %d" % (tok, status))

def tok_and_tc(raw, cleaned, lang, tc_model):
    """
    Tokenize and truecase without the need for any intermediate file
    """
    with open(raw, "r", encoding = "utf-8") as r, open(cleaned, "w", encoding = "utf-8") as w:
        tokenizing = subprocess.Popen(["tokenizer.perl", "-l", lang], stdin=r, stdout=subprocess.PIPE, universal_newlines=True)
        truecasing = subprocess.Popen(["truecase.perl", "--model", tc_model], stdin=tokenizing.stdout, stdout=w, universal_newlines=True)
        status = truecasing.wait()
    if status:
        raise RuntimeError("Text cleaning of %s failed with exit code %d" % (raw, status))

def tok_and_learn_tc(raw, lang, suffix=suffix_DEFAULT, write_dir=write_dir_DEFAULT, verbose = False):
    """
    Tokenize a corpus, learn its truecasing, and return the corpus and its truecase model
    """
    tok = os.path.abspath( os.path.join(write_dir, os.path.basename(raw) + tok_suffix) )
    tokenize(raw, tok, lang)
    if verbose: print("Wrote tokenized corpus %s" % tok)

    tc_model = os.path.abspath( os.path.join(write_dir, lang + tc_model_suffix) )
    learn_tc(tok, tc_model)
    if verbose: print("Learned truecase model %s" % tc_model)

    clean = os.path.abspath( os.path.join(write_dir, os.path.basename(raw) + suffix) )
    truecase(tok, clean, tc_model)
    if verbose: print("Wrote truecased corpus %s" % clean)

    return (clean, tc_model)

def process_train_corp(source_raw, dest_raw, langs, joint=False, suffix=suffix_DEFAULT, write_dir=write_dir_DEFAULT, verbose=False):
    (source_clean, source_tc_model) = tok_and_learn_tc(source_raw, langs[0], suffix=suffix, write_dir=write_dir, verbose=verbose)
    (dest_clean, dest_tc_model) = tok_and_learn_tc(dest_raw, langs[1], suffix=suffix, write_dir=write_dir, verbose=verbose)


def main(train, codes, langs, joint=False, suffix=suffix_DEFAULT, write_dir=write_dir_DEFAULT, extra_source = None, extra_dest = None, verbose=False):
    if not os.path.isdir(write_dir):
        raise ValueError("write_dir %s does not exist" % os.path.abspath(write_dir))
  
    process_train_corp(train[0].name, train[1].name, langs, joint=joint, suffix=suffix, write_dir=write_dir, verbose=verbose)

    """
    for source in extra_source:

    for dest in extra_dest:
    """

    
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    main(args.train, args.codes, args.langs, joint = args.joint, suffix=args.suffix, write_dir=args.write_dir, extra_source=args.source, extra_dest=args.dest, verbose=args.verbose)
