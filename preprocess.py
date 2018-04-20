import sys
import os
import argparse
import subprocess

subword_nmt = os.path.join("/home/ualelm", "subword_fork")
if subword_nmt not in sys.path:
    sys.path.append(subword_nmt)
import learn_bpe


codes_suffix = ".codes"
tok_suffix = ".tok"
tc_suffix = ".tc"
tc_model_suffix = tc_suffix + "_model"

num_sequences_DEFAULT = 200000
write_dir_DEFAULT = os.path.abspath(".")
suffix_DEFAULT = tok_suffix + tc_suffix

def create_parser():
    parser = argparse.ArgumentParser(description="Tokenize, truecase, and learn BPE codes of plain-text parallel corpora")

    parser.add_argument("--train", required=True, nargs=2, metavar=("<src_path>", "<ref_path>"), type = argparse.FileType("r"), help="Source and destination training corpora")
    parser.add_argument("--langs", "-l", required=True, nargs=2, metavar=("xx", "xx"), type=str, help="ISO 2-char language codes for source and target languages")

    parser.add_argument("--joint", "-j", action="store_true", help="Generate joint BPE codes file instead of two separate ones")
    parser.add_argument("--num-sequences", "-s", default=num_sequences_DEFAULT, metavar="n", type=int, help="Number of codes to write to a BPE File (default %(default)s)")

    parser.add_argument("--suffix", default=suffix_DEFAULT, metavar=".xxxx", type=str, help="suffix to append to cleaned file names (default %(default)s)")

    parser.add_argument("--write-dir", "-d", default=write_dir_DEFAULT, metavar="<dir>", type=os.path.abspath, help="Directory to write processed corpora (default current directory")

    parser.add_argument("--extra-source", nargs="+", metavar=("<src_path>"), type = argparse.FileType("r"), help="Additional source corpora to tokenize and truecase")
    parser.add_argument("--extra-dest", nargs="+", metavar=("<dest_path>"), type = argparse.FileType("r"), help="Additional destination corpora to tokenize and truecase")

    parser.add_argument("--verbose", "-v", action="store_true", help="Show paths of all files written")

    return parser


def process_training():
    return 0

def tokenize(raw, tok, lang):
    """
    Tokenize plain text corpus.
    """
    with open(raw, "r", encoding = "utf-8") as r, open(tok, "w", encoding = "utf-8") as w:
        tokenizing = subprocess.Popen(["tokenizer.perl", "-l", lang, "-threads", str(4)], stdin=r, stdout=w, universal_newlines=True)
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


def learn_codes(source_corp, dest_corp, source_codes, dest_codes, num_sequences, verbose=False):
    """
    source_corp - file path to source corpus
    dest_corp - file path to destination corpus
    source_codes - file path to write source codes
    dest_codes - file path to write destination codes
    num_sequences - maximum number of sequences to learn
    verbose - print all messages
    """
    with open(source_corp, "r", encoding="utf-8") as corp, open(source_codes, "w", encoding="utf-8") as codes:
        learn_bpe.main(corp, codes, num_sequences, verbose=verbose)
    if verbose: print("Wrote codes file %s" % source_codes)
    with open(dest_corp, "r", encoding="utf-8") as corp, open(dest_codes, "w", encoding="utf-8") as codes:
        learn_bpe.main(corp, codes, num_sequences, verbose=verbose)
    if verbose: print("Wrote codes file %s" % dest_codes)
      
def learn_joint_codes(source_corp, dest_corp, joint_codes, num_sequences, verbose=False):
    """
    source_corp - file path to source corpus
    dest_corp - file path to destination corpus
    joint_codes - file path to write joint codes
    num_sequences - maximum number of sequences to learn
    verbose - print all messages
    """
    temp_file = "temp_cat_file"
    with open(source_corp, "r", encoding="utf-8") as s, open(dest_corp, "r", encoding="utf-8") as d, open(temp_file, "w", encoding="utf-8") as t:
        lines = s.readlines() + d.readlines()
        t.writelines(lines)
    if verbose: print("Concatenated %s and %s to temporary file %s" % (source_corp, dest_corp, temp_file))
    with open(temp_file, "r", encoding="utf-8") as temp, open(joint_codes, "w", encoding="utf-8") as joint:
        learn_bpe.main(temp, joint, num_sequences, verbose=verbose)
    os.remove(temp_file)
    if verbose:
        print("Deleted temporary file %s" % temp_file)
        print("Wrote joint codes file %s" % joint_codes)

def main(train, langs, joint=False, num_sequences=num_sequences_DEFAULT, suffix=suffix_DEFAULT, write_dir=write_dir_DEFAULT, extra_source = None, extra_dest = None, verbose=False):
    if not os.path.isdir(write_dir):
        raise ValueError("write_dir %s does not exist" % os.path.abspath(write_dir))

    (source_clean, source_tc_model) = tok_and_learn_tc(train[0].name, langs[0], suffix=suffix, write_dir=write_dir, verbose=verbose)
    (dest_clean, dest_tc_model) = tok_and_learn_tc(train[1].name, langs[1], suffix=suffix, write_dir=write_dir, verbose=verbose)

    if joint:
        joint_codes = os.path.abspath( os.path.join(write_dir, langs[0] + "-" + langs[1] + codes_suffix) )
        learn_joint_codes(source_clean, dest_clean, joint_codes, num_sequences, verbose)
    else:
        source_codes = os.path.abspath( os.path.join(write_dir, langs[0] + codes_suffix) )
        dest_codes = os.path.abspath( os.path.join(write_dir, langs[1] + codes_suffix) )
        learn_codes(source_clean, dest_clean, source_codes, dest_codes, num_sequences, verbose)

    if extra_source: 
        for source in extra_source:
            cleaned = os.path.abspath( os.path.join(write_dir, os.path.basename(source.name) + suffix) )
            tok_and_tc(source.name, cleaned, langs[0], source_tc_model)
            if verbose: print("Wrote cleaned source corpus %s" % cleaned)

    if extra_dest:
        for dest in extra_dest:
            cleaned = os.path.abspath( os.path.join(write_dir, os.path.basename(dest.name) + suffix) )
            tok_and_tc(dest.name, cleaned, langs[1], dest_tc_model)
            if verbose: print("Wrote cleaned dest corpus %s" % cleaned)
    
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    main(args.train, args.langs,
         joint = args.joint, num_sequences=args.num_sequences, suffix=args.suffix, write_dir=args.write_dir,
         extra_source=args.extra_source, extra_dest=args.extra_dest, verbose=args.verbose)
