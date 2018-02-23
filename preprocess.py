import os
import argparse
import subprocess


write_dir_DEFAULT = os.path.abspath(".")
suffix_DEFAULT = ".tok.tc"

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

def tok_and_tc(raw, cleaned, lang, tc_model):
    with open(raw, "r", encoding = "utf-8") as r, open(cleaned, "w", encoding = "utf-8" as w:
        tokenizing = subprocess.Popen(["tokenizer.perl", "-l", lang], stdin=r, stdout=subprocess.PIPE)
        truecaseing = subprocess.Popen(["truecase.perl", "--model", tc_model], stdin=tokenizeing.stdout, stdout=w)
        status = truecasing.wait()
    if status:
        raise RuntimeError("Text cleaning of %s failed with exit code %d" % (raw, status))

def main(train, codes, langs, joint=False, suffix=suffix_DEFAULT, write_dir=write_dir_DEFAULT, extra_source = None, extra_dest = None):
    
    if not os.path.isdir(write_dir):
        raise ValueError("write_dir %s does not exist" % os.path.abspath(write_dir))

    train_source_clean = os.path.abspath( os.path.join(write_dir, os.path.basename(train[0].name) + suffix) )
    train_dest_clean = os.path.abspath( os.path.join(write_dir, os.path.basename(train[1].name) + suffix) )

    print(train_source_clean)
    print(train_dest_clean)

    

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    main(args.train, args.codes, args.langs, joint = args.joint, suffix=args.suffix, write_dir=args.write_dir, extra_source=args.source, extra_dest=args.dest)
