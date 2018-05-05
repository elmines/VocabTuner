import xml.etree.ElementTree as ET
import xml.dom.minidom
import argparse
import random
import sys

def str2ratio(argument):
    vals = argument.split(":")
    if len(vals) != 3:
        raise ValueError("Size ratio must be of format a:b:c")

    total = 0
    for val in vals:
       total += float(val)
    return tuple( [ float(val) / total for val in vals ] )

def create_parser():
    parser = argparse.ArgumentParser(description="Convert plain-text parallel corpora to SGML format")

    parser.add_argument("--input", "-i", required=True, nargs=2, metavar=("<src_path>", "<ref_path>"), type = argparse.FileType("r"))
    parser.add_argument("--trglang", "-l", default="en", metavar=("xx"), type=str)

    parser.add_argument("--train", nargs=2, required=False, metavar=("<src_path>", "<ref_path>"), type = argparse.FileType("w"), help="Output: training sets in plain text")

    parser.add_argument("--dev", nargs=2, metavar=("<src_path>", "<ref_path>"), type = argparse.FileType("w"), help="Output: dev sets in SGML")
    parser.add_argument("--dev_plain", nargs=2, metavar=("<src_path>", "<ref_path>"), type = argparse.FileType("w"), help = "Output: dev sets in plain text")
    parser.add_argument("--dev_id", default="", help="setid XML attribute for development sets")

    parser.add_argument("--test", nargs=2, metavar=("<src_path>", "<ref_path>"), type = argparse.FileType("w"), help="Output: test sets in SGML")
    parser.add_argument("--test_plain", nargs=2, metavar=("<src_path>", "<ref_path>"), type = argparse.FileType("w"), help ="Output: test sets in plain text")
    parser.add_argument("--test_id", default="", help="setid XML attribute for test sets")


    parser.add_argument("--test-sequences", default=0, metavar="n", type=int, help="Set aside n sequences for testing")
    parser.add_argument("--dev-sequences", default=0, metavar="n", type=int, help="Set aside n sequenes for dev")
    parser.add_argument("--ratio", "-r", default = None, metavar="train:dev:test", type = str2ratio, help = "Ratio of sizes of training, development, and test sets (overrides --dev-sequences and --test-sequences")

    parser.add_argument("--max-sequences", "--max", metavar="n", type=int, help="Maximum number of sequences to use for training, development, and test sets")

    parser.add_argument("--seed", "-s", default=1, metavar="n", type=int, help="Random number seed for shuffling text corpora")

    parser.add_argument("--verbose", "-v", action="store_true", help="Display helpful messages")

    return parser

def write_xml(path, lines, root):
    giant_doc = ET.SubElement(root, "doc")
    giant_doc.set("sysid", "ref")
    giant_doc.set("docid", "xxxxx")
    giant_doc.set("genre", "xxxxx")
    giant_doc.set("origlang", "xx")
    giant_paragraph = ET.SubElement(giant_doc, "p")
    for i in range(len(lines)):
        segment = ET.SubElement(giant_paragraph, "seg")
        segment.text = lines[i].strip()
        segment.set("id", str(i + 1))

    str_rep = xml.dom.minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(newl="\n", indent="")
    str_rep = str_rep[ str_rep.find("\n") + 1 : ] #Hack to get rid of XML document declaration

    #print(str_rep[:2000])
    with open(path, "w", encoding="utf-8") as out:
        out.write(str_rep)

    #ET.ElementTree(root).write(path, encoding = "utf-8", short_empty_elements=False)

def write_ref_xml(path, lines, setid, trglang):
    ref_root = ET.Element("refset")
    ref_root.set("setid", setid)
    ref_root.set("srclang", "any")
    ref_root.set("trglang", trglang)
    write_xml(path, lines, ref_root)

def write_src_xml(path, lines, setid):
    src_root = ET.Element("srcset")
    src_root.set("setid", setid)
    src_root.set("srclang", "any")
    write_xml(path, lines, src_root)

def write_plain(path, lines):
    with open(path, "w", encoding = "utf-8") as out:
        out.write("\n".join(lines))

def verbose_message(verbose, path):
    if verbose: sys.stderr.write("Wrote %s\n" % path)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    random.seed( args.seed )
    init_state = random.getstate()

    with open(args.input[0].name, "r", encoding = "utf-8") as source:
        src_lines = [line.strip() for line in source.readlines()]
        random.shuffle(src_lines)
        if args.max_sequences and args.max_sequences < len(src_lines):
            src_lines = src_lines[:args.max_sequences]
        
    random.setstate(init_state) #Shuffle the ref_lines exactly the same as the src_lines
    with open(args.input[1].name, "r", encoding = "utf-8") as reference:
        ref_lines = [line.strip() for line in reference.readlines()]
        random.shuffle(ref_lines)
        if args.max_sequences and args.max_sequences < len(ref_lines):
            ref_lines = ref_lines[:args.max_sequences]


    if not(args.dev_sequences or args.test_sequences):
        ratio = args.ratio if args.ratio else (1.0, 0.0, 0.0)
        train_start = 0
        train_end = train_start + int( len(src_lines) * args.ratio[0])

        dev_start = train_end
        dev_end = dev_start + int(len(src_lines) * args.ratio[1])

        test_start = dev_end
        test_end = test_start + int(len(src_lines) * args.ratio[2])
    elif args.dev_sequences and args.test_sequences:
        if args.dev_sequences + args.test_sequences > len(src_lines):
            raise ValueError("Requested more total --dev_sequences and --test_sequences than available in --input")
        test_start = 0
        test_end = test_start + args.test_sequences

        dev_start = test_end
        dev_end = dev_start + args.dev_sequences

        train_start = dev_end
        train_end = len(src_lines)
    else:
        if args.dev_sequences:
            if args.dev_sequences > len(src_lines):
                raise ValueError("Requested more total --dev_sequences than available in --input")
            test_start = test_end = 0
            dev_start = 0
            dev_end = dev_start + args.dev_sequences
            train_start = dev_end
        else:
            if args.test_sequences > len(src_lines):
                raise ValueError("Requested more total --test_sequences than available in --input")
            dev_start = dev_end = 0
            test_start = 0
            test_end = test_start + args.test_sequences
            train_start = test_end
        train_end = len(src_lines) 

    train_src = src_lines[train_start:train_end]
    train_ref = ref_lines[train_start:train_end]

    dev_src = src_lines[dev_start:dev_end]
    dev_ref = ref_lines[dev_start:dev_end]

    test_src = src_lines[test_start:test_end]
    test_ref = ref_lines[test_start:test_end]

    if args.verbose:
        print("   Number of total training sequences =", len(train_src))
        print("Number of total development sequences =", len(dev_src))
        print("       Number of total test sequences =", len(test_src))


    if len(train_src) > 0 and args.train:
        write_plain(args.train[0].name, train_src)
        verbose_message(args.verbose, args.train[0].name)

        write_plain(args.train[1].name, train_ref)
        verbose_message(args.verbose, args.train[1].name)
    elif args.train and args.verbose:
        sys.stderr.write("No training lines to write\n")

    if len(dev_src) > 0 and (args.dev or args.dev_plain):
        if args.dev:
            write_src_xml(args.dev[0].name, dev_src, args.dev_id)
            verbose_message(args.verbose, args.dev[0].name)

            write_ref_xml(args.dev[1].name, dev_ref, args.dev_id, args.trglang)
            verbose_message(args.verbose, args.dev[1].name)
        if args.dev_plain:
            write_plain(args.dev_plain[0].name, dev_src)
            verbose_message(args.verbose, args.dev_plain[0].name)

            write_plain(args.dev_plain[1].name, dev_ref)
            verbose_message(args.verbose, args.dev_plain[1].name)
    elif (args.dev or args.dev_plain) and args.verbose:
        sys.stderr.write("No development lines to write\n") 

    if len(test_src) > 0 and (args.test or args.test_plain):
        if args.test:
            write_src_xml(args.test[0].name, test_src, args.test_id)
            verbose_message(args.verbose, args.test[0].name)

            write_ref_xml(args.test[1].name, test_ref, args.test_id, args.trglang)
            verbose_message(args.verbose, args.test[1].name)
        if args.test_plain:
            write_plain(args.test_plain[0].name, test_src)
            verbose_message(args.verbose, args.test_plain[0].name)

            write_plain(args.test_plain[1].name, test_ref)
            verbose_message(args.verbose, args.test_plain[1].name)
    elif (args.test or args.test_plain) and args.verbose:
        sys.stderr.write("No test lines to write\n") 



