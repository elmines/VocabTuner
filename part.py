import xml.etree.ElementTree as ET
import xml.dom.minidom
import argparse
import random

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

    parser.add_argument("--train", nargs=2, required=True, metavar=("<src_path>", "<ref_path>"), type = argparse.FileType("w"), help="Output: training sets in plain text")

    parser.add_argument("--dev", nargs=2, metavar=("<src_path>", "<ref_path>"), type = argparse.FileType("w"), help="Output: dev sets in SGML")
    parser.add_argument("--dev_plain", metavar="<src_path>", type = argparse.FileType("w"), help = "Output: source dev set in plain text")
    parser.add_argument("--dev_id", default="", help="setid XML attribute for development sets")

    parser.add_argument("--test", nargs=2, metavar=("<src_path>", "<ref_path>"), type = argparse.FileType("w"), help="Output: test sets in SGML")
    parser.add_argument("--test_plain", metavar="<src_path>", type = argparse.FileType("w"), help ="Output: source test set in plain text")
    parser.add_argument("--test_id", default="", help="setid XML attribute for test sets")

    parser.add_argument("--ratio", "-r", default = (1.0, 0.0, 0.0), metavar="train:dev:test", type = str2ratio, help = "Ratio of sizes of training, development, and test sets")
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
    if verbose: print("Wrote %s" % path)

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

    train_end = int( len(src_lines) * args.ratio[0])
    train_src = src_lines[0:train_end]
    train_ref = ref_lines[0:train_end]

    dev_end = train_end + int(len(src_lines) * args.ratio[1])
    dev_src = src_lines[train_end:dev_end]
    dev_ref = ref_lines[train_end:dev_end]

    test_src = src_lines[dev_end: ]
    test_ref = ref_lines[dev_end: ]

    if args.verbose:
        print("   Number of total training sequences =", len(train_src))
        print("Number of total development sequences =", len(dev_src))
        print("       Number of total test sequences =", len(test_src))


    write_plain(args.train[0].name, train_src)
    write_plain(args.train[1].name, train_ref)

    if args.dev:
        write_src_xml(args.dev[0].name, dev_src, args.dev_id)
        verbose_message(args.verbose, args.dev[0].name)

        write_ref_xml(args.dev[1].name, dev_ref, args.dev_id, args.trglang)
        verbose_message(args.verbose, args.dev[1].name)
    if args.dev_plain:
        write_plain(args.dev_plain.name, dev_src)
        verbose_message(args.verbose, args.dev_plain.name)

    if args.test:
        write_src_xml(args.test[0].name, test_src, args.test_id)
        verbose_message(args.verbose, args.test[0].name)

        write_ref_xml(args.test[1].name, test_ref, args.test_id, args.trglang)
        verbose_message(args.verbose, args.test[1].name)
    if args.test_plain:
        write_plain(args.test_plain.name, test_src)
        verbose_message(args.verbose, args.test_plain.name)
