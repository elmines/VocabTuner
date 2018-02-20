import xml.etree.ElementTree as ET
import argparse


def str2ratio(argument):
    vals = argument.split(":")
    if len(argument) != 3:
        raise ValueError("Set size ratio can only have 3 quantities")

    total = 0
    for val in vals:
       total += float(val)
    return ( float(val) / total for val in vals )

def create_parser():
    parser = argparse.ArgumentParser(description="Convert plain-text parallel corpora to SGML format")

    #parser.add_argument("--output", "-o", required=True, nargs=2, metavar=("<src>", "<ref>"), type = str)

    parser.add_argument("--input", "-i", required=True, nargs=2, metavar=("<src>", "<ref>"), type = argparse.FileType("r"))
    parser.add_argument("--trglang", "-l", default="en", metavar=("xx"), type=str)

    parser.add_argument("--train", nargs=2, required=True, metavar=("<src>", "<ref>"), type = argparse.FileType("w"), help="Output: training sets in plain text")

    parser.add_argument("--dev", nargs=2, metavar=("<src>", "<ref>"), type = argparse.FileType("w"), help="Output: dev sets in SGML")
    parser.add_argument("--dev-src", nargs=2, metavar="<src>", type = argparse.FileType("w"), help = "Output: source dev set in plain text")

    parser.add_argument("--test", nargs=2, metavar=("<src>", "<ref>"), type = argparse.FileType("w"), help="Output: test sets in SGML")
    parser.add_argument("--test-src", metavar="<src>", type = argparse.FileType("w"), help ="Output: source test set in plain text")

    parser.add_ratio("--ratio", metavar="a:b:c", default = (0.5, 0.25, 0.25), type = str2ratio, help = "Ratio of sizes of source, dev, and test sets")

    parser.add_argument("--id", default="", help="setid XML attribute for source and reference sets")

    return parser

def gen_tree(raw, root):
    giant_doc = ET.SubElement(root, "doc")
    giant_doc.set("sysid", "ref")
    giant_doc.set("docid", "xxxxx")
    giant_doc.set("genre", "xxxxx")
    giant_doc.set("origlang", "xx")
    giant_paragraph = ET.SubElement(giant_doc, "p")

    with open(raw, "r") as r:
        i = 1
        for line in r.readlines():
            segment = ET.SubElement(giant_paragraph, "seg")
            segment.text = line.strip()
            segment.set("id", str(i))
            i += 1

    return ET.ElementTree(root)

def write_xml(raw_source, raw_ref, xml_source, xml_ref, trglang, setid):
    src_root = ET.Element("srcset")
    src_root.set("setid", setid)
    src_root.set("srclang", "any")
    srcset = gen_tree(raw_source, src_root)

    ref_root = ET.Element("refset")
    ref_root.set("setid", setid)
    src_root.set("srclang", "any")
    ref_root.set("trglang", trglang)
    refset = gen_tree(raw_ref, ref_root)

    srcset.write(xml_source, encoding = "utf-8")
    refset.write(xml_ref, encoding = "utf-8")

    return 0

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.train[0].name, "r", encoding = "utf-8") as source:
        src_lines = source.readlines()
    with open(args.train[1].name, "r", encoding = "utf-8") as reference:
        ref_lines = reference.readlines()


    train_end = int( len(src_lines) * args.ratio[0])
    train_src = src_lines[0:train_end]
    train_ref = ref_lines[0:train_end]

    dev_end = train_end + int(len(src_lines) * args.ratio[1])
    dev_src = src_lines[train_end:dev_end]
    dev_ref = ref_lines[train_end:dev_end]

    test_src = src_lines[dev_end: ]
    test_ref = src_lines[dev_end: ]


    write_xml( (args.input)[0].name, (args.input[1]).name, (args.output[0]).name, (args.output[1]).name, args.trglang, args.id)




