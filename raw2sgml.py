
import xml.etree.ElementTree as ET
import argparse
#import codecs


"""
<refset setid="newstest2016" srclang="any" trglang="de">
<doc sysid="ref" docid="Wirtschaftsblatt.at.22162" genre="news" origlang="de">
<p>
<seg id="1">Obama empfängt Netanyahu</seg>
<seg id="2">Das Verhältnis zwischen Obama und Netanyahu ist nicht gerade freundschaftlich.</seg>
<seg id="3">Die beiden wollten über die Umsetzung der internationalen Vereinbarung sowie über Teherans destabilisierende Maßnahmen im Nahen Osten sprechen.</seg>
<seg id="4">Bei der Begegnung soll es aber auch um den Konflikt mit den Palästinensern und die diskutierte Zwei-Staaten-Lösung gehen.</seg>
<seg id="5">Das Verhältnis zwischen Obama und Netanyahu ist seit Jahren gespannt.</seg>
<seg id="6">Washington kritisiert den andauernden Siedlungsbau Israels und wirft Netanyahu mangelnden Willen beim Friedensprozess vor.</seg>
"""

"""
<srcset setid="newstest2016" srclang="any">
<doc sysid="ref" docid="Wirtschaftsblatt.at.22162" genre="news" origlang="de">
<p>
<seg id="1">Obama receives Netanyahu</seg>
<seg id="2">The relationship between Obama and Netanyahu is not exactly friendly.</seg>
<seg id="3">The two wanted to talk about the implementation of the international agreement and about Teheran's destabilising activities in the Middle East.</seg>
<seg id="4">The meeting was also planned to cover the conflict with the Palestinians and the disputed two state solution.</seg>
<seg id="5">Relations between Obama and Netanyahu have been strained for years.</seg>
<seg id="6">Washington criticises the continuous building of settlements in Israel and accuses Netanyahu of a lack of initiative in the peace process.</seg>
<seg id="7">The relationship between the two has further deteriorated because of the deal that Obama negotiated on Iran's atomic programme, .</seg>
<seg id="8">In March, at the invitation of the Republicans, Netanyahu made a controversial speech to the US Congress, which was partly seen as an affront to Obama.</seg>
<seg id="9">The speech had not been agreed with Obama, who had rejected a meeting with reference to the election that was at that time impending in Israel.</seg>
</p>
</doc>
<doc sysid="ref" docid="abcnews.151477" genre="news" origlang="en">
<p>
<seg id="1">In 911 Call, Professor Admits to Shooting Girlfriend</seg>
"""

def create_parser():
    parser = argparse.ArgumentParser(description="Convert plain-text parallel corpora to SGML format")

    parser.add_argument("--input", "-i", required=True, nargs=2, metavar=("<src>", "<ref>"), type = argparse.FileType("r"))
    parser.add_argument("--output", "-o", required=True, nargs=2, metavar=("<src>", "<ref>"), type = argparse.FileType("w"))
    parser.add_argument("--trg_lang", "-l", default="en", metavar=("xx"), type=str)
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

def write_xml(raw_source, raw_ref, xml_source, xml_ref, trg_lang, setid):
    src_root = ET.Element("srcset")
    src_root.set("setid", setid)
    srcset = gen_tree(raw_source, src_root)

    ref_root = ET.Element("refset")
    ref_root.set("setid", setid)
    ref_root.set("trglang", trg_lang)
    refset = gen_tree(raw_ref, ref_root)

    srcset.write(xml_source, encoding = "utf-8")
    refset.write(xml_ref, encoding = "utf-8")

    return 0

parser = create_parser()
args = parser.parse_args()

write_xml( (args.input)[0].name, (args.input[1]).name, (args.output[0]).name, (args.output[1]).name, args.trg_lang, args.id)
