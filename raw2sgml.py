
import xml.etree.ElementTree as ET
import argparse
import codecs


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

    parser.add_argument("-i", "--input", required=True, nargs=2, metavar=("<src>", "<ref>"), type = argparse.FileType("r"))
    parser.add_argument("-o", "--output", required=True, nargs=2, metavar=("<src>", "<ref>"), type = argparse.FileType("w"))
    parser.add_argument("-l", "--trg_lang", default="en", metavar=("xx"), type=str)

    return parser

def write_xml(raw_source, raw_ref, xml_source, xml_ref, trg_lang):
    return 0

parser = create_parser()
parser.parse_args()
