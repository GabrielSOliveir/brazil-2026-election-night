"""Minimal decoder for Brazilian electronic ballot-box bulletins (Boletim de Urna, ``.bu``).

The TSE publishes one signed bulletin per polling section. The file is ASN.1/DER
(an ``EntidadeEnvelopeGenerico`` wrapping an ``EntidadeBoletimUrna`` inside an OCTET
STRING). Rather than depending on the official ASN.1 schema, this walker parses the
generic DER tree (recursing into OCTET STRINGs that themselves contain DER) and pulls
out the presidential race:

* the election block (``ele`` = 6257 for 1st round 2026, 6258 for 2nd round),
* eligible voters (``aptos``) and turnout (``comparecimento``),
* nominal votes per candidate number, blanks and nulls.

It does **not** verify the digital signature.

Usage::

    python tse/bu_decoder.py tests/fixtures/bu_2026_r2_budapest_s1696.bu 6258
"""
from __future__ import annotations
import sys
from dataclasses import dataclass, field


@dataclass
class Node:
    tag: int
    start: int
    end: int
    kids: list["Node"] | None = None


def parse_der(buf: bytes, start: int = 0, end: int | None = None, depth: int = 0) -> list[Node] | None:
    end = len(buf) if end is None else end
    out, i = [], start
    while i < end:
        tag = buf[i]; i += 1
        length = buf[i]; i += 1
        if length & 0x80:
            n = length & 0x7F
            length = int.from_bytes(buf[i:i + n], "big"); i += n
        vs, ve = i, i + length
        if ve > end:
            return None
        node = Node(tag, vs, ve)
        if tag & 0x20:  # constructed
            node.kids = parse_der(buf, vs, ve, depth + 1)
            if node.kids is None:
                return None
        elif tag == 0x04 and length > 2 and depth < 12:  # OCTET STRING that may hold nested DER
            kids = parse_der(buf, vs, ve, depth + 1)
            if kids:
                node.kids = kids
        out.append(node)
        i = ve
    return out


def _int(buf: bytes, n: Node) -> int:
    return int.from_bytes(buf[n.start:n.end], "big")


@dataclass
class PresidentialResult:
    aptos: int
    comparecimento: int | None = None
    votes: dict[int, int] = field(default_factory=dict)
    blank: int = 0
    null: int = 0


def extract_president(buf: bytes, election: int) -> PresidentialResult | None:
    tree = parse_der(buf)
    if tree is None:
        return None
    res: PresidentialResult | None = None

    def walk(nodes: list[Node], inside: bool) -> None:
        nonlocal res
        for n in nodes:
            if not n.kids:
                continue
            k, here = n.kids, inside
            # SEQUENCE { INTEGER election, INTEGER aptos, ... }
            if len(k) > 3 and k[0].tag == 0x02 and _int(buf, k[0]) == election and k[1].tag == 0x02:
                here = True
                res = res or PresidentialResult(aptos=_int(buf, k[1]))
            # SEQUENCE { ENUMERATED tipo, INTEGER comparecimento, ... }
            if here and res and res.comparecimento is None and len(k) >= 2 and k[0].tag == 0x0A and k[1].tag == 0x02:
                res.comparecimento = _int(buf, k[1])
            # SEQUENCE { [1] cargo == 1 (presidente), INTEGER, SEQUENCE OF votos }
            if here and len(k) >= 3 and k[0].tag == 0x81 and _int(buf, k[0]) == 1 and k[1].tag == 0x02 and k[2].tag == 0x30:
                for v in k[2].kids or []:
                    vk = v.kids or []
                    tp = next((x for x in vk if x.tag == 0x81), None)
                    q = next((x for x in vk if x.tag == 0x82), None)
                    ident = next((x for x in vk if x.tag == 0xA3), None)
                    if not tp or not q:
                        continue
                    t, qty = _int(buf, tp), _int(buf, q)
                    if t == 1 and ident and ident.kids:      # nominal vote
                        res.votes[_int(buf, ident.kids[1])] = qty
                    elif t == 2:                              # blank
                        res.blank += qty
                    elif t == 3:                              # null
                        res.null += qty
            walk(k, here)

    walk(tree, False)
    return res


if __name__ == "__main__":
    path, ele = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6258
    print(extract_president(open(path, "rb").read(), ele))
