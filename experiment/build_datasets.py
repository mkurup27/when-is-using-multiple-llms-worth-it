"""Build the three task sets the experiment runs on.

- classification: 50 Banking77 test messages, one from each of 50 randomly chosen intents
  (PolyAI, CC-BY-4.0).
- extraction: 50 synthetic invoices and billing emails generated from known field values.
  Half are "clean" (one candidate value) and half have distractors (subtotal vs. total,
  issue vs. due date, a later correction).
- research: 25 DRACO tasks (perplexity-ai/draco), sampled across its 10 domains, with rubrics.

Run: python build_datasets.py
"""
import csv
import io
import json
import random
import urllib.request
from collections import defaultdict
from datetime import date, timedelta

from common import DATA, FINAL_INSTRUCTION, write_jsonl

SEED = 4155
BANKING_TEST = ("https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
                "master/banking_data/test.csv")
BANKING_LABELS = ("https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
                  "master/banking_data/categories.json")
DRACO = "https://huggingface.co/datasets/perplexity-ai/draco/resolve/main/test.jsonl"


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8")


def build_classification(rng):
    labels = json.loads(fetch(BANKING_LABELS))
    by_label = defaultdict(list)
    for row in csv.DictReader(io.StringIO(fetch(BANKING_TEST))):
        by_label[row["category"]].append(row["text"])
    chosen = rng.sample(sorted(by_label), 50)
    label_list = "\n".join(f"- {l}" for l in labels)
    rows = []
    for i, label in enumerate(chosen):
        text = rng.choice(by_label[label])
        prompt = (
            "Classify this banking customer message into exactly one intent.\n\n"
            f"Allowed intents:\n{label_list}\n\n"
            f"Message: {text}"
            + FINAL_INSTRUCTION
        )
        rows.append({"id": f"cls-{i:03d}", "task_type": "classification", "kind": "label",
                     "prompt": prompt, "message": text, "answer": label})
    return rows


VENDORS = ["Northwind Supply Co.", "Harbor Lane Logistics", "Bluefin Office Systems",
           "Cedar Ridge Analytics", "Ironleaf Manufacturing", "Quarry Street Print House",
           "Silverline Facilities Group", "Tamarack Cloud Services", "Oakmont Legal Partners",
           "Meridian Freight LLC"]
CLIENTS = ["Acme Retail Ltd.", "Brightpath Health", "Corvid Media", "Delta Ridge Schools",
           "Evergreen Foods"]
ITEMS = ["Consulting hours", "Freight handling", "Annual license", "Printer maintenance",
         "Data migration", "Office supplies", "Support retainer", "Site inspection"]

FIELDS = {
    "total": ("amount", "the total amount due, as a number with two decimals and no currency symbol"),
    "due_date": ("date", "the payment due date, in YYYY-MM-DD format"),
    "invoice_number": ("text", "the invoice number, exactly as written"),
    "vendor": ("text", "the name of the company that issued the invoice"),
    "po_number": ("text", "the customer's purchase order (PO) number, exactly as written"),
}


def money(x):
    return f"${x:,.2f}"


def make_document(rng, distractor):
    vendor, client = rng.choice(VENDORS), rng.choice(CLIENTS)
    issued = date(2026, 1, 5) + timedelta(days=rng.randint(0, 250))
    due = issued + timedelta(days=rng.choice([14, 30, 45, 60]))
    inv = f"INV-{rng.randint(1000, 9999)}"
    po = f"PO-{rng.randint(10000, 99999)}"
    lines = [(rng.choice(ITEMS), round(rng.uniform(120, 4800), 2)) for _ in range(rng.randint(1, 3))]
    subtotal = round(sum(a for _, a in lines), 2)
    tax = round(subtotal * rng.choice([0.0, 0.05, 0.08, 0.2]), 2)
    total = round(subtotal + tax, 2)
    truth = {"total": f"{total:.2f}", "due_date": due.isoformat(), "invoice_number": inv,
             "vendor": vendor, "po_number": po}

    if not distractor:
        body = (f"INVOICE {inv}\nFrom: {vendor}\nBill to: {client}\nPO: {po}\n"
                f"Payment due: {due.strftime('%B %d, %Y')}\n"
                + "".join(f"{d}: {money(a)}\n" for d, a in lines)
                + f"Amount due: {money(total)}\n")
        return body, truth

    delivered = issued - timedelta(days=rng.randint(2, 20))
    old_po = f"PO-{rng.randint(10000, 99999)}"
    body = (
        f"Subject: Invoice {inv} from {vendor}\n\n"
        f"Hi {client} accounts team,\n\n"
        f"Please find our invoice {inv} below, issued {issued.strftime('%d %b %Y')} for work "
        f"delivered on {delivered.strftime('%d %b %Y')}. Our previous invoice "
        f"INV-{rng.randint(1000, 9999)} referenced {old_po}, which has now closed; this one "
        f"is billed against your new purchase order {po}.\n\n"
        + "".join(f"  {d} ........ {money(a)}\n" for d, a in lines)
        + f"  Subtotal ........ {money(subtotal)}\n  Tax ........ {money(tax)}\n"
        f"  Total ........ {money(total + 150)}\n\n"
        f"Terms: net {(due - issued).days} days from the issue date.\n\n"
        f"Correction (sent 1 hour later): the total above included a $150.00 setup fee that "
        f"was already paid. The corrected amount due is {money(total)}. All other details "
        f"are unchanged.\n\nThanks,\nBilling team, {vendor}\n"
    )
    return body, truth


def build_extraction(rng):
    rows = []
    for i in range(50):
        distractor = i % 2 == 1
        doc, truth = make_document(rng, distractor)
        field = list(FIELDS)[i % len(FIELDS)]
        kind, description = FIELDS[field]
        prompt = (f"Extract {description} from this document.\n\n---\n{doc}---"
                  + FINAL_INSTRUCTION)
        rows.append({"id": f"ext-{i:03d}", "task_type": "extraction", "kind": kind,
                     "field": field, "difficulty": "distractor" if distractor else "clean",
                     "prompt": prompt, "answer": truth[field]})
    return rows


def build_research(rng):
    tasks = [json.loads(l) for l in fetch(DRACO).splitlines() if l.strip()]
    by_domain = defaultdict(list)
    for t in tasks:
        by_domain[t["domain"]].append(t)
    picked = []
    domains = sorted(by_domain)
    for d in domains:
        rng.shuffle(by_domain[d])
    while len(picked) < 25:
        for d in domains:
            if by_domain[d] and len(picked) < 25:
                picked.append(by_domain[d].pop())
    rows = []
    for t in picked:
        prompt = (t["problem"] + "\n\nYou do not have web access for this task. Answer as "
                  "thoroughly as you can from your own knowledge, and say where you are unsure.")
        rows.append({"id": f"res-{t['id']}", "task_type": "research", "kind": "rubric",
                     "domain": t["domain"], "draco_id": t["id"], "prompt": prompt,
                     "rubric": json.loads(t["answer"])})
    return rows


AMOUNT_FORMAT = ("as a number with two decimals, using a dot as the decimal separator, "
                 "with no currency symbol or thousands separator")


def eur(x):
    return "€" + f"{x:,.2f}".replace(",", " ").replace(".", ",").replace(" ", ".")


def dmy(d):
    return d.strftime("%d/%m/%Y")


def label(rng, clean, noisy):
    """Scanned-text noise on labels only, never on the values being extracted."""
    return noisy if rng.random() < 0.4 else clean


def hard_balance(rng):
    vendor, client = rng.choice(VENDORS), rng.choice(CLIENTS)
    fmt = eur if rng.random() < 0.5 else money
    inv, other = f"INV-{rng.randint(1000, 9999)}", f"INV-{rng.randint(1000, 9999)}"
    subtotal = round(rng.uniform(2000, 9000), 2)
    total = round(subtotal * 1.2, 2)
    credit = round(rng.uniform(100, 600), 2)
    paid = round(rng.uniform(500, 1500), 2)
    fee = round(rng.uniform(25, 90), 2)
    doc = (
        f"From: Accounts Payable, {client}\nTo: Billing, {vendor}\nSubject: RE: RE: {inv}\n\n"
        f"Hi, we sent a partial payment of {fmt(paid)} against {inv} this morning. "
        f"The rest will follow next week.\n\n"
        f"> From: Billing, {vendor}\n"
        f"> Credit note CN-{rng.randint(100, 999)} for {fmt(credit)} (returned goods) has now "
        f"been applied to {inv}. A late fee of {fmt(fee)} would normally apply, but we are "
        f"waiving it this time.\n>\n"
        f">> From: Billing, {vendor}\n"
        f">> Attached: {inv} and {other}.\n"
        f">> {label(rng, 'Invoice', 'lnvoice')} {inv}: {label(rng, 'Subtotal', 'Subtota1')} "
        f"{fmt(subtotal)}, VAT 20% {fmt(round(total - subtotal, 2))}, "
        f"{label(rng, 'Total', 'Tota1')} {fmt(total)}.\n"
        f">> {label(rng, 'Invoice', 'lnvoice')} {other}: Total {fmt(round(rng.uniform(800, 3000), 2))} "
        f"(already paid in full, for your records).\n"
    )
    q = f"the amount still owed on invoice {inv} after every payment, credit, and fee in this thread, {AMOUNT_FORMAT}"
    return doc, q, f"{total - credit - paid:.2f}", "amount"


def hard_due_date(rng):
    vendor, client = rng.choice(VENDORS), rng.choice(CLIENTS)
    issued = date(2026, 1, 2) + timedelta(days=rng.randint(0, 300))
    net = rng.choice([30, 45, 60])
    extend = rng.random() < 0.5
    inv = f"INV-{rng.randint(1000, 9999)}"
    doc = (
        f"{vendor} | Date format: DD/MM/YYYY\n"
        f"Statement date: {dmy(issued + timedelta(days=rng.randint(3, 9)))}\n"
        f"{label(rng, 'Invoice', 'lnvoice')} {inv} for {client}\n"
        f"Invoice date: {dmy(issued)}\n"
        f"Delivery date: {dmy(issued - timedelta(days=rng.randint(2, 12)))}\n"
        f"Terms: payment due {net} days after the invoice date.\n"
        f"{label(rng, 'Total', 'Tota1')}: {money(round(rng.uniform(400, 6000), 2))}\n"
    )
    due = issued + timedelta(days=net)
    if extend:
        doc += ("\n--- Email from Billing, a week later ---\n"
                f"As discussed on the phone, we have agreed to extend the payment deadline on "
                f"{inv} by 14 days. No other terms change.\n")
        due += timedelta(days=14)
    q = "the date payment is finally due on this invoice, taking any agreed changes into account, in YYYY-MM-DD format"
    return doc, q, due.isoformat(), "date"


def hard_invoice_number(rng):
    client = rng.choice(CLIENTS)
    item = rng.choice(ITEMS)
    others = rng.sample([i for i in ITEMS if i != item], 2)
    ids = rng.sample(range(1000, 9999), 4)
    rows = [(f"INV-{ids[0]}", item, "Paid"), (f"INV-{ids[1]}", item, "Open"),
            (f"INV-{ids[2]}", others[0], "Open"), (f"INV-{ids[3]}", others[1], "Void")]
    rng.shuffle(rows)
    start = date(2026, 1, 5) + timedelta(days=rng.randint(0, 200))
    lines = "".join(
        f"  {inv}  {dmy(start + timedelta(days=7 * k))}  {desc:<20} "
        f"{money(round(rng.uniform(150, 4000), 2)):>12}  {status}\n"
        for k, (inv, desc, status) in enumerate(rows))
    doc = (f"ACCOUNT STATEMENT for {client} (dates DD/MM/YYYY)\n"
           f"  Invoice    Date        Description          Amount        Status\n{lines}")
    q = f"the invoice number of the open (unpaid) invoice for {item.lower()}, exactly as written"
    return doc, q, f"INV-{ids[1]}", "text"


def hard_vendor(rng):
    supplier, forwarder = rng.sample(VENDORS, 2)
    factor = rng.choice(["Keystone Receivables Capital", "Arbor Trade Finance",
                         "Lighthouse Factoring Ltd."])
    client = rng.choice(CLIENTS)
    doc = (
        f"Fwd: invoice for payment\nFrom: {forwarder} (procurement services for {client})\n\n"
        f"Please process the invoice from our supplier below.\n\n"
        f"---------- Forwarded ----------\n"
        f"{factor.upper()} | Receivables Management\n"
        f"{label(rng, 'Invoice', 'lnvoice')} INV-{rng.randint(1000, 9999)}, issued on behalf of "
        f"{supplier} under our factoring agreement, for {rng.choice(ITEMS).lower()} delivered "
        f"to {client}.\nRemit payment to: {factor}\n"
        f"{label(rng, 'Total', 'Tota1')}: {money(round(rng.uniform(400, 6000), 2))}\n"
    )
    q = "the name of the company that supplied the goods or services on this invoice"
    return doc, q, supplier, "text"


def hard_po(rng):
    vendor, client = rng.choice(VENDORS), rng.choice(CLIENTS)
    old_po, new_po = (f"PO-{n}" for n in rng.sample(range(10000, 99999), 2))
    doc = (
        f"{label(rng, 'Invoice', 'lnvoice')} INV-{rng.randint(1000, 9999)} from {vendor} to {client}\n"
        f"PO: {old_po} (see note)\nCustomer ref: CR-{rng.randint(10000, 99999)}\n"
        f"Sales order: SO-{rng.randint(10000, 99999)}\n"
        f"{label(rng, 'Total', 'Tota1')}: {money(round(rng.uniform(400, 6000), 2))}\n\n"
        f"Note from {client} purchasing: {old_po} was cancelled after a budget change and "
        f"replaced by {new_po}. Please bill all remaining work against the replacement order.\n"
    )
    q = "the purchase order (PO) number this invoice should be billed against, exactly as written"
    return doc, q, new_po, "text"


def build_extraction_hard(rng):
    makers = [("balance_due", hard_balance), ("due_date", hard_due_date),
              ("invoice_number", hard_invoice_number), ("vendor", hard_vendor),
              ("po_number", hard_po)]
    rows = []
    for i in range(50):
        field, make = makers[i % len(makers)]
        doc, q, answer, kind = make(rng)
        prompt = f"From this document, extract {q}.\n\n---\n{doc}---" + FINAL_INSTRUCTION
        rows.append({"id": f"exh-{i:03d}", "task_type": "extraction_hard", "kind": kind,
                     "field": field, "difficulty": field, "prompt": prompt, "answer": answer})
    return rows


if __name__ == "__main__":
    import sys

    only = sys.argv[1:]
    rng = random.Random(SEED)
    builders = {"classification": lambda: build_classification(rng),
                "extraction": lambda: build_extraction(rng),
                "research": lambda: build_research(rng),
                "extraction_hard": lambda: build_extraction_hard(random.Random(SEED + 1))}
    sets = {}
    for name, build in builders.items():
        rows = build()  # always build in order so the shared seed sequence never changes
        if not only or name in only:
            sets[name] = rows
    for name, rows in sets.items():
        write_jsonl(DATA / f"{name}.jsonl", rows)
        print(f"{name}: {len(rows)} items -> data/{name}.jsonl")
