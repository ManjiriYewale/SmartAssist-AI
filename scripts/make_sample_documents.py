"""Generate the sample documents in ./sample_data (deterministic).

Requires the dev extras:  pip install reportlab python-docx
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "sample_data"
OUT.mkdir(exist_ok=True)

POLICY_PAGES = [
    ("Acme Corp Employee Handbook - Leave Policy", [
        "Annual leave: Every full-time employee is entitled to 24 days of paid annual leave per calendar year.",
        "Unused annual leave up to 10 days can be carried forward to the next year. Anything above 10 days expires on 31 March.",
        "Sick leave: Employees receive 12 days of paid sick leave per year. A medical certificate is required for absences longer than 3 consecutive days.",
        "Maternity leave is 26 weeks fully paid. Paternity leave is 10 working days fully paid and must be taken within 6 months of the birth.",
    ]),
    ("Acme Corp Employee Handbook - Remote Work and Expenses", [
        "Remote work: Employees may work remotely up to 3 days per week with manager approval. Core collaboration hours are 11:00 to 16:00 IST.",
        "Home office allowance: A one-time allowance of INR 15,000 is provided for setting up a home office, and INR 1,500 per month for internet.",
        "Travel expenses must be submitted within 30 days with original receipts. Hotel stays are reimbursed up to INR 6,000 per night in metro cities.",
        "Security: Company laptops must use full-disk encryption and a VPN when accessing internal systems from outside the office.",
    ]),
]


def make_pdf():
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(OUT / "employee_handbook.pdf"), pagesize=A4)
    for title, paras in POLICY_PAGES:
        c.setFont("Helvetica-Bold", 15)
        c.drawString(50, 790, title)
        c.setFont("Helvetica", 10.5)
        y = 750
        for p in paras:
            words, line = p.split(), ""
            for w in words:
                if len(line) + len(w) > 95:
                    c.drawString(50, y, line)
                    y -= 15
                    line = ""
                line += w + " "
            c.drawString(50, y, line)
            y -= 28
        c.showPage()
    c.save()


def make_docx():
    from docx import Document

    def report(year, revenue, profit, staff, offices, notes, path):
        d = Document()
        d.add_heading(f"Acme Corp Annual Report {year}", 0)
        d.add_heading("Financial highlights", 1)
        d.add_paragraph(f"Total revenue for {year} was INR {revenue} crore, with a net profit of INR {profit} crore.")
        d.add_heading("People", 1)
        d.add_paragraph(f"Acme Corp ended {year} with {staff} employees across {offices} offices.")
        d.add_heading("Strategy", 1)
        for n in notes:
            d.add_paragraph(n, style="List Bullet")
        d.save(path)

    report(2023, 410, 52, 1850, 6,
           ["Focus on enterprise customers in India and the Middle East.",
            "Launched the SmartBooks accounting product in Q3.",
            "Customer churn was 9 percent."], OUT / "annual_report_2023.docx")
    report(2024, 520, 71, 2300, 8,
           ["Expanded into Southeast Asia with offices in Singapore and Jakarta.",
            "Launched the SmartBooks Pro and SmartPay payment products.",
            "Customer churn improved to 6 percent.",
            "Introduced a hybrid work policy allowing 3 remote days per week."], OUT / "annual_report_2024.docx")


def make_csv():
    random.seed(42)
    regions = ["North", "South", "East", "West"]
    products = {"Laptop": 55000, "Phone": 24000, "Tablet": 18000, "Monitor": 12000}
    rows = []
    for i in range(60):
        region, product = random.choice(regions), random.choice(list(products))
        units = random.randint(1, 20)
        price = products[product]
        rows.append([f"2024-{1 + i // 5:02d}-{1 + (i % 5) * 5:02d}", region, product, units, price, units * price])
    rows[7][1] = ""        # deliberate missing values for the missing-value tool
    rows[19][3] = ""
    rows[33][1] = ""
    with open(OUT / "sales_data.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "region", "product", "units", "unit_price", "revenue"])
        w.writerows(rows)


def make_xlsx():
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Q1_Expenses"
    ws.append(["department", "category", "amount", "month"])
    data = [("Engineering", "Cloud", 125000, "Jan"), ("Engineering", "Tools", 32000, "Feb"),
            ("Marketing", "Ads", 98000, "Jan"), ("Marketing", "Events", 156000, "Mar"),
            ("HR", "Recruitment", 74000, "Feb"), ("Sales", "Travel", 88000, "Mar"),
            ("Sales", "Commission", 210000, "Mar"), ("HR", "Training", 41000, "Jan")]
    for r in data:
        ws.append(list(r))
    ws2 = wb.create_sheet("Headcount")
    ws2.append(["department", "employees"])
    for r in [("Engineering", 120), ("Marketing", 35), ("HR", 12), ("Sales", 60)]:
        ws2.append(list(r))
    wb.save(OUT / "expenses_q1.xlsx")


def make_txt():
    (OUT / "project_notes.txt").write_text(
        "Project Phoenix - meeting notes\n\n"
        "The Phoenix migration project moves the billing system from the legacy data centre to the cloud.\n"
        "Go-live date is 15 September 2025. The project budget is INR 4.5 crore and the project lead is Meera Nair.\n"
        "Main risks: data migration errors, vendor delays and staff training. Mitigation: two dry runs and a rollback plan.\n",
        encoding="utf-8")


if __name__ == "__main__":
    make_pdf(); make_docx(); make_csv(); make_xlsx(); make_txt()
    print("Sample documents written to", OUT)
