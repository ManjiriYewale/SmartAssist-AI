# Sample documents & demo questions

Upload everything in `sample_data/` (Document Library → select all → *Process*).

| File | Content |
|---|---|
| `employee_handbook.pdf` | 2-page HR policy (leave, remote work, expenses) – tests PDF page citations |
| `annual_report_2023.docx`, `annual_report_2024.docx` | two company reports – tests comparison |
| `sales_data.csv` | 60 sales rows with 3 deliberate missing values – tests dataset analysis |
| `expenses_q1.xlsx` | 2 sheets (`Q1_Expenses`, `Headcount`) – tests Excel + sheet handling |
| `project_notes.txt` | meeting notes – tests TXT + summarisation |

## Questions to try

**Document Q&A (RAG, with sources)**
- How many days of annual leave does an employee get? → 24 days, *employee_handbook.pdf, p. 1*
- What is the home office allowance? → INR 15,000 (+ INR 1,500/month internet)
- Who is the project lead of Phoenix and when is go-live? → Meera Nair, 15 September 2025
- *What is the capital of Mongolia according to my documents?* → must say the information is not in the documents (no hallucination)

**Summarisation**
- Summarize project_notes.txt
- Give me the key points of the employee handbook
- Give a detailed summary of annual_report_2024.docx

**Comparison**
- Compare annual_report_2023.docx and annual_report_2024.docx
- What changed in revenue and headcount between the two reports?

**Dataset analysis (CSV / Excel)**
- Analyze sales_data.csv (overview)
- Find the highest revenue in sales_data.csv (returns the matching row)
- Total revenue per region / per product in sales_data.csv
- Are there missing values in sales_data.csv?
- What is the total amount in expenses_q1.xlsx? And the total employees on the Headcount sheet?

**Calculator**
- Calculate 18% GST on 2500 → GST 450, total 2950 (CGST 225 / SGST 225)
- The price including GST is 1180 – what is the base price? → 1000
- What is the average of 12, 18 and 30?

**Multi-tool**
- What is the total revenue in sales_data.csv, and how much is 18% GST on that total?
- Which documents did I upload, and is expenses_q1.xlsx processed?

**Follow-ups (conversation memory)**
- "What is the sick leave policy?" → then → "and for maternity?"
