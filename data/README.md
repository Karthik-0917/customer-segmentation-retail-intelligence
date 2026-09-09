# Dataset: UCI Online Retail

The raw dataset is **gitignored** and must be downloaded manually — no API
keys, credentials, or paid services required.

## Source

- **Name:** Online Retail (UCI Machine Learning Repository, dataset id 352)
- **Official URL:** https://archive.ics.uci.edu/dataset/352/online+retail
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0),
  as stated on the UCI dataset page.
- **Attribution:** Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine
  Learning Repository. https://doi.org/10.24432/C5BW33
- **Description:** All transactions between 01/12/2010 and 09/12/2011 for a
  UK-based, registered non-store online retailer selling mainly unique
  all-occasion gifts; many customers are wholesalers.

## How to obtain

1. Open the official URL above and click **Download** (a `.zip`).
2. Extract it — you get a workbook named `Online Retail.xlsx`.
3. Rename/save it to exactly:

   ```
   data/Online_Retail.xlsx
   ```

   (the underscore replaces the space; `src/data_loader.py` expects this
   path, configured in `configs/config.yaml`).

Prefer the official UCI source over third-party mirrors. This project uses
Online Retail (not Online Retail II).

## Expected schema (transaction-level)

| Column      | Type     | Notes                                                     |
| ----------- | -------- | --------------------------------------------------------- |
| InvoiceNo   | string   | Invoice number; a leading **"C"** marks a cancellation    |
| StockCode   | string   | Product code                                              |
| Description | string   | Product name (has missing values)                         |
| Quantity    | int      | Units per row; negative on cancellations/adjustments      |
| InvoiceDate | datetime | Invoice date and time                                     |
| UnitPrice   | float    | Sterling price per unit; zero/negative on special rows    |
| CustomerID  | float    | Customer identifier; **missing on a large share of rows** |
| Country     | string   | Customer country                                          |

## Why the raw file is not committed

It is ~23 MB of third-party transaction-level data. Committing it would
bloat the repository and is unnecessary: every derived artifact in
`models/` and `outputs/` is reproducible from this file via
`python -m src.pipeline`. Only this README is committed from `data/`.

## Cleaning assumptions (implemented in `src/preprocessing.py`)

- Exact duplicate rows are treated as feed artifacts and dropped.
- Rows without `CustomerID` cannot be attributed to a customer and are
  excluded from customer-level segmentation.
- Invoices starting with "C" are cancellations (documented UCI convention)
  and are excluded from the positive-purchase dataset.
- Remaining rows with non-positive `Quantity` or `UnitPrice` (bad-debt
  adjustments such as invoices starting with "A", manual corrections,
  zero-priced samples) are excluded.
- Every removal is counted and reported in `outputs/data_quality.json`.
