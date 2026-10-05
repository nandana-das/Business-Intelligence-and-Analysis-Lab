# Exp 15 – Power BI build guide (use if the .pbit does not open cleanly)

## 0. Load data
Home > Get Data > Text/CSV > `data.csv` (this is the UCI Online Retail file) > Transform Data. Paste in Advanced Editor (change path):

```
let
    Source = Csv.Document(File.Contents("D:\Nandana\MTECH\Semester 3\BIA\Lab\Practical 15\data.csv"),[Delimiter=",", Columns=8, Encoding=1252, QuoteStyle=QuoteStyle.None]),
    #"Promoted Headers" = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    #"Changed Type" = Table.TransformColumnTypes(#"Promoted Headers",{{"InvoiceNo", type text}, {"StockCode", type text}, {"Description", type text}, {"Quantity", Int64.Type}, {"InvoiceDate", type datetime}, {"UnitPrice", type number}, {"CustomerID", type text}, {"Country", type text}}, "en-US"),
    #"Removed Cancellations" = Table.SelectRows(#"Changed Type", each not Text.StartsWith([InvoiceNo], "C")),
    #"Valid Rows" = Table.SelectRows(#"Removed Cancellations", each [Quantity] > 0 and [UnitPrice] > 0),
    #"Added SalesAmount" = Table.AddColumn(#"Valid Rows", "SalesAmount", each [Quantity] * [UnitPrice], type number),
    #"Added Date" = Table.AddColumn(#"Added SalesAmount", "Date", each Date.From([InvoiceDate]), type date)
in
    #"Added Date"
```
Rename the query `Online Retail`. Close & Apply.

## 1. Calendar table (Modeling > New table)
```
Calendar = ADDCOLUMNS(CALENDAR(MIN('Online Retail'[Date]), MAX('Online Retail'[Date])),
  "Year", YEAR([Date]), "Month Number", MONTH([Date]), "Month", FORMAT([Date],"MMM"),
  "Quarter", "Q" & FORMAT([Date],"Q"), "Year-Month", FORMAT([Date],"yyyy-MM"))
```
Sort `Month` by `Month Number`. Relationship: Calendar[Date] -> 'Online Retail'[Date] (one to many).

## 2. Measures
```
Total Sales = SUM('Online Retail'[SalesAmount])
Total Quantity = SUM('Online Retail'[Quantity])
Total Transactions = DISTINCTCOUNT('Online Retail'[InvoiceNo])
Average Transaction Value = DIVIDE([Total Sales],[Total Transactions])
Total Customers = DISTINCTCOUNT('Online Retail'[CustomerID])
```

## 3. Pages
- **Page 1 BI Overview:** 4-5 cards; line (Calendar[Year-Month], Total Sales); bar (Country, Total Sales); bar (Description, Total Sales, Filters pane > Top N = 10 by Total Sales); column (Country, Total Quantity); slicers Country, Year, Month, Description.
- **Page 2 AI-Driven Analytics:** Decomposition Tree (Analyze = Total Sales; Explain by = Country, Description). Key Influencers (Analyze = SalesAmount; Explain by = Country, Month, Quantity, UnitPrice). Line chart (Calendar[Date], Total Sales) > Analytics pane: **Find anomalies** On; **Forecast** On, length 30 days.
- **Page 3 Cloud BI:** Home > Publish > workspace; open in Power BI Service; pin Total Sales, Total Customers, trend, top products, country to a dashboard (screenshot it).
- **Page 4 Advanced Visualization:** cards, line, map (Country, size = Total Sales), treemap (Description), matrix heat map (Rows Country, Columns Month, Values Total Sales, Cell elements > Background colour > gradient), forecast chart.
- **Real-time:** import `Live_Sales.csv`, add `Live Sales = Quantity * UnitPrice`, build cards (current sales, orders, quantity) + bars by Product/Country. Append rows, then Refresh.

## 4. Export
File > Export > PDF; screenshots of each page for the record.
