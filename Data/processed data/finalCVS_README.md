# code used in openrefine:

---

##  Price.cvs:
- REGION: delete
- SETTLEMENTDATE: GREL -> value.toDate('dd/MM').toString('MM-dd')
- TOTALDEMAND: no change
- PRICECATEGORY: no change
- Imported into Database
- Using SQL to locate the max price category and the max price

---

## For both the max price and the max price category under one table:
```SQL
select settlementdate, max(totaldemand),  CASE MAX(CASE pricecategory
         WHEN 'LOW' THEN 1
         WHEN 'MEDIUM' THEN 2
         WHEN 'HIGH' THEN 3
         WHEN 'EXTREME' THEN 4
         END)  WHEN 1 THEN 'LOW'
         WHEN 2 THEN 'MEDIUM'
         WHEN 3 THEN 'HIGH'
         WHEN 4 THEN 'EXTREME'
         END AS pricecategory
from price
GROUP BY settlementdate
ORDER BY settlementdate
```

---

# Other:

```SQL
SELECT settlementdate,
       max(totaldemand),
       pricecategory
  FROM price
 GROUP BY settlementdate
 ORDER BY settlementdate;
```

```SQL
select settlementdate, totaldemand,  CASE MAX(CASE pricecategory
         WHEN 'LOW' THEN 1
         WHEN 'MEDIUM' THEN 2
         WHEN 'HIGH' THEN 3
         WHEN 'EXTREME' THEN 4
         END)  WHEN 1 THEN 'LOW'
         WHEN 2 THEN 'MEDIUM'
         WHEN 3 THEN 'HIGH'
         WHEN 4 THEN 'EXTREME'
         END AS pricecategory
from price
GROUP BY settlementdate
ORDER BY settlementdate
```
---

##  Price.cvs:

- Rearrange date format: GREL -> value.toDate('dd/MM').toString('MM-dd')
- Change all numerical columns to numerical data: GREL -> value.toNumber()
- Fix time format: GREL -> value.toNumber()
- Imported into Database
- Using SQL to join both tables for further analysis

```SQL
SELECT *
  FROM new_price p
       INNER JOIN
       weather w ON p.SETTLEMENTDATE = w.Date;
```