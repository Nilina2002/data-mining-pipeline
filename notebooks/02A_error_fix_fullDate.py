import pandas as pd


start_date = pd.Timestamp('2010-12-01')
end_date   = pd.Timestamp('2011-12-31')

dates = pd.date_range(start=start_date, end=end_date, freq='D')

dim_date = pd.DataFrame({
    'DateKey'    : dates.strftime('%Y%m%d').astype(int),
    'FullDate'   : dates.date,   # date only (no time) — required by Power BI
    'Day'        : dates.day,
    'Month'      : dates.month,
    'MonthName'  : dates.strftime('%B'),
    'Quarter'    : dates.quarter,
    'Year'       : dates.year,
    'DayOfWeek'  : dates.day_name(),
    'WeekNumber' : dates.isocalendar().week.astype(int),
    'IsWeekend'  : dates.dayofweek >= 5
})

dim_date.to_csv('../data/warehouse/dim_date.csv', index=False)
print(f" dim_date rebuilt: {len(dim_date)} continuous days")
print(f"   From: {dim_date['FullDate'].min()}")
print(f"   To  : {dim_date['FullDate'].max()}")
print(f"   Gaps: 0 — every day included ")