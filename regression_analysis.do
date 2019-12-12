cd "C:\Users\josep\python_files\CptS_575\project"
use treat_control.dta, clear

xi: reg diff_abs_prc add i.year
xi: reg diff_market_cap add i.year

xi: reg diff_abs_prc add
xi: reg diff_market_cap add
