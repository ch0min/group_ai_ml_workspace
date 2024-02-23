**_FOLDERS_**
data folder for datasets

- raw er for rå datasets så lav en COPY af raw datasettet før vi cleaner det og fjern \_raw i navnet.
- processed er for cleaned datasets

  src folder for assignments, tasks og opgaveløsninger

  docs og references for henvisninger og guidance dokumenter

**_ WORKSPACE _**
group_ai_ml_workspace.code-workspace kan trykkes på som et "shortcut" for at åbne dette workspace.

**_ ANACONDA _**
conda cheatsheet: https://secure-res.craft.do/v2/GP1nDkEUcB3QgKyx2LMtscNmizCtQsm9AQWArmeezJbf4BBqtB1br4zw5o6ReN4bWUoEWVMVdJP1G5TaNtj2DrSDz7fJg7mfApWXyoX2CPoovSUtbRuKkvbwcd8PCN5jN9bKVpZyyjuZ8g6AWs2Yr2cNQZ87WMrQiEUNdxNKGFwJNfaKvEdV54K1qXdy9FAkVnShc41FqHahgKc7By3X6fDPu5SHY7b3Lv8EJj54ooUUfqncioExAmprtMJ2TxnnpRLMLtWFZY7RFf5WAMDru6hq88ifH5y7X8YH91kcgx7BEtraprzwf6yDDK1UFWQxvdgyjKSa/conda-cheatsheet.pdf 0. environment.yml

1. Download Anaconda
2. Shift+Control P eller Shift+Cmd P og vælg Python Conda interpreter
3. Kør dette i terminalen conda env create -f environment.yml
4. Check om det er korrekt sat op - conda env list
5. conda activate group_ai_ml_workspace
   - Hver gang du åbner workspace activate group_ai_ml_workspace for at køre i Anaconda Environment
6. Din terminal burde se sådan her ud:
   - (group_ai_ml_workspace) (navn)@(navn) group_ai_ml_workspace %
     eller
   - check "conda env list", så burde der være en "\*" ud fra det valgte environment.

**_ BRANCHES _**
Lav din branch
Christoffer
Jonathan
Mark
Oscar
