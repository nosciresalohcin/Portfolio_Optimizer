import pandas as pd, requests
from io import StringIO

url = "https://docs.google.com/spreadsheets/d/1DgZZmm4UKEKcYFAVn4W06oUMrMI9KKDFpk6zaL6-9Yw/export?format=csv&gid=1056429339"
r = requests.get(url)
r.raise_for_status()
df = pd.read_csv(StringIO(r.text))
print(df.shape)
print(df.head())
