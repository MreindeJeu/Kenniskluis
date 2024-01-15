import pandas as pd
from ydata_profiling import ProfileReport

# Replace 'your_file.csv' with the path to your CSV file
file_path = '/Users/BerenddeJeu/Desktop/School Shizzle/Gebied_07/Vondsten/07_vondsten_lijst.csv'
df = pd.read_csv(file_path, sep=';')

profile = ProfileReport(df, title="Vondsten 07 Profiling Report")
profile.to_file("test.html")