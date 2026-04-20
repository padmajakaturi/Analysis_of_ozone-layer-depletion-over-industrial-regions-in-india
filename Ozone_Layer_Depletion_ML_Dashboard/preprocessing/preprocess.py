import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path)
    # df['Date'] = pd.to_datetime(df['Date'])
    #df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
    df['Year'] = df['Date'].dt.year
    df = df.dropna()
    return df
