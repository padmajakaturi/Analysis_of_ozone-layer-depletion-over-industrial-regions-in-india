import numpy as np

def ozone_trend(df, region):
    data = df[df['Region'] == region]
    slope = np.polyfit(data['Year'], data['Ozone_DU'], 1)[0]
    return slope
