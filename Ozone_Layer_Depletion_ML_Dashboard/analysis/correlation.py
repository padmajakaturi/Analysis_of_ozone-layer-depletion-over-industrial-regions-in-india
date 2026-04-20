import matplotlib.pyplot as plt

def plot_cfc_vs_ozone(df, region):
    data = df[df['Region'] == region]

    plt.figure()
    # fig, ax = plt.subplots(figsize=(3, 3))
    plt.scatter(data['CFC_ppm'], data['Ozone_DU'])
    plt.xlabel("CFC Emissions (ppm)")
    plt.ylabel("Ozone Concentration (DU)")
    plt.title("CFC Emissions vs Ozone Concentration")
    return plt
