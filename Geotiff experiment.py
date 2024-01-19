import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Open een GeoTIFF-bestand
geo_tiff_path = "/Users/charell/Downloads/FAA_Cont1mGal_UTM18N_NAD83.tif"
dataset = rasterio.open(geo_tiff_path)

# Haal de geotransformatie op voor het bepalen van de x-coördinaten
geotransform = dataset.transform
x_size = dataset.width
x_coordinates = np.linspace(geotransform[0], geotransform[0] + geotransform[1] * x_size, x_size)

# Toon metadata van alle banden
for i in range(1, dataset.count + 1):
    band = dataset.read(i)
    metadata = dataset.tags(i)
    print(f"Band {i} Metadata:")
    print(f"  - Description: {metadata.get('DESCRIPTION', 'Niet gespecificeerd')}")
    print(f"  - Min value: {np.min(band)}")
    print(f"  - Max value: {np.max(band)}")
    print()

# Lees de rasterband met de vrij lucht anomalieën
band_number = 1  # Pas dit aan afhankelijk van welke band je wilt gebruiken
anomaly_band = dataset.read(band_number)
band_data = dataset.read(band_number)


# Bereken statistieken
mean_value = np.mean(band_data)
min_value = np.min(band_data)
max_value = np.max(band_data)



# Haal informatie op over het GeoTIFF-bestand
print("Breedte (aantal pixels):", dataset.width)
print("Hoogte (aantal pixels):", dataset.height)

# Haal georeferentie-informatie op
print("Geotransformatie:", dataset.transform)

# Haal de projectie op
print("Projectie:", dataset.crs)        

# Lees een rasterband
data = dataset.read(1)

# Bereken statistieken
mean_value = np.mean(anomaly_band)
min_value = np.min(anomaly_band)
max_value = np.max(anomaly_band)

# Toon statistieken
print(f"Gemiddelde waarde: {mean_value}")
print(f"Minimale waarde: {min_value}")
print(f"Maximale waarde: {max_value}")

# Toon het histogram
plt.hist(anomaly_band.flatten(), bins=50, edgecolor='black')
plt.title('Histogram van Anomalie-waarden')
plt.xlabel('Anomalie-waarde')
plt.ylabel('Frequentie')
plt.show()

# Lees twee rasterbanden als x- en y-coördinaten
x_band_number = 1  # Pas dit aan op basis van je gegevens
y_band_number = 2  # Pas dit aan op basis van je gegevens
x_data = dataset.read(x_band_number)
y_data = dataset.read(y_band_number)

# Controleer of de gekozen band bestaat in het GeoTIFF-bestand
if band_number <= dataset.count:
    band_data = dataset.read(band_number)
    
    
    # Sluit het GeoTIFF-bestand
    dataset.close()

    # Maak een lijngrafiek
plt.figure(figsize=(10, 6))
plt.plot(x_coordinates, band_data[0, :])  # We gebruiken hier de eerste rij van de band, pas dit aan op basis van je gegevens
plt.title('Lijngrafiek van GeoTIFF-gegevens')
plt.xlabel('X-coördinaat')
plt.ylabel('Waarde')
plt.show()
    
# Flatten de gegevens om ze te gebruiken in een scatterplot
x_flattened = x_data.flatten()
y_flattened = y_data.flatten()

# Maak een scatterplot
plt.figure(figsize=(10, 8))
plt.scatter(x_flattened, y_flattened, marker='.', alpha=0.5)
plt.title('Scatterplot van GeoTIFF-gegevens')
plt.xlabel('X-coördinaat')
plt.ylabel('Y-coördinaat')
plt.show()

# Visualiseer de vrij lucht anomalieën
plt.figure(figsize=(10, 10))
plt.imshow(anomaly_band, cmap='viridis')  # Pas de colormap aan afhankelijk van je voorkeur
plt.colorbar(label='Anomalie-waarden')
plt.title('Vrij Lucht Anomalieën')
plt.show()

# Visualiseer de vrij lucht anomalieën als een contourplot
plt.figure(figsize=(10, 10))
contour = plt.contour(anomaly_band, cmap='viridis', levels=20)  # Pas het aantal niveaus (levels) aan indien nodig
plt.colorbar(contour, label='Anomalie-waarden')
plt.title('Contourplot van Vrij Lucht Anomalieën')
plt.show()


