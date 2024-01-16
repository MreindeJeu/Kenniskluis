import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import squarify
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from ydata_profiling import ProfileReport


# Load the CSV file into a DataFrame
csv_file_path = '/Users/BerenddeJeu/Desktop/School Shizzle/Gebied_07/Vondsten/07_vondsten_lijst.csv'
df = pd.read_csv(csv_file_path, delimiter=';')

# Genereer een profile rapport voor de csv file
profile = ProfileReport(df, title="Vondsten 07 Profiling Report")
profile.to_file("Vondsten 07 Profiling Report.html")

# Mapping of archaeological periods to date ranges (update as needed)
archaeological_periods_mapping = {
    'paleolithicum': 'everything until 8800 BC',
    'mesolithicum': '8800 BC - 5300 BC',
    'neolithicum': '5300 BC - 2000 BC',
    'vroege bronstijd': '2000 BC - 1800 BC',
    'midden bronstijd': '1800 BC - 1100 BC',
    'late brondstijd': '1100 BC - 800 BC',
    'vroege ijzertijd': '800 BC - 500 BC',
    'midden ijzertijd': '500 BC - 250 BC',
    'late ijzertijd': '250 BC - 12 BC',
    'vroeg romeinsetijd': '12 BC - 70 AD',
    'midden romeinsetijd': '70 AD - 270 AD',
    'laat romeinsetijd': '270 AD - 450 AD',
    'vroege middeleeuwen': '450 AD - 1050 AD',
    'late middeleeuwen': '1050 AD - 1500 AD',
    'vroege nieuwetijd': '1500 AD - 1650 AD',
    'midden nieuwetijd': '1650 AD - 1850 AD',
    'late nieuwetijd': '1850 AD - 1945 AD',   
}

# Function to convert archaeological periods to date ranges
def convert_to_date_range(period):
    return archaeological_periods_mapping.get(period.lower(), period)

# Function to identify the categories of data
def analyze_data(data):
    categorical_cols = []
    numeric_cols = []
    cat_num_cols = []
    map_cols = []
    network_cols = []
    time_series_cols = []

    for col in data.columns:
        if 'datering' in col.lower() or 'date' in col.lower() or 'fase' in col.lower():
            time_series_cols.append(col)
            # Convert archaeological periods to date ranges
            data[col] = data[col].apply(convert_to_date_range)
        elif data[col].dtype == 'O':  # 'O' represents object (categorical) data type
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(data[col]):
            numeric_cols.append(col)

    return {
        'Categorical Columns': categorical_cols,
        'Numeric Columns': numeric_cols,
        'Categorical and Numeric Columns': cat_num_cols,
        'Map Columns': map_cols,
        'Network Columns': network_cols,
        'Time Series Columns': time_series_cols
    }

# Analyze the data
data_categories = analyze_data(df)

# Load the image classification model
model_path = "/Users/BerenddeJeu/Desktop/untitled folder/Kenniskluis eindmodules/test1/model.savedmodel"
model = load_model(model_path, compile=False)

# Load the labels
labels_path = "/Users/BerenddeJeu/Desktop/untitled folder/Kenniskluis eindmodules/test1/labels.txt"
class_names = open(labels_path, "r").readlines()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Function to classify the image
def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name[2:], confidence_score

# Function to save and display visualizations for numeric columns
def save_and_display_numeric_visualizations(data, numeric_cols):
    for i, col in enumerate(numeric_cols):
        if len(numeric_cols) == 1:
            
            #histogram
            x_col = numeric_cols[0]
            plt.figure(figsize=(12, 6))
            sns.histplot(data[x_col], kde=True, bins=20)
            plt.title(f'Histogram for {x_col}')
        
            #density plot
            plt.figure(figsize=(12, 6))
            sns.kdeplot(data[x_col], fill=True)
            plt.title(f'Density Plot for {x_col}')
  
        elif len(numeric_cols) == 2:
        
            #Violin Plot
            plt.figure(figsize=(12, 6))
            sns.violinplot(data=data[numeric_cols], inner="points", palette='viridis')
            plt.title(f'Violinplot for {", ".join(numeric_cols)}')
        
            # Scatterplot with marginal points
            plt.figure(figsize=(12, 6))
            sns.scatterplot(x=numeric_cols[0], y=numeric_cols[1], data=data, hue=data[numeric_cols[0]], palette='viridis', legend='full', alpha=0.7)
       
            #histogram
            sns.histplot(data[numeric_cols], kde=True, bins=20, element="step", fill=False, common_norm=False)
            plt.title(f'Scatterplot with Marginal Points for {", ".join(numeric_cols)}')
  
        elif len(numeric_cols) >= 3:
       
            # Violinplot
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 2, 1)
            sns.violinplot(data=data[numeric_cols], inner="points", palette='viridis')
            plt.title(f'Violinplot for {", ".join(numeric_cols)}')
        
            # Boxplot
            plt.subplot(2, 2, 2)
            sns.boxplot(data=data[numeric_cols], palette='viridis')
            plt.title(f'Boxplot for {", ".join(numeric_cols)}')
            
            # Stacked Area Plot
            plt.subplot(2, 2, 3)
            sns.lineplot(data=data[numeric_cols].cumsum(axis=1))
            plt.title(f'Stacked Area Plot for {", ".join(numeric_cols)}')
            
            # Streamgraph
            plt.subplot(2, 2, 4)
            sns.lineplot(data=data[numeric_cols], dashes=False)
            plt.title(f'Streamgraph for {", ".join(numeric_cols)}')
            
        plt.show()
     
        # Save the visualization as an image
        image_path = f'numeric_visualization_{i}.png'
        plt.savefig(image_path)
        plt.close()

        # Classify the saved image
        class_name, confidence_score = classify_image(image_path)
        print(f"Image Classification Result for {col}: Class: {class_name}, Confidence Score: {confidence_score}")

# Function to save and display visualizations for categorical columns
def save_and_display_categoric_visualizations(data, categorical_cols):
    for i, col in enumerate(categorical_cols):
        values = data[col].value_counts()
        if len(values) == 1:
            
            # Barplot
            plt.figure(figsize=(8, 6))
            sns.barplot(x=values.index, y=values)
            plt.title(f'Barplot for {col}')
            
            # Save the visualization as an image
            image_path = f'categoric_visualization_{i}.png'
            plt.savefig(image_path)
            plt.close()

            # Classify the saved image
            class_name, confidence_score = classify_image(image_path)
            print(f"Image Classification Result for {col}: Class: {class_name}, Confidence Score: {confidence_score}")

        elif len(values) == 2:
          
            # Lollipop plot
            plt.figure(figsize=(8, 6))
            sns.barplot(x=values.index, y=values)
            sns.scatterplot(x=values.index, y=values, color='red', marker='o', s=100)
            plt.title(f'Lollipop Plot for {col}')
            
            # Save the visualization as an image
            image_path = f'categoric_visualization_{i}.png'
            plt.savefig(image_path)
            plt.close()

            # Classify the saved image
            class_name, confidence_score = classify_image(image_path)
            print(f"Image Classification Result for {col}: Class: {class_name}, Confidence Score: {confidence_score}")

        elif len(values) > 2:
            # Wordcloud
            plt.figure(figsize=(8, 8))
            wordcloud = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(values)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Wordcloud for {col}')
            
            # Doughnut plot
            plt.figure(figsize=(8, 8))
            plt.pie(values, labels=values.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis', len(values)))
            centre_circle = plt.Circle((0,0),0.70,fc='white')
            fig = plt.gcf()
            fig.gca().add_artist(centre_circle)
            plt.title(f'Doughnut Plot for {col}')
            
            # Pie chart
            plt.figure(figsize=(8, 8))
            plt.pie(values, labels=values.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis', len(values)))
            plt.title(f'Pie Chart for {col}')
            
            # Treemap
            plt.figure(figsize=(10, 8))
            squarify.plot(sizes=values, label=values.index, color=sns.color_palette('viridis', len(values)))
            plt.axis('off')
            plt.title(f'Treemap for {col}')
            
            # Circular Packing
            plt.figure(figsize=(10, 8))
            sns.heatmap([values], annot=True, fmt="d", cmap='viridis', cbar=False)
            plt.title(f'Circular Packing for {col}')
            
        plt.show()
        
        # Save the visualization as an image
        image_path = f'categoric_visualization_{i}.png'
        plt.savefig(image_path)
        plt.close()

        # Classify the saved image
        class_name, confidence_score = classify_image(image_path)
        print(f"Image Classification Result for {col}: Class: {class_name}, Confidence Score: {confidence_score}")

# Save and display visualizations for categorical columns
save_and_display_categoric_visualizations(df, data_categories['Categorical Columns'])

# Save and display visualizations for numeric columns
save_and_display_numeric_visualizations(df, data_categories['Numeric Columns'])





import rasterio

# Open een GeoTIFF-bestand
geo_tiff_path = "/Users/BerenddeJeu/Desktop/untitled folder/FAA_UTM18N_NAD83.tif"
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
