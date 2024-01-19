import streamlit as st
import os
from streamlit_option_menu import option_menu
import imgkit
import pandas as pd
import rasterio
import numpy as np
import matplotlib.pyplot as plt

#laad GEOTIFF
geo_tiff_path = "/Users/charell/Downloads/FAA_UTM18N_NAD83.tif"
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

# Toon het histogram in een aparte figuur
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(anomaly_band.flatten(), bins=50, edgecolor='black')
ax.set_title('Histogram van Anomalie-waarden')
ax.set_xlabel('Anomalie-waarde')
ax.set_ylabel('Frequentie')

# Load the CSV file into a DataFrame
csv_file_path = '/Users/charell/Downloads/Kenniskluis eindmodules/Gebied_07 - Copy/Vondsten/07_vondsten_lijst.csv'
df = pd.read_csv(csv_file_path, delimiter=';')


# Verkrijg de lijst met bestanden in de map met afbeeldingen
image_folder = '/Users/charell/Downloads/Kenniskluis eindmodules/trainingdata_controlemodel/Good'
image_files = os.listdir(image_folder)

# Maximum aantal afbeeldingen dat op pagina 2 wordt weergegeven
max_images_expander = 4


def load_html_file(file_path):
    """
    Laad de inhoud van een HTML-bestand en geef het terug als een string.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content

def convert_html_to_image(html_content, output_path):
    """
    Zet HTML-inhoud om naar een afbeelding en sla het op het opgegeven pad op.
    """
    imgkit.from_string(html_content, output_path)

def page_one():
    with st.container():
        left_column, middle_column, right_column = st.columns(3)
        # Voeg hier de inhoud van pagina 1 toe
        image_path = "/Users/charell/Desktop/Minor Big Data & Design/Logo startup.png"
        image_width = 300

        with middle_column:
            st.image(image_path, use_column_width=False, width=image_width)
        
            # Voeg een uploadknop toe met een aangepaste tekst en accepteert alleen bepaalde bestandstypen
            uploaded_file = st.file_uploader("Upload een bestand", type=["csv", "xlsx"])
    
    # Initialisatie van st.session_state
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Home"
        
        # Een knop toevoegen om de sessie te initialiseren
        st.button("Init Session State")

    
    # Controleer of er een bestand is geüpload
    if uploaded_file is not None:
        st.success("Bestand succesvol geüpload!")
        st.write("Inhoud van het bestand:")
        st.write(uploaded_file)

        # Als bestand is geüpload, ga naar pagina 2
        st.session_state.selected_page = "Jouw project"
        
        # Lees het bestand in een DataFrame
        df = pd.read_csv(uploaded_file, delimiter=';')
        # Toon de kolommen in een expander
        with st.sidebar:
            with st.expander("Variabelen"):
                st.table(df.columns)

    

def page_two():
    with st.container():
        st.title("Jouw Project")
        st.write("---")
        left_column, right_column = st.columns(2)
    with left_column:
        # Voeg hier de inhoud van pagina 2 toe  
        image_path3 = "/Users/charell/Desktop/Minor Big Data & Design/Vakje.png"
        image_width3 = 460
        
        # Visualiseer de vrij lucht anomalieën als een contourplot
        fig2, ax = plt.subplots(figsize=(10, 10))
        contour = ax.contour(anomaly_band, cmap='viridis', levels=20)  # Pas het aantal niveaus (levels) aan indien nodig
        fig.colorbar(contour, ax=ax, label='Anomalie-waarden')  # Voeg de kleurenbalk toe aan de figuur
        ax.set_title('Contourplot van Vrij Lucht Anomalieën')
        
        #Histogram GEOTIFF
        st.pyplot(fig)
        st.image(image_path3, use_column_width=False, width=image_width3)
        #Contourplot GEOTIFF
    
    with right_column:
        st.pyplot(fig2)
        image_path2 = "/Users/charell/Downloads/Kenniskluis eindmodules/trainingdata_controlemodel/Good/Figure 2024-01-15 134547 (61).png"
        st.image(image_path2, use_column_width=True)
  
    
    with st.sidebar:
        with st.expander("Visualisaties"):
            # Loop over de eerste vier afbeeldingen
            for i in range(min(max_images_expander, len(image_files))):
                # Controleer of het bestand een afbeelding is (bijv. png, jpg, jpeg)
                if image_files[i].lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(image_folder, image_files[i])
                    
                    # Toon de afbeelding met optioneel bijschrift
                    st.image(image_path, caption=f'Bijschrift voor {image_files[i]}', use_column_width=True)

def page_three():
    # Voeg hier de inhoud van pagina 3 toe
    st.title("Jouw Data Overzicht")
    st.write("---")
    # Vervang 'FILE_PATH_HIER' door het daadwerkelijke pad naar je HTML-bestand
    html_file_path = '/Users/charell/Downloads/Vondsten 07 Profiling Report.html'
    # Laad de inhoud van het HTML-bestand
    html_content = load_html_file(html_file_path)
    # Zet HTML-inhoud om naar een afbeelding
    image_path = 'html_image.png'
    convert_html_to_image(html_content, image_path)
    # Toon de afbeelding in Streamlit binnen een uitklapbare sectie
    # with st.expander("Toon afbeelding"):
       # st.image(image_path)
    
    st.components.v1.html(html_content, height=800, scrolling=True)
    
    
def page_four():
    st.title("Community")
    st.write("---")
    #afbeelding
    image_path4 = "/Users/charell/Desktop/Scherm­afbeelding 2024-01-19 om 03.49.40.png"
    image_width4 = 900
    st.image(image_path4, use_column_width=False, width=image_width4)
    
    
    
    
    

def main():
    # Voeg stijldefinitie toe voor de sidebar om deze naar links uit te lijnen
    st.set_page_config(layout="wide")
    
    st.markdown("""
        <style>
            .sidebar .sidebar-content {
                width: 300px;
                position: fixed;
                top: 1;
                left: 1;
                bottom: 1;
                overflow-y: auto;
            }
        </style>
    """, unsafe_allow_html=True)
    
    
    # Verberg de zijbalk
    hide_menu_style = """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    

    # Verkrijg de geselecteerde pagina
    with st.sidebar:
        selected_page = option_menu(
            menu_title="Menu",
            options=["Home", "Jouw project", "Data overzicht", "Community"],
            icons=["house", "book", "clipboard-data", "people"],
            menu_icon="cast",
            orientation="vertical"        
        )

    if selected_page == "Home":
        page_one()
    elif selected_page == "Jouw project":
        page_two()
    elif selected_page == "Data overzicht":
        page_three()
    elif selected_page == "Community":
        page_four()

if __name__ == "__main__":
    main()
