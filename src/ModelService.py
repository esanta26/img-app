import base64
import glob
import rasterio as rio
import numpy as np
import rasterio.plot as rio_plot
from keras.models import Model, load_model
import tensorflow as tf
from matplotlib import colors
import matplotlib.pyplot as plt
import io
import os


class ModelService:

    def ndmi(self, raster_array: np.array) -> np.array:
        # Las dimensiones aca son [bandas, altura, ancho]
        # Se toma con base en la modificacion con 12 bandas
        # Mayor informacion, por favor ver definicion del metodo: load_source_img
        # NDMI (Sentinel 2) = (B8 – B11) / (B8 + B11)
        swir_1_11 = raster_array[10, :, :]
        infrarrojo_8 = raster_array[7, :, :]

        # Evitar divisiones por cero e inestabilidades
        epsilon = 1e-8

        return (infrarrojo_8 - swir_1_11) / (infrarrojo_8 + swir_1_11)

    def ndvi(self, raster_array: np.array) -> np.array:
        # Las dimensiones aca son [bandas, altura, ancho]
        # Formula: NDVI (Sentinel 2) = (B8 – B4) / (B8 + B4)
        # Se toma con base en la modificacion con 12 bandas
        # Mayor informacion, por favor ver definicion del metodo: load_source_img
        red_channel = raster_array[3, :, :]
        infrared_channel = raster_array[7, :, :]

        # Evitar divisiones por cero e inestabilidades
        epsilon = 1e-8
        return ((infrared_channel - red_channel) / ((infrared_channel + red_channel) + epsilon))

    def bsi(self, raster_array: np.array) -> np.array:
        # Las dimensiones aca son [bandas, altura, ancho]
        # Se toma con base en la modificacion con 12 bandas
        # Mayor informacion, por favor ver definicion del metodo: load_source_img
        # BSI (Sentinel 2) = (B11 + B4) – (B8 + B2) / (B11 + B4) + (B8 + B2)

        swir_1_11 = raster_array[10, :, :]
        rojo_4 = raster_array[3, :, :]
        infrarrojo_8 = raster_array[7, :, :]
        azul_2 = raster_array[1, :, :]

        # Evitar divisiones por cero e inestabilidades
        epsilon = 1e-8

        # SWIR <-> ROJO
        swir_rojo = swir_1_11 + rojo_4
        # NIR <-> AZUL
        nir_azul = infrarrojo_8 + azul_2

        # Retornar
        return (swir_rojo - nir_azul) / (swir_rojo + nir_azul + epsilon)

    def load_channel_raster(self, path_raster_tiff: str, channel: int = 1) -> np.array:
        rsp = None
        with rio.open(path_raster_tiff, "r") as rf:
            rsp = rf.read(channel)
        return rsp

    def load_source_img(self, img_folder_path: str) -> np.array:
        # Obtener la referencia a las bandas
        # Un poco excesivo, pero por si las moscas, especificaremos el patron de cada imagen a mano
        aerosol_1 = glob.glob(f"{img_folder_path}/B01.*")[0]
        azul_2 = glob.glob(f"{img_folder_path}/B02.*")[0]
        verde_3 = glob.glob(f"{img_folder_path}/B03.*")[0]
        rojo_4 = glob.glob(f"{img_folder_path}/B04.*")[0]
        rojo_frontera_1_5 = glob.glob(f"{img_folder_path}/B05.*")[0]
        rojo_frontera_2_6 = glob.glob(f"{img_folder_path}/B06.*")[0]
        rojo_frontera_3_7 = glob.glob(f"{img_folder_path}/B07.*")[0]
        infrarrojo_8 = glob.glob(f"{img_folder_path}/B08.*")[0]
        infrarrojo_8A_9 = glob.glob(f"{img_folder_path}/B8A.*")[0]
        vapor_agua_9_10 = glob.glob(f"{img_folder_path}/B09.*")[0]
        onda_corta_1_11 = glob.glob(f"{img_folder_path}/B11.*")[0]
        onda_corta_2_12 = glob.glob(f"{img_folder_path}/B12.*")[0]

        # Cargar las cuatro bandas
        channels_list = [
            aerosol_1,
            azul_2,
            verde_3,
            rojo_4,
            rojo_frontera_1_5,
            rojo_frontera_2_6,
            rojo_frontera_3_7,
            infrarrojo_8,
            infrarrojo_8A_9,
            vapor_agua_9_10,
            onda_corta_1_11,
            onda_corta_2_12
        ]

        raster_bands = [
            self.load_channel_raster(r)
            for r in channels_list
        ]

        # Evitar los NaN, +Infinite, -Infinite
        raster_bands_clean = [
            np.nan_to_num(r, nan=0, posinf=0, neginf=0)
            for r in raster_bands
        ]

        # Normalizar los canales
        norm_data = lambda x: ((x - np.mean(x)) / np.std(x))

        # Aplicar
        norm_raster_bands = [
            norm_data(raster_band)
            for raster_band in raster_bands_clean
        ]

        # Construir el arreglo y retornar
        return np.array(norm_raster_bands)

    def load_img_index(self, source_path):
        # Cargar el raster inicial -> [canal, alto, ancho]
        source_raster = self.load_source_img(source_path)

        # Transformar la imagen
        ndvi_transformed = self.ndvi(source_raster)
        bsi_transformed = self.bsi(source_raster)
        ndmi_transformed = self.ndmi(source_raster)

        # Empaquetar y retornar
        index_array = np.array([
            ndvi_transformed,
            bsi_transformed,
            ndmi_transformed
        ])

        return rio_plot.reshape_as_image(index_array)

    def predict(self, img_path):
        print('hola')
        print(os.path.dirname(os.path.realpath(__file__)))
        model = load_model(os.path.abspath("models/model_index.h5"))
        print(model.layers[2].get_weights())
        example_load = self.load_img_index(os.path.abspath(f'data/ref_landcovernet_v1_source_{img_path}'))
        # print(example_load)
        ex_load_2 = np.array([example_load])
        predictions = model.predict(x=ex_load_2)
        print(predictions.shape)
        print(predictions)

        mask = np.argmax(predictions, axis=3)

        # Pintar
        # Crear un colormap personalizado para cada una de las etiquetas
        # según el documento
        cmap = colors.ListedColormap([
            "#0000ff",
            "#888888",
            "#d1a46d",
            "#f5f5ff",
            "#d64c2b",
            "#186818",
            "#00ff00"
        ])
        meaning = [
            "Agua",
            "Suelo desnudo artificial",
            "Suelo desnudo natural",
            "Nieve",
            "Bosque (Woody soil)",
            "Terreno cultivado",
            "Vegetación (semi) natural"
        ]
        boundaries = list(range(1, 8))
        norm = colors.BoundaryNorm(boundaries, cmap.N)

        fig, ax = plt.subplots()
        img = ax.imshow(mask[0], cmap=cmap, norm=norm)
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_ticks(boundaries)
        cbar.set_ticklabels(meaning)
        cbar.config_axis
        my_stringIObytes = io.BytesIO()
        plt.savefig(my_stringIObytes, format='jpg')
        my_stringIObytes.seek(0)
        my_base64_jpgData = base64.b64encode(my_stringIObytes.read())
        print(my_base64_jpgData)
        return my_base64_jpgData

# if __name__ == '__main__':
#     model_service = ModelService()
#     prediction = model_service.predict('37NGE_03_20180121')
#     unique, counts = np.unique(prediction[0], return_counts=True)
#     print(prediction)
