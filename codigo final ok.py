# -*- coding: utf-8 -*-
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import geometry_mask, geometry_window
from scipy.ndimage import label
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import warnings
import datetime
import traceback
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (confusion_matrix, accuracy_score, cohen_kappa_score,
                           classification_report, balanced_accuracy_score,
                           jaccard_score, precision_score, recall_score, f1_score)
from sklearn.ensemble import RandomForestClassifier

try:
    import pytz
except ImportError:
    pytz = None

# --- CONSTANTES ---

# Método de Validação ('K_FOLD' ou 'HOLD_OUT' ou 'ORIGINAL')
VALIDATION_METHOD = 'K_FOLD'
# Parâmetros de Validação
HOLD_OUT_TEST_SIZE = 0.3
K_FOLD_N_SPLITS = 5

# Caminhos das Bandas Sentinel para cada ano
SENTINEL_BANDS_2020 = {
    'B02': r"C:\Users\natan\OneDrive\Desktop\mestrado\2020\R10m\T24KVG_20201007T125311_B02_10m.jp2",
    'B03': r"C:\Users\natan\OneDrive\Desktop\mestrado\2020\R10m\T24KVG_20201007T125311_B03_10m.jp2",
    'B04': r"C:\Users\natan\OneDrive\Desktop\mestrado\2020\R10m\T24KVG_20201007T125311_B04_10m.jp2",
    'B08': r"C:\Users\natan\OneDrive\Desktop\mestrado\2020\R10m\T24KVG_20201007T125311_B08_10m.jp2"
}
SENTINEL_BANDS_2023 = {
    'B02': r"C:\Users\natan\OneDrive\Desktop\mestrado\2023\R10m\T24KVG_20230927T125309_B02_10m.jp2",
    'B03': r"C:\Users\natan\OneDrive\Desktop\mestrado\2023\R10m\T24KVG_20230927T125309_B03_10m.jp2",
    'B04': r"C:\Users\natan\OneDrive\Desktop\mestrado\2023\R10m\T24KVG_20230927T125309_B04_10m.jp2",
    'B08': r"C:\Users\natan\OneDrive\Desktop\mestrado\2023\R10m\T24KVG_20230927T125309_B08_10m.jp2"
}

# Caminho para o Shapefile da Área de Estudo
STUDY_AREA_SHP = r"C:\Users\natan\OneDrive\Desktop\mestrado\limite\limite.shp"
# Diretório para Salvar Resultados do RF (Mapas, Shapes, Sumários por ano)
OUTPUT_DIR_RF = r'C:\Users\natan\OneDrive\Desktop\mestrado\resultados_rf_only_filtrado'
# Caminho para o Shapefile de Ground Truth (ÚNICO para ambos os anos)
GROUND_TRUTH_SHP = r'C:\Users\natan\OneDrive\Desktop\mestrado\resultados grouth\pontos2.shp'
# Diretório para Salvar Relatórios de Validação (Individuais e Comparativo)
OUTPUT_DIR_VALID_RF = r'C:\Users\natan\OneDrive\Desktop\mestrado\resultados_rf_only_filtrado\validacao'

# Parâmetros do Random Forest
RF_N_ESTIMATORS = 100
RF_RANDOM_STATE = 42
RF_CLASS_WEIGHT = 'balanced' # 'balanced' tenta dar mais peso à classe minoritária

# Parâmetros do Filtro de Tamanho 
APLICAR_FILTRO_TAMANHO_RF = True # True para aplicar, False para não aplicar
RF_GRUPO_MIN = 2  # Tamanho mínimo do grupo de pixels conectados (inclusive)
RF_GRUPO_MAX = 3 # Tamanho máximo do grupo de pixels conectados (inclusive)

# --- COLUNAS E CLASSES NO GROUND TRUTH ---
# !! IMPORTANTE: COLUNA_ANO_REF = None porque o shape GT não tem mais coluna de ano !!
COLUNA_ANO_REF = None   # Define como None para NÃO filtrar GT por ano
COLUNA_CLASSE_REF = 'NIVEL_III' # Nome da coluna com as classes (ex: 'UsoSolo', 'NIVEL_III')

# Definição das Classes
CLASSES_LABELS = [0, 1] # Labels numéricos (0 para Não-Alvo, 1 para Alvo)
CLASSE_ALVO_NOME = 'Sapucaia' # Nome da classe de interesse (como aparece na coluna COLUNA_CLASSE_REF)
CLASSE_ALVO_LABEL = 1 # Label numérico correspondente à classe alvo

# Valores NoData
INTERNAL_NODATA = -9999 # Valor usado internamente para NoData durante leitura
OUTPUT_NODATA = 255     # Valor de NoData a ser gravado nos TIFs de saída (UInt8)


# --- FUNÇÕES AUXILIARES ---

def identificar_grupos_conectados_mask(input_mask, min_size=2, max_size=1000000):
    """Identifica grupos conectados em máscara booleana. Retorna máscara final e coords."""
    input_mask = input_mask.astype(bool)
    if not np.any(input_mask):
        return np.zeros_like(input_mask, dtype=bool), []

    structure = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=bool) # Conectividade 8
    array_rotulado, num_total_componentes = label(input_mask, structure=structure)
    grupos_validos_coords_list = []
    final_mask = np.zeros_like(input_mask, dtype=bool)

    if num_total_componentes > 0:
        component_ids, tamanhos = np.unique(array_rotulado, return_counts=True)
        # Ignora o background (label 0)
        component_ids = component_ids[1:]
        tamanhos = tamanhos[1:]

        ids_tamanho_certo = component_ids[(tamanhos >= min_size) & (tamanhos <= max_size)]

        if len(ids_tamanho_certo) > 0:
            final_mask = np.isin(array_rotulado, ids_tamanho_certo)
            print(f"Filtro de tamanho encontrou {len(ids_tamanho_certo)} grupos entre {min_size} e {max_size} pixels.")
            # Extrai as coordenadas (linha, coluna) dos pixels que passaram no filtro
            coords_final_rc = np.argwhere(final_mask)
            if coords_final_rc.size > 0:
                 # Retorna como uma lista contendo UMA lista de coordenadas [[r1,c1], [r2,c2], ...]
                 grupos_validos_coords_list = [coords_final_rc.tolist()]
    else:
        print("Nenhum grupo encontrado na máscara inicial para aplicar filtro.")

    return final_mask, grupos_validos_coords_list

def exportar_shapes_finais(transform_geo, crs_geo, todos_grupos_coords, output_dir_rf, sufixo_ano_metodo):
    """Exporta shapefile com pontos dos pixels detectados para um ano específico."""
    print(f"Exportando shapefile para {sufixo_ano_metodo}...")
    output_shapefile = os.path.join(output_dir_rf, f'shapes_alvo_{sufixo_ano_metodo}.shp')

    if not todos_grupos_coords:
        print(f"Aviso: Nenhuma coordenada fornecida para {sufixo_ano_metodo}. Shapefile não gerado.")
        return None

    lista_shapes = []
    pontos_processados = 0
    # Espera uma lista contendo UMA lista de coordenadas [[r1,c1], [r2,c2], ...]
    if todos_grupos_coords and isinstance(todos_grupos_coords[0], list):
         coords_flat_list = todos_grupos_coords[0]
         id_grupo_unico = f"{sufixo_ano_metodo}_filtered" # ID genérico para os pontos
         ano_shape = sufixo_ano_metodo.split('_')[0] # Extrai ano do sufixo (ex: '2020_rf' -> '2020')
         metodo_shape = sufixo_ano_metodo.split('_')[-1] # Extrai método (ex: '2020_rf' -> 'rf')

         for coord_pair in coords_flat_list:
              try:
                  # Garante que linha e coluna são inteiros
                  lin_int = int(round(coord_pair[0]))
                  col_int = int(round(coord_pair[1]))
                  # Converte linha, coluna para coordenadas geográficas (centro do pixel)
                  x, y = rasterio.transform.xy(transform_geo, lin_int, col_int, offset="center")
                  lista_shapes.append({
                      "geometry": Point(x, y),
                      "metodo": metodo_shape,
                      "ano": ano_shape, # Adiciona ano como atributo
                      "id_grupo": id_grupo_unico, # ID para agrupar visualmente
                      "lin_img": lin_int,
                      "col_img": col_int
                  })
                  pontos_processados += 1
              except IndexError:
                  print(f"Erro {sufixo_ano_metodo}: Formato inesperado de coordenadas: {coord_pair}. Pulando.")
                  continue
              except Exception as e:
                  print(f"Erro {sufixo_ano_metodo}: Conversão de coordenadas falhou para {coord_pair}. Erro: {e}")
                  continue
    else:
         print(f"Aviso: Estrutura de coordenadas inesperada para {sufixo_ano_metodo}. Shapefile não gerado.")
         return None


    if not lista_shapes:
        print(f"Aviso: Nenhuma geometria Point criada para {sufixo_ano_metodo}. Shapefile não gerado.")
        return None
    try:
        valid_crs = crs_geo if crs_geo else None
        if valid_crs is None:
            warnings.warn(f"CRS não definido para {sufixo_ano_metodo}. Shapefile será gerado sem CRS.")

        # Cria o GeoDataFrame
        gdf = gpd.GeoDataFrame(lista_shapes, crs=valid_crs)
        # Tenta salvar com UTF-8, se falhar, tenta com latin-1
        try:
            gdf.to_file(output_shapefile, driver='ESRI Shapefile', encoding='utf-8')
        except Exception as e_enc:
            warnings.warn(f"Falha ao salvar shape {sufixo_ano_metodo} com UTF-8 ({e_enc}). Tentando latin-1.")
            try:
                gdf.to_file(output_shapefile, driver='ESRI Shapefile', encoding='latin-1')
            except Exception as e_enc2:
                print(f"Erro crítico ao salvar shape {sufixo_ano_metodo} com latin-1: {e_enc2}")
                return None
        print(f"Shapefile {sufixo_ano_metodo} ({pontos_processados} pontos) exportado: {output_shapefile}")
        return output_shapefile
    except Exception as e:
        print(f"Erro ao criar/salvar GeoDataFrame {sufixo_ano_metodo}: {e}")
        traceback.print_exc()
        return None

def exportar_tif_classificacao(mascara_final_numerica, out_profile, output_dir_rf, nome_sufixo_metodo):
    """Exporta máscara final de classificação (0 ou 1, nodata=OUTPUT_NODATA) como GeoTIFF."""
    output_tif = os.path.join(output_dir_rf, f'raster_classificacao_{nome_sufixo_metodo}.tif')
    if mascara_final_numerica is None:
        print(f"Aviso: Máscara numérica para {nome_sufixo_metodo} é None. TIF não gerado.")
        return None

    # Garante que a máscara está no formato UInt8
    if mascara_final_numerica.dtype != np.uint8:
        try:
            mascara_uint8 = mascara_final_numerica.astype(np.uint8)
        except Exception as e:
            print(f"Erro ao converter máscara {nome_sufixo_metodo} para uint8: {e}")
            return None
    else:
        mascara_uint8 = mascara_final_numerica

    pixels_alvo = np.sum(mascara_uint8 == CLASSE_ALVO_LABEL)
    print(f"Exportando TIF de classificação para {nome_sufixo_metodo} ({pixels_alvo} pixels alvo)...")

    try:
        # Prepara o perfil final para o TIF de saída (1 banda, UInt8, compressão)
        profile_final = out_profile.copy()
        profile_final.update(dtype=rasterio.uint8, count=1, nodata=OUTPUT_NODATA, compress='lzw', driver='GTiff')
        # Remove parâmetros que podem causar conflito com LZW
        profile_final.pop('blockxsize', None); profile_final.pop('blockysize', None)
        profile_final.pop('tiled', None)

        # Verifica informações essenciais do perfil
        if profile_final.get('crs') is None:
            warnings.warn(f"Perfil TIF {nome_sufixo_metodo} não tem CRS.")
        if profile_final.get('transform') is None:
            raise ValueError(f"Perfil TIF {nome_sufixo_metodo} não tem Transform (afinidade).")
        # Garante que as dimensões no perfil correspondem à máscara
        if profile_final['height'] != mascara_uint8.shape[0] or profile_final['width'] != mascara_uint8.shape[1]:
             warnings.warn(f"Dimensões no perfil TIF não correspondem à máscara {nome_sufixo_metodo}. Ajustando perfil.")
             profile_final['height'] = mascara_uint8.shape[0]
             profile_final['width'] = mascara_uint8.shape[1]

        # Escreve o TIF
        with rasterio.open(output_tif, 'w', **profile_final) as dst:
            dst.write(mascara_uint8, 1)
        print(f"TIF de classificação {nome_sufixo_metodo} exportado: {output_tif}")
        return output_tif
    except Exception as e:
        print(f"Erro crítico ao exportar TIF {nome_sufixo_metodo}: {e}")
        traceback.print_exc()
        return None

# --- FUNÇÃO DE PROCESSAMENTO RANDOM FOREST POR ANO ---
def processar_ano_rf(ano_str, bandas_paths, shape_limite_path, shp_ground_truth_path, output_dir_rf, col_classe_gt):
    """
    Processa por RF para um ano específico, usando TODOS os pontos GT.
    Retorna: (metricas_validação, path_tif_final, path_shp_final, gdf_gt_usado, array_raster_final, profile_raster_final)
    """
    print(f"\n--- Iniciando Processamento e Validação ({VALIDATION_METHOD}) para {ano_str} ---")
    metodo_id = f"{ano_str}_rf" # ID para arquivos deste ano (ex: 2020_rf)
    sumario_path = os.path.join(output_dir_rf, f'sumario_{metodo_id}.txt')

    # Inicializa variáveis de retorno e internas
    raster_classificado_path_final = None; shapes_path_final = None; out_profile = None;
    X_all = []; y_all = [];
    pixels_in_geom = 0; total_pixels_preditos = 0;
    rf_classification_raster_final_map = None # Array NumPy do raster final
    out_profile_final_map = None # Perfil do raster final
    pixels_alvo_inicial_rf = 0; pixels_alvo_final_rf = 0;
    metricas_validacao = None # Dicionário para guardar métricas
    gdf_gt_valid = None # GDF com os pontos GT realmente usados

    try:
        # 1. Leitura e Preparação dos Dados Geoespaciais
        print("Lendo shapefile de limite...")
        limite_gdf = gpd.read_file(shape_limite_path)
        if limite_gdf.empty: raise ValueError("Shapefile de limite vazio.")

        # Abre raster de referência para obter CRS e perfil base
        try:
            ref_raster_path = bandas_paths['B04'] # Usa B04 como referência
            if not os.path.exists(ref_raster_path):
                raise FileNotFoundError(f"Arquivo da Banda B04 não encontrado para {ano_str}: {ref_raster_path}")
            print(f"Abrindo raster de referência: {ref_raster_path}")
            with rasterio.open(ref_raster_path) as src_ref:
                raster_crs = src_ref.crs
                raster_profile = src_ref.profile
                raster_width = src_ref.width
                raster_height = src_ref.height
            print(f"CRS do Raster: {raster_crs}")
        except FileNotFoundError as fnf_err: raise fnf_err
        except rasterio.RasterioIOError as e: raise rasterio.RasterioIOError(f"Erro ao abrir raster ref {ref_raster_path}: {e}")
        except Exception as e_ref: raise ValueError(f"Erro ao processar raster de referência: {e_ref}")

        # Reprojeta limite se necessário
        if limite_gdf.crs != raster_crs:
            print(f"Reprojetando limite de {limite_gdf.crs} para {raster_crs}...")
            limite_gdf = limite_gdf.to_crs(raster_crs)
        limite_geom = limite_gdf.geometry.tolist()

        # Abre todas as bandas necessárias para o ano atual com 'with' para garantir fechamento
        sources = {}
        try:
            print("Abrindo arquivos das bandas Sentinel...")
            for band, path in bandas_paths.items():
                 if not os.path.exists(path): raise FileNotFoundError(f"Arquivo da banda {band} não encontrado para {ano_str}: {path}")
                 sources[band] = rasterio.open(path)

            src_b04 = sources['B04'] # Referência para janela e perfil

            # Calcula a janela com base na geometria de limite
            print("Calculando janela de leitura...")
            out_window = geometry_window(src_b04, limite_geom, pad_x=1, pad_y=1)
            out_window = out_window.intersection(Window(0, 0, raster_width, raster_height)) # Garante que a janela não exceda os limites do raster
            if not (out_window.width > 0 and out_window.height > 0):
                raise ValueError(f"Área de estudo (limite) parece estar fora da extensão do raster para {ano_str}.")
            out_transform = rasterio.windows.transform(out_window, src_b04.transform)

            # Define o perfil de saída inicial (para leitura e referência)
            out_profile = raster_profile.copy()
            out_profile.update(height=out_window.height, width=out_window.width, transform=out_transform, nodata=INTERNAL_NODATA, driver='GTiff', count=len(sources))
            out_profile.pop('blockxsize', None); out_profile.pop('blockysize', None); out_profile.pop('tiled', None) # Remove tiling info

            # Lê os dados das bandas dentro da janela
            print(f"Lendo dados das bandas {ano_str} ({int(out_window.width)}x{int(out_window.height)} pixels)...")
            band_data = {}
            for band, src in sources.items():
                 print(f"  Lendo banda {band}...")
                 band_data[band] = src.read(1, window=out_window, boundless=True, fill_value=INTERNAL_NODATA)

            # Dados das bandas individuais
            b02_data = band_data['B02']
            b03_data = band_data['B03']
            b04_data = band_data['B04']
            b08_data = band_data['B08']

        finally:
            # Garante que todos os arquivos raster abertos sejam fechados
            for src in sources.values():
                if src and not src.closed:
                    src.close()
            print("Arquivos das bandas Sentinel fechados.")

        # Cria máscara da geometria e máscara de dados válidos
        print("Criando máscaras...")
        mask_geom = geometry_mask(limite_geom, out_shape=b04_data.shape, transform=out_transform, invert=True, all_touched=True)
        pixels_in_geom = int(np.sum(mask_geom))
        if pixels_in_geom == 0:
            raise StopIteration(f"Nenhum pixel do raster coincide com a área de estudo (limite) para {ano_str}.")
        print(f"Pixels dentro da geometria: {pixels_in_geom}")

        # Máscara de pixels válidos (dentro da geometria E com dados válidos em TODAS as bandas)
        valid_data_mask_pred = mask_geom & \
                               (b02_data != INTERNAL_NODATA) & (b03_data != INTERNAL_NODATA) & \
                               (b04_data != INTERNAL_NODATA) & (b08_data != INTERNAL_NODATA)
        pixels_validos = int(np.sum(valid_data_mask_pred))
        if pixels_validos == 0:
            raise StopIteration(f"Nenhum pixel com dados válidos em todas as bandas encontrado na área de estudo para {ano_str}.")
        print(f"Pixels com dados válidos na área: {pixels_validos}")


        # 2. Preparar Dados de Treinamento/Validação (Ground Truth)
        print(f"\nPreparando TODOS os dados GT do shapefile: {os.path.basename(shp_ground_truth_path)} para o ano {ano_str}...")
        gdf_gt = gpd.read_file(shp_ground_truth_path)
        if gdf_gt.empty: raise ValueError("Shapefile de Ground Truth vazio.")
        print(f"Total de pontos GT lidos: {len(gdf_gt)}. Usando todos os pontos válidos para {ano_str} (COLUNA_ANO_REF = None).")

        # Garante que a coluna de classe existe
        if col_classe_gt not in gdf_gt.columns:
            raise ValueError(f"Coluna classe GT '{col_classe_gt}' não encontrada no shapefile.")

        # Reprojeta GT se necessário
        if gdf_gt.crs != raster_crs:
            print(f"Reprojetando GT de {gdf_gt.crs} para {raster_crs}...")
            gdf_gt = gdf_gt.to_crs(raster_crs)

        # Extrai coordenadas (x, y) e converte para (linha, coluna) do raster LIDO (na janela)
        coords_gt_xy = [(p.x, p.y) for p in gdf_gt.geometry]
        # rowcol retorna linha, coluna relativas à janela lida (out_transform)
        rows_gt, cols_gt = rasterio.transform.rowcol(out_transform, [c[0] for c in coords_gt_xy], [c[1] for c in coords_gt_xy])

        # Filtra pontos GT que caem fora da janela lida ou têm dados inválidos nas bandas
        valid_indices_gt = [] # Índices do GDF original que são válidos
        X_all_list = []       # Lista para guardar os valores das bandas [B2, B3, B4, B8]
        gdf_indices_map = {}  # Mapeia índice da lista filtrada -> índice do GDF original

        print(f"Extraindo valores das bandas de {ano_str} para {len(rows_gt)} pontos GT...")
        count_outside = 0
        count_nodata = 0
        for i, (r, c) in enumerate(zip(rows_gt, cols_gt)):
            # Verifica se a linha/coluna está DENTRO dos limites do array lido (0 a height-1, 0 a width-1)
            if 0 <= r < b04_data.shape[0] and 0 <= c < b04_data.shape[1]:
                # Extrai valores das bandas para este ponto
                b02_val = b02_data[r, c]
                b03_val = b03_data[r, c]
                b04_val = b04_data[r, c]
                b08_val = b08_data[r, c]
                # Verifica se TODOS os valores são diferentes do NoData interno
                if all(val != INTERNAL_NODATA for val in [b02_val, b03_val, b04_val, b08_val]):
                    current_valid_index = len(valid_indices_gt) # Índice na lista de válidos
                    valid_indices_gt.append(i) # Guarda índice original do GDF
                    X_all_list.append([b02_val, b03_val, b04_val, b08_val])
                    gdf_indices_map[current_valid_index] = i # Mapeia valid_index -> original_gdf_index
                else:
                    count_nodata += 1 # Ponto caiu em pixel com NoData em alguma banda
            else:
                count_outside += 1 # Ponto caiu fora da janela lida do raster

        if count_outside > 0: print(f"  {count_outside} pontos GT caíram fora da área do raster lida.")
        if count_nodata > 0: print(f"  {count_nodata} pontos GT caíram em pixels com NoData em alguma banda.")

        if not valid_indices_gt:
             raise ValueError(f"Nenhum ponto GT válido encontrado dentro da área lida e com dados válidos para {ano_str}.")

        # Converte a lista de features para array NumPy
        X_all = np.array(X_all_list) # Features (Bandas) dos pontos GT válidos

        # Cria GDF filtrado contendo apenas os pontos GT válidos e na ordem correta
        # Usa o mapeamento para garantir que a ordem do GDF corresponde a X_all e y_all
        gdf_gt_valid = gdf_gt.iloc[[gdf_indices_map[i] for i in range(len(valid_indices_gt))]].copy()

        # Mapeia classes de texto para numérico (0 ou 1) usando a coluna definida
        def normalize_text(text): return str(text).strip() if pd.notna(text) else ""
        classe_map = {CLASSE_ALVO_NOME: CLASSE_ALVO_LABEL} # Mapeia apenas a classe alvo
        # Aplica o mapeamento, qualquer outra classe vira 0 (Não-Alvo)
        gdf_gt_valid['Ref_Num'] = gdf_gt_valid[col_classe_gt].apply(normalize_text).map(classe_map).fillna(0).astype(int)
        y_all = gdf_gt_valid['Ref_Num'].values # Labels (Classes 0 ou 1)

        print(f"Total de amostras GT válidas para treino/validação de {ano_str}: {X_all.shape[0]}")
        print(f"  Distribuição das classes GT (0=NãoAlvo, 1=Alvo): {dict(zip(*np.unique(y_all, return_counts=True)))}")
        if len(np.unique(y_all)) < 2:
            warnings.warn(f"ATENÇÃO: Apenas uma classe GT presente ({np.unique(y_all)}) para {ano_str}. A validação e o treino podem ser limitados ou inválidos.")


        # 3. Treinamento e Validação Conforme o Método Escolhido
        rf_model_final = None # Modelo treinado com todos os dados para o mapa final

        if VALIDATION_METHOD == 'HOLD_OUT':
            print("\n--- Executando Validação HOLD-OUT ---")
            if X_all.shape[0] < 2 : raise ValueError("Não há dados suficientes para dividir em treino/teste.")
            # Stratify só funciona se houver pelo menos 2 amostras de cada classe (ou n_splits < n_samples_per_class)
            stratify_option = y_all if len(np.unique(y_all)) > 1 else None

            X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
                X_all, y_all, np.arange(X_all.shape[0]), # Inclui índices para rastrear pontos
                test_size=HOLD_OUT_TEST_SIZE,
                random_state=RF_RANDOM_STATE,
                stratify=stratify_option
            )
            print(f"Divisão Hold-Out: {len(y_train)} treino, {len(y_test)} teste.")
            print(f"  Distribuição Treino: {dict(zip(*np.unique(y_train, return_counts=True)))}")
            print(f"  Distribuição Teste:  {dict(zip(*np.unique(y_test, return_counts=True)))}")

            # Treina no conjunto de treino
            print("Treinando modelo RF (Hold-Out)...")
            rf_model_holdout = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE, class_weight=RF_CLASS_WEIGHT, n_jobs=-1)
            rf_model_holdout.fit(X_train, y_train)

            # Prediz no raster inteiro com o modelo Hold-Out
            print("Predizendo no raster completo (Modelo Hold-Out)...")
            rows_pred, cols_pred = np.where(valid_data_mask_pred)
            X_pred_map = np.vstack([b02_data[rows_pred, cols_pred], b03_data[rows_pred, cols_pred],
                                    b04_data[rows_pred, cols_pred], b08_data[rows_pred, cols_pred]]).T
            total_pixels_preditos = X_pred_map.shape[0]
            predictions_map = rf_model_holdout.predict(X_pred_map)

            # Cria raster inicial com predição (antes do filtro)
            rf_classification_raster_initial = np.full(b04_data.shape, OUTPUT_NODATA, dtype=np.uint8)
            rf_classification_raster_initial[valid_data_mask_pred] = 0 # Não-Alvo onde válido
            rf_classification_raster_initial[rows_pred, cols_pred] = predictions_map.astype(np.uint8)
            pixels_alvo_inicial_rf_holdout = int(np.sum(rf_classification_raster_initial == CLASSE_ALVO_LABEL))
            print(f"Pixels Alvo ANTES do filtro (Modelo Hold-Out): {pixels_alvo_inicial_rf_holdout}")

            # Aplica filtro de tamanho a este raster inicial do Hold-Out
            rf_classification_raster_filtered_holdout = rf_classification_raster_initial # Default se filtro não for aplicado
            if APLICAR_FILTRO_TAMANHO_RF and pixels_alvo_inicial_rf_holdout > 0:
                print(f"Aplicando filtro de tamanho ({RF_GRUPO_MIN}-{RF_GRUPO_MAX} pixels) ao mapa Hold-Out...")
                rf_alvo_mask_initial = (rf_classification_raster_initial == CLASSE_ALVO_LABEL)
                final_mask_bool_filtered, _ = identificar_grupos_conectados_mask( # Ignora coords aqui
                    rf_alvo_mask_initial, min_size=RF_GRUPO_MIN, max_size=RF_GRUPO_MAX
                )
                # Cria o raster filtrado
                rf_classification_raster_filtered_holdout = np.full(b04_data.shape, OUTPUT_NODATA, dtype=np.uint8)
                rf_classification_raster_filtered_holdout[valid_data_mask_pred] = 0 # Mantém Não-Alvo onde válido
                rf_classification_raster_filtered_holdout[final_mask_bool_filtered] = CLASSE_ALVO_LABEL # Aplica Alvo filtrado
                print(f"Pixels Alvo APÓS filtro (Modelo Hold-Out): {int(np.sum(final_mask_bool_filtered))}")
            elif not APLICAR_FILTRO_TAMANHO_RF:
                print("Filtro de tamanho NÃO aplicado.")
            else: # pixels_alvo_inicial_rf_holdout == 0
                 print("Filtro de tamanho não aplicado (nenhum pixel Alvo inicial no mapa Hold-Out).")

            # Extrai predições **do raster filtrado** APENAS para os pontos de teste
            print(f"Extraindo predições do mapa filtrado Hold-Out para os {len(indices_test)} pontos de teste...")
            gdf_test_points = gdf_gt_valid.iloc[indices_test] # Pega GDF dos pontos de teste
            coords_test_xy = [(p.x, p.y) for p in gdf_test_points.geometry]

            # Usa src fictício em memória para sample, pois o raster está em memória
            try:
                with rasterio.io.MemoryFile() as memfile:
                     with memfile.open(driver='GTiff', height=rf_classification_raster_filtered_holdout.shape[0],
                                       width=rf_classification_raster_filtered_holdout.shape[1], count=1,
                                       dtype=rf_classification_raster_filtered_holdout.dtype, crs=raster_crs,
                                       transform=out_transform, nodata=OUTPUT_NODATA) as dataset:
                         dataset.write(rf_classification_raster_filtered_holdout, 1)
                         # Sample retorna uma lista de arrays, pegamos o primeiro elemento de cada
                         y_pred_test_map = [val[0] for val in dataset.sample(coords_test_xy)]
            except Exception as e_sample:
                 print(f"Erro ao extrair valores do raster filtrado para validação Hold-Out: {e_sample}")
                 y_pred_test_map = [OUTPUT_NODATA] * len(coords_test_xy) # Assume falha para todos

            # Limpa predições inválidas (nodata) e calcula métricas
            y_true_test_final = []
            y_pred_test_final = []
            original_y_test_len = len(y_test)
            for yt, yp in zip(y_test, y_pred_test_map):
                # Checa se a predição é válida (não nodata E está nos labels definidos)
                if yp != OUTPUT_NODATA and yp in CLASSES_LABELS:
                    y_true_test_final.append(yt)
                    y_pred_test_final.append(yp)

            if len(y_true_test_final) < original_y_test_len:
                print(f"Aviso Hold-Out: {original_y_test_len - len(y_true_test_final)} pontos de teste removidos da validação por terem valor NoData/inválido no mapa final filtrado.")

            # Calcula métricas no conjunto de teste válido
            if not y_true_test_final:
                print("Erro Hold-Out: Nenhum ponto de teste válido restante para calcular métricas.")
                metricas_validacao = None
            else:
                print(f"Calculando métricas de validação Hold-Out em {len(y_true_test_final)} pontos válidos...")
                metricas_validacao = calcular_metricas(y_true_test_final, y_pred_test_final, CLASSES_LABELS)
                if metricas_validacao: # Adiciona info extra se métricas foram calculadas
                     metricas_validacao['metodo_validacao'] = f'Hold-Out ({HOLD_OUT_TEST_SIZE*100:.0f}% teste)'
                     metricas_validacao['pontos_teste'] = len(y_true_test_final)

            # Treina modelo final com TODOS os dados para o mapa final
            print("\nTreinando modelo RF final com TODOS os dados GT...")
            rf_model_final = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE, class_weight=RF_CLASS_WEIGHT, n_jobs=-1)
            rf_model_final.fit(X_all, y_all)


        elif VALIDATION_METHOD == 'K_FOLD':
            print(f"\n--- Executando Validação K-FOLD ({K_FOLD_N_SPLITS} folds) ---")
            if X_all.shape[0] < K_FOLD_N_SPLITS :
                raise ValueError(f"Não há dados suficientes ({X_all.shape[0]}) para {K_FOLD_N_SPLITS}-Fold CV.")
            # Stratify só funciona se houver pelo menos n_splits amostras de cada classe
            stratify_option = y_all if len(np.unique(y_all)) > 1 else None

            try:
                skf = StratifiedKFold(n_splits=K_FOLD_N_SPLITS, shuffle=True, random_state=RF_RANDOM_STATE)
                y_true_all_folds = []
                y_pred_all_folds = [] # Guarda predições BRUTAS do RF aqui (direto nos pontos)
                fold_num = 0

                # Itera sobre os folds
                for train_index, test_index in skf.split(X_all, stratify_option):
                    fold_num += 1
                    print(f"\nProcessando Fold {fold_num}/{K_FOLD_N_SPLITS}...")
                    X_train_fold, X_test_fold = X_all[train_index], X_all[test_index]
                    y_train_fold, y_test_fold = y_all[train_index], y_all[test_index]

                    print(f"  Treinando modelo RF no fold {fold_num} ({len(y_train_fold)} amostras)...")
                    rf_model_fold = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE, class_weight=RF_CLASS_WEIGHT, n_jobs=-1)
                    rf_model_fold.fit(X_train_fold, y_train_fold)

                    print(f"  Predizendo nos {len(y_test_fold)} pontos de teste do fold {fold_num}...")
                    y_pred_fold = rf_model_fold.predict(X_test_fold) # Predição direta nos pontos de teste

                    y_true_all_folds.extend(y_test_fold)
                    y_pred_all_folds.extend(y_pred_fold)

            except ValueError as e_skf:
                 # Erro comum se uma classe tem menos amostras que n_splits
                 print(f"\nERRO durante StratifiedKFold: {e_skf}")
                 print("Verifique se cada classe no GT tem pelo menos K_FOLD_N_SPLITS amostras.")
                 raise e_skf # Re-levanta o erro para parar a execução

            # Calcula métricas agregadas da validação cruzada (baseadas nas predições diretas do RF)
            print(f"\nCalculando métricas agregadas da Validação Cruzada K-Fold ({len(y_true_all_folds)} pontos)...")
            if not y_true_all_folds:
                 print("Erro K-Fold: Nenhuma predição válida coletada dos folds.")
                 metricas_validacao = None
            else:
                metricas_validacao = calcular_metricas(y_true_all_folds, y_pred_all_folds, CLASSES_LABELS)
                if metricas_validacao: # Adiciona info extra se métricas foram calculadas
                     metricas_validacao['metodo_validacao'] = f'{K_FOLD_N_SPLITS}-Fold Cross-Validation'
                     metricas_validacao['pontos_teste'] = len(y_true_all_folds)

            # Treina modelo final com TODOS os dados para o mapa final
            print("\nTreinando modelo RF final com TODOS os dados GT...")
            rf_model_final = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE, class_weight=RF_CLASS_WEIGHT, n_jobs=-1)
            rf_model_final.fit(X_all, y_all)

        elif VALIDATION_METHOD == 'ORIGINAL':
             print("\n--- Executando Método ORIGINAL (Treino e Validação no Mesmo Conjunto) ---")
             # Treina modelo final com TODOS os dados
             print("Treinando modelo RF com TODOS os dados GT...")
             rf_model_final = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE, class_weight=RF_CLASS_WEIGHT, n_jobs=-1)
             rf_model_final.fit(X_all, y_all)
             # Validação é feita nos dados de treino
             print("Predizendo nos dados de treino para métricas (método ORIGINAL)...")
             y_pred_train = rf_model_final.predict(X_all)
             metricas_validacao = calcular_metricas(y_all, y_pred_train, CLASSES_LABELS)
             if metricas_validacao:
                 metricas_validacao['metodo_validacao'] = 'Original (Treino=Teste)'
                 metricas_validacao['pontos_teste'] = X_all.shape[0]
             else:
                 metricas_validacao = {'metodo_validacao': 'Original (Treino=Teste)', 'pontos_teste': X_all.shape[0]} # Placeholder
             print("AVISO: Método 'ORIGINAL' realiza validação nos dados de treino, pode ser excessivamente otimista.")

        else:
            # Caso um método inválido seja especificado
            raise ValueError(f"VALIDATION_METHOD '{VALIDATION_METHOD}' inválido. Use 'HOLD_OUT', 'K_FOLD' ou 'ORIGINAL'.")

        # 4. Predição Final no Mapa e Pós-Processamento (usando o rf_model_final)
        if rf_model_final is None:
             raise ValueError("Modelo final (rf_model_final) não foi treinado devido a erro anterior ou método inválido.")

        print("\nGerando mapa de classificação FINAL...")
        # Reutiliza os índices de pixels válidos já calculados
        rows_pred_final, cols_pred_final = np.where(valid_data_mask_pred)
        # Reutiliza os dados das bandas já extraídos para esses pixels se possível (otimização)
        if 'X_pred_map' in locals() and X_pred_map.shape[0] == len(rows_pred_final):
             X_pred_map_final = X_pred_map # Reusa se Hold-Out já calculou
             print("(Reutilizando dados de pixel pré-calculados para predição final)")
        else:
             # Recalcula se K-Fold ou Original (não calcularam X_pred_map antes)
             print("(Calculando dados de pixel para predição final)")
             X_pred_map_final = np.vstack([b02_data[rows_pred_final, cols_pred_final], b03_data[rows_pred_final, cols_pred_final],
                                           b04_data[rows_pred_final, cols_pred_final], b08_data[rows_pred_final, cols_pred_final]]).T
        total_pixels_preditos = X_pred_map_final.shape[0]
        print(f"Predizendo {total_pixels_preditos} pixels com o modelo final...")
        predictions_map_final = rf_model_final.predict(X_pred_map_final)

        # Cria raster inicial do mapa final
        rf_classification_raster_initial_finalmap = np.full(b04_data.shape, OUTPUT_NODATA, dtype=np.uint8)
        rf_classification_raster_initial_finalmap[valid_data_mask_pred] = 0 # Define Não-Alvo onde válido
        rf_classification_raster_initial_finalmap[rows_pred_final, cols_pred_final] = predictions_map_final.astype(np.uint8) # Aplica predições
        pixels_alvo_inicial_rf = int(np.sum(rf_classification_raster_initial_finalmap == CLASSE_ALVO_LABEL))
        print(f"Pixels Alvo ANTES do filtro (Modelo Final): {pixels_alvo_inicial_rf}")

        # Aplica filtro de tamanho ao mapa final
        coords_para_shape_final = []
        # Inicia o raster final como o inicial (caso filtro não seja aplicado ou não encontre nada)
        rf_classification_raster_final_map = rf_classification_raster_initial_finalmap.copy()

        if APLICAR_FILTRO_TAMANHO_RF and pixels_alvo_inicial_rf > 0:
            print(f"Aplicando filtro de tamanho ({RF_GRUPO_MIN}-{RF_GRUPO_MAX} pixels) ao mapa final...")
            rf_alvo_mask_initial_finalmap = (rf_classification_raster_initial_finalmap == CLASSE_ALVO_LABEL)
            final_mask_bool_filtered_finalmap, grupos_coords_filtrados_finalmap = identificar_grupos_conectados_mask(
                rf_alvo_mask_initial_finalmap, min_size=RF_GRUPO_MIN, max_size=RF_GRUPO_MAX
            )
            # Cria o raster final filtrado (sobrescreve o anterior)
            rf_classification_raster_final_map = np.full(b04_data.shape, OUTPUT_NODATA, dtype=np.uint8)
            rf_classification_raster_final_map[valid_data_mask_pred] = 0 # Mantém Não-Alvo onde válido
            rf_classification_raster_final_map[final_mask_bool_filtered_finalmap] = CLASSE_ALVO_LABEL # Aplica Alvo filtrado

            pixels_alvo_final_rf = int(np.sum(final_mask_bool_filtered_finalmap))
            print(f"Pixels Alvo APÓS filtro (Modelo Final): {pixels_alvo_final_rf}")
            coords_para_shape_final = grupos_coords_filtrados_finalmap # Lista de coordenadas [[r,c],...] para o shape

        else: # Filtro não aplicado ou nenhum pixel alvo inicial
            if not APLICAR_FILTRO_TAMANHO_RF: print("\nFiltro de tamanho NÃO aplicado ao mapa final.")
            elif pixels_alvo_inicial_rf == 0: print("\nFiltro de tamanho não aplicado (nenhum pixel Alvo inicial no mapa final).")
            # Contagem final é a mesma da inicial
            pixels_alvo_final_rf = pixels_alvo_inicial_rf
            # Pega coordenadas se houver algum pixel alvo (mesmo sem filtro)
            if pixels_alvo_final_rf > 0:
                 coords_rf_alvo_rc_final = np.argwhere(rf_classification_raster_final_map == CLASSE_ALVO_LABEL)
                 if coords_rf_alvo_rc_final.size > 0:
                      # Agrupa como UMA lista de coordenadas [[r,c], ...]
                      coords_para_shape_final = [coords_rf_alvo_rc_final.tolist()]

        # 5. Exportar Resultados Finais (Mapa e Shapes do Modelo Final)
        print("\nExportando resultados finais (Mapa TIF e Shapefile)...")
        # Prepara o perfil final para o TIF de classificação (1 banda, uint8)
        out_profile_final_map = out_profile.copy();
        out_profile_final_map['count'] = 1
        out_profile_final_map['dtype'] = rasterio.uint8
        out_profile_final_map['nodata'] = OUTPUT_NODATA

        # Exporta o TIF final (filtrado ou não)
        raster_classificado_path_final = exportar_tif_classificacao(
            rf_classification_raster_final_map, out_profile_final_map, output_dir_rf, metodo_id
        )
        # Exporta o Shapefile final (se houver coordenadas)
        if coords_para_shape_final:
             shapes_path_final = exportar_shapes_finais(
                 transform_geo=out_profile_final_map['transform'],
                 crs_geo=out_profile_final_map['crs'],
                 todos_grupos_coords=coords_para_shape_final,
                 output_dir_rf=output_dir_rf,
                 sufixo_ano_metodo=metodo_id # Passa ID do ano/método (ex: 2020_rf)
             )
        else:
            print(f"Nenhum pixel alvo ({CLASSE_ALVO_NOME}) encontrado para exportar no shapefile final de {ano_str}.")
            shapes_path_final = None


    except StopIteration as stop_msg:
        # Erro esperado se não houver pixels válidos ou na área de estudo
        print(f"Processamento RF {ano_str} interrompido: {stop_msg}")
        metricas_validacao = None # Garante que não há métricas
        rf_classification_raster_final_map = None # Garante que não há raster
        out_profile_final_map = None # Garante que não há perfil
    except (FileNotFoundError, rasterio.RasterioIOError, ValueError, MemoryError, Exception) as e:
        # Captura outros erros críticos
        print(f"Erro Crítico no Processamento/Validação RF {ano_str}: {e}")
        traceback.print_exc()
        metricas_validacao, raster_classificado_path_final, shapes_path_final = None, None, None
        gdf_gt_valid = None
        rf_classification_raster_final_map = None
        out_profile_final_map = None
    finally:
        # Escreve Sumário RF para o ANO ATUAL, mesmo se houve erro (com N/A)
        print(f"\nEscrevendo sumário Random Forest para {ano_str} em: {sumario_path}")
        try:
            # Coleta informações para o sumário (com tratamento para caso de erro anterior)
            num_pontos_total_gt = X_all.shape[0] if 'X_all' in locals() and X_all is not None else 'N/A'
            dist_total_gt_str = str(dict(zip(*np.unique(y_all, return_counts=True)))) if 'y_all' in locals() and y_all is not None and len(y_all)>0 else 'N/A'
            pixels_alvo_apos_filtro = pixels_alvo_final_rf if 'pixels_alvo_final_rf' in locals() else 'N/A'
            metodo_val_str = metricas_validacao['metodo_validacao'] if metricas_validacao and 'metodo_validacao' in metricas_validacao else f'N/A ({VALIDATION_METHOD}?)'
            pontos_val_str = metricas_validacao['pontos_teste'] if metricas_validacao and 'pontos_teste' in metricas_validacao else 'N/A'
            pixels_geom_str = str(pixels_in_geom) if 'pixels_in_geom' in locals() else 'N/A'
            pixels_pred_str = str(total_pixels_preditos) if 'total_pixels_preditos' in locals() else 'N/A'
            pixels_ant_filtro_str = str(pixels_alvo_inicial_rf) if 'pixels_alvo_inicial_rf' in locals() else 'N/A'
            path_tif_str = os.path.basename(raster_classificado_path_final) if raster_classificado_path_final and os.path.exists(raster_classificado_path_final) else 'N/A ou Falha'
            path_shp_str = os.path.basename(shapes_path_final) if shapes_path_final and os.path.exists(shapes_path_final) else 'N/A ou Vazio'

            with open(sumario_path, 'w', encoding='utf-8') as f_sum:
                f_sum.write(f"Sumário da Classificação e Validação [Random Forest] - Ano {ano_str}\n")
                f_sum.write(f"Processado em: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f_sum.write("="*40 + "\n"); f_sum.write("Método: Random Forest Classifier\n")
                f_sum.write("Parâmetros Modelo:\n"); f_sum.write(f"  - n_estimators: {RF_N_ESTIMATORS}, random_state: {RF_RANDOM_STATE}, class_weight: {RF_CLASS_WEIGHT}\n")
                f_sum.write("Pós-processamento:\n")
                if APLICAR_FILTRO_TAMANHO_RF: f_sum.write(f"  - Filtro Tamanho Grupo: SIM (Min: {RF_GRUPO_MIN}, Max: {RF_GRUPO_MAX} pixels)\n")
                else: f_sum.write(f"  - Filtro Tamanho Grupo: NÃO\n")
                f_sum.write("="*40 + "\n"); f_sum.write("Dados de Referência (Ground Truth):\n")
                f_sum.write(f"  - Shapefile GT: {os.path.basename(shp_ground_truth_path)}\n")
                f_sum.write(f"  - Coluna Classe GT: '{col_classe_gt}'\n")
                f_sum.write(f"  - !! AVISO: Usando TODOS os {num_pontos_total_gt} pontos GT válidos para o ano {ano_str} (sem filtro de ano) !!\n")
                f_sum.write(f"  - Distribuição GT (0=NA, 1=Alvo): {dist_total_gt_str}\n")
                f_sum.write("="*40 + "\n"); f_sum.write("Validação Interna:\n")
                f_sum.write(f"  - Método: {metodo_val_str}\n")
                f_sum.write(f"  - Pontos Usados na Validação: {pontos_val_str}\n")
                f_sum.write("="*40 + "\n"); f_sum.write("Predição e Filtragem (Modelo Final):\n")
                f_sum.write(f"  - Pixels dentro da geometria: {pixels_geom_str}\n")
                f_sum.write(f"  - Total pixels preditos: {pixels_pred_str}\n")
                f_sum.write(f"  - Pixels Alvo ({CLASSE_ALVO_LABEL}) ANTES do filtro: {pixels_ant_filtro_str}\n")
                f_sum.write(f"  - Pixels Alvo ({CLASSE_ALVO_LABEL}) FINAIS (após filtro, se aplicado): {pixels_alvo_apos_filtro}\n")
                f_sum.write("="*40 + "\n"); f_sum.write("Resultados Finais Exportados:\n")
                f_sum.write(f"  - Raster Classificado Final: {path_tif_str}\n")
                f_sum.write(f"  - Shapefile Pontos Alvo Finais: {path_shp_str}\n")
        except Exception as e_sum:
            print(f"Erro ao escrever sumário RF {ano_str}: {e_sum}")

    print(f"Processamento e Validação RF ({VALIDATION_METHOD}) {ano_str} concluído.")
    # Retorna as métricas, caminhos dos arquivos, GDF usado, E o raster final + perfil
    return (metricas_validacao, raster_classificado_path_final, shapes_path_final,
            gdf_gt_valid, rf_classification_raster_final_map, out_profile_final_map) # <-- MODIFICADO: Retorna 6 itens


# --- FUNÇÃO AUXILIAR PARA CALCULAR MÉTRICAS ---
def calcular_metricas(y_true, y_pred, labels):
    """ Calcula um dicionário de métricas de classificação. """
    if len(y_true) == 0 or len(y_pred) == 0:
        print("Aviso: Não é possível calcular métricas com listas de referência ou predição vazias.")
        return None
    try:
        # Calcula a matriz de confusão
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Extrai TN, FP, FN, TP de forma segura (considera matrizes menores se uma classe não aparece)
        tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
        fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
        tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0

        # Gera o relatório de classificação como dicionário e texto
        report_dict = classification_report(y_true, y_pred, labels=labels, target_names=[f"Classe {l}" for l in labels], zero_division=0, output_dict=True)
        report_text = classification_report(y_true, y_pred, labels=labels, target_names=[f"Classe {l}" for l in labels], zero_division=0)
        target_label_str = f'Classe {CLASSE_ALVO_LABEL}' # String da classe alvo no report_dict

        # Coleta métricas do dicionário do classification_report com segurança (retorna 0 se não encontrar)
        precision_alvo = report_dict.get(target_label_str, {}).get('precision', 0)
        recall_alvo = report_dict.get(target_label_str, {}).get('recall', 0)
        f1_alvo = report_dict.get(target_label_str, {}).get('f1-score', 0)

        # Calcula outras métricas gerais
        acc_geral = accuracy_score(y_true, y_pred)
        # adjusted=False é a acurácia balanceada padrão; adjusted=True tenta corrigir por chance
        acc_balanc = balanced_accuracy_score(y_true, y_pred, adjusted=False)
        kappa_val = cohen_kappa_score(y_true, y_pred, labels=labels)
        # Usa average='binary' para ter IoU específico da classe alvo positiva (pos_label)
        iou_alvo = jaccard_score(y_true, y_pred, pos_label=CLASSE_ALVO_LABEL, average='binary', zero_division=0)

        # Monta o dicionário de retorno
        metrics = {
            'pontos_matriz': len(y_true), # Total de pontos usados na matriz
            'acuracia_geral': acc_geral,
            'acuracia_balanceada': acc_balanc,
            'kappa': kappa_val,
            'iou_alvo': iou_alvo,
            'precisao_alvo': precision_alvo,
            'recall_alvo': recall_alvo,
            'f1_alvo': f1_alvo,
            'cm_tn': tn, 'cm_fp': fp, 'cm_fn': fn, 'cm_tp': tp,
            'classification_report_text': report_text # Guarda o texto formatado do relatório
        }
        # 'metodo_validacao' e 'pontos_teste' são adicionados posteriormente na chamada principal
        return metrics
    except Exception as e:
        print(f"Erro inesperado ao calcular métricas: {e}")
        traceback.print_exc()
        return None


# --- FUNÇÃO PARA ESCREVER RELATÓRIO DE VALIDAÇÃO INDIVIDUAL POR ANO ---
def escrever_relatorio_validacao(metodo_id, ano_str, shp_gt_path, raster_classificado_path, dir_output_valid_rf, col_classe_gt, metricas, gdf_gt_usado):
    """ Escreve o relatório de validação individual para um ano. """
    # Verifica se as métricas são válidas
    if metricas is None or not isinstance(metricas, dict) or 'metodo_validacao' not in metricas:
        print(f"Erro Validação RF: Métricas inválidas ou não fornecidas para {ano_str}. Relatório individual não gerado.")
        # Cria um arquivo de erro para indicar a falha
        output_txt_erro = os.path.join(dir_output_valid_rf, f"validacao_{ano_str}_rf_ERRO.txt")
        with open(output_txt_erro, "w", encoding='utf-8') as f:
             f.write(f"Relatório de Validação [Random Forest] - Ano {ano_str} - ID: {metodo_id}\n")
             f.write(f"Processado em: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
             f.write("="*60 + "\n")
             f.write("ERRO: Métricas de validação não puderam ser calculadas ou foram fornecidas incorretamente.\n")
        return

    metodo_val_desc = metricas.get('metodo_validacao', f'N/A ({VALIDATION_METHOD} esperado)')
    print(f"\n--- Escrevendo Relatório de Validação Individual ({metodo_val_desc}) para {ano_str} ---")
    # Nome do arquivo inclui ano e método de validação
    output_txt = os.path.join(dir_output_valid_rf, f"validacao_{ano_str}_rf_{VALIDATION_METHOD}.txt")

    try:
        os.makedirs(dir_output_valid_rf, exist_ok=True)
        with open(output_txt, "w", encoding='utf-8') as f:
             f.write(f"Relatório de Validação [Random Forest] - Ano {ano_str}\n")
             f.write(f"Método de Validação Utilizado: {metodo_val_desc}\n")
             f.write(f"Processado em: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
             f.write("="*60 + "\n")
             f.write(f"Raster Classificado (Mapa Final): {os.path.basename(raster_classificado_path) if raster_classificado_path else 'N/A'}\n")
             f.write(f"Shapefile Ground Truth Usado: {os.path.basename(shp_gt_path)}\n")
             f.write(f"!! AVISO: TODOS os pontos GT foram usados (COLUNA_ANO_REF = None) !!\n")
             f.write(f"Coluna Classe GT: '{col_classe_gt}' (Alvo: '{CLASSE_ALVO_NOME}' -> {CLASSE_ALVO_LABEL})\n")
             f.write(f"Total Pontos GT Válidos (usados em treino/teste): {len(gdf_gt_usado) if gdf_gt_usado is not None else 'N/A'}\n")
             f.write(f"Total Pontos Usados na Matriz de Confusão ({metodo_val_desc}): {metricas.get('pontos_matriz', 'N/A')}\n")
             f.write(f"\nLabels: {CLASSES_LABELS} (0=Não Alvo, 1=Alvo)\n")

             # Matriz de Confusão
             f.write("\nMatriz de Confusão:\n"); f.write("(Linhas=Referência [GT], Colunas=Mapa [Predito])\n")
             cm_tn = metricas.get('cm_tn', 'N/A'); cm_fp = metricas.get('cm_fp', 'N/A')
             cm_fn = metricas.get('cm_fn', 'N/A'); cm_tp = metricas.get('cm_tp', 'N/A')
             pontos_matriz = metricas.get('pontos_matriz', 'N/A')
             # Verifica se os valores da CM são numéricos antes de formatar
             is_numeric = all(isinstance(v, (int, float, np.number)) for v in [cm_tn, cm_fp, cm_fn, cm_tp])

             if is_numeric and pontos_matriz != 'N/A' and isinstance(pontos_matriz, int):
                 header = f"{'Ref ->':<8}|{f' {CLASSES_LABELS[0]} ':^8}|{f' {CLASSES_LABELS[1]} ':^8}| Total Ref\n"
                 f.write(header); f.write("-" * len(header.strip()) + "\n")
                 f.write(f"{CLASSES_LABELS[0]:<8}|{cm_tn:^8d}|{cm_fp:^8d}| {cm_tn+cm_fp:^9d}\n")
                 f.write(f"{CLASSES_LABELS[1]:<8}|{cm_fn:^8d}|{cm_tp:^8d}| {cm_fn+cm_tp:^9d}\n")
                 f.write("-" * len(header.strip()) + "\n")
                 f.write(f"{'MapaTot':<8}|{cm_tn+cm_fn:^8d}|{cm_fp+cm_tp:^8d}| {pontos_matriz:^9d}\n")
                 f.write(f"\n  TN={cm_tn}, FP={cm_fp} (Comissão Alvo), FN={cm_fn} (Omissão Alvo), TP={cm_tp}\n")
             else:
                 f.write("  (Matriz de confusão não pôde ser exibida - dados ausentes ou inválidos)\n")
                 f.write(f"  TN={cm_tn}, FP={cm_fp}, FN={cm_fn}, TP={cm_tp}\n")

             # Métricas Gerais e da Classe Alvo
             f.write("\nMétricas Gerais:\n")
             acc_geral_str = f"{metricas.get('acuracia_geral', 'N/A'):.4f}" if isinstance(metricas.get('acuracia_geral'), (int, float, np.number)) else 'N/A'
             acc_bal_str = f"{metricas.get('acuracia_balanceada', 'N/A'):.4f}" if isinstance(metricas.get('acuracia_balanceada'), (int, float, np.number)) else 'N/A'
             kappa_str = f"{metricas.get('kappa', 'N/A'):.4f}" if isinstance(metricas.get('kappa'), (int, float, np.number)) else 'N/A'
             f.write(f"  Acurácia Geral: {acc_geral_str}\n")
             f.write(f"  Acurácia Balanceada: {acc_bal_str}\n")
             f.write(f"  Kappa: {kappa_str}\n")

             f.write(f"\nMétricas Classe Alvo ({CLASSE_ALVO_LABEL} = '{CLASSE_ALVO_NOME}'):\n")
             iou_str = f"{metricas.get('iou_alvo', 'N/A'):.4f}" if isinstance(metricas.get('iou_alvo'), (int, float, np.number)) else 'N/A'
             prec_str = f"{metricas.get('precisao_alvo', 'N/A'):.4f}" if isinstance(metricas.get('precisao_alvo'), (int, float, np.number)) else 'N/A'
             rec_str = f"{metricas.get('recall_alvo', 'N/A'):.4f}" if isinstance(metricas.get('recall_alvo'), (int, float, np.number)) else 'N/A'
             f1_str = f"{metricas.get('f1_alvo', 'N/A'):.4f}" if isinstance(metricas.get('f1_alvo'), (int, float, np.number)) else 'N/A'
             f.write(f"  IoU (Jaccard): {iou_str}\n")
             f.write(f"  Precisão:      {prec_str}\n")
             f.write(f"  Recall (Sens.): {rec_str}\n")
             f.write(f"  F1-Score:      {f1_str}\n")

             # Relatório Detalhado (sklearn)
             f.write("\nRelatório Classificação Detalhado (sklearn):\n")
             f.write(metricas.get('classification_report_text', 'N/A'))

        print(f"Relatório de validação Random Forest {ano_str} ({VALIDATION_METHOD}) salvo: {output_txt}")

    except Exception as e:
        print(f"Erro inesperado ao escrever relatório de validação {ano_str}: {e}")
        traceback.print_exc()


# --- FUNÇÃO PARA ESCREVER RELATÓRIO COMPARATIVO DE MÉTRICAS ---
def escrever_relatorio_comparativo(resultados_validacao, dir_output_valid_rf):
    """ Escreve um relatório comparando as métricas de validação entre os anos processados. """
    anos = sorted(list(resultados_validacao.keys())) # Ordena os anos (ex: ['2020', '2023'])
    if len(anos) < 1: # Precisa de pelo menos um resultado
        print("Aviso: Nenhum resultado de validação disponível para gerar relatório comparativo.")
        return
    if len(anos) < 2:
        print("Aviso: Relatório comparativo requer resultados de pelo menos dois anos.")
        # Poderia opcionalmente só imprimir o resultado do único ano aqui

    output_txt = os.path.join(dir_output_valid_rf, f"comparacao_validacao_rf_{VALIDATION_METHOD}.txt")
    print(f"\n--- Escrevendo Relatório Comparativo de Validação ({VALIDATION_METHOD}) ---")

    try:
        os.makedirs(dir_output_valid_rf, exist_ok=True)
        with open(output_txt, "w", encoding='utf-8') as f:
            f.write(f"Relatório Comparativo de Validação [Random Forest]\n")
            # Pega o método de validação do primeiro resultado (devem ser iguais)
            metodo_val = resultados_validacao[anos[0]].get('metodo_validacao', f'N/A ({VALIDATION_METHOD} esperado)')
            f.write(f"Método de Validação Utilizado: {metodo_val}\n")
            f.write(f"Gerado em: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Shapefile Ground Truth Base: {os.path.basename(GROUND_TRUTH_SHP)}\n")
            f.write(f"!! AVISO: O MESMO conjunto de pontos GT foi usado para validar todos os anos (COLUNA_ANO_REF = None) !!\n")
            f.write("="*70 + "\n")
            # Cabeçalho da tabela
            header_line = f"{'Métrica':<25}"
            for ano in anos:
                header_line += f"| {ano:^15} "
            f.write(header_line + "\n")
            f.write("-" * len(header_line.strip()) + "\n")

            # Função auxiliar interna para formatar e escrever linha da tabela
            def write_metric_line(metric_key, display_name, precision=4):
                line = f"{display_name:<25}"
                for ano in anos:
                    # Acessa o dicionário de métricas para o ano atual
                    metricas_ano = resultados_validacao.get(ano, {})
                    # Pega o valor da métrica específica
                    valor = metricas_ano.get(metric_key, 'N/A')
                    # Formata se for número, senão escreve N/A
                    if isinstance(valor, (int, float, np.number)):
                         # Formato especial para inteiros (precision=0)
                         fmt = f".{precision}f" if precision > 0 else "d"
                         line += f"| {valor:^15{fmt}} "
                    else:
                         line += f"| {'N/A':^15} "
                f.write(line + "\n")

            # Escreve as métricas principais na tabela
            write_metric_line('pontos_teste', 'Pontos na Validação', 0)
            f.write("-" * len(header_line.strip()) + "\n") # Linha separadora
            write_metric_line('acuracia_geral', 'Acurácia Geral')
            write_metric_line('acuracia_balanceada', 'Acurácia Balanceada')
            write_metric_line('kappa', 'Kappa')
            f.write("-" * len(header_line.strip()) + "\n") # Linha separadora
            f.write(f"Classe Alvo: {CLASSE_ALVO_NOME} ({CLASSE_ALVO_LABEL})\n")
            write_metric_line('iou_alvo', '  IoU (Jaccard)')
            write_metric_line('precisao_alvo', '  Precisão')
            write_metric_line('recall_alvo', '  Recall (Sensibilidade)')
            write_metric_line('f1_alvo', '  F1-Score')
            f.write("-" * len(header_line.strip()) + "\n") # Linha separadora
            f.write("Matriz Confusão (Alvo):\n")
            write_metric_line('cm_tp', '  Verd. Positivos (TP)', 0)
            write_metric_line('cm_fp', '  Falsos Positivos (FP)', 0)
            write_metric_line('cm_fn', '  Falsos Negativos (FN)', 0)
            write_metric_line('cm_tn', '  Verd. Negativos (TN)', 0)
            f.write("="*70 + "\n") # Linha final

        print(f"Relatório comparativo de validação salvo: {output_txt}")

    except Exception as e:
        print(f"Erro inesperado ao escrever relatório comparativo: {e}")
        traceback.print_exc()


# --- FUNÇÃO PARA EXPORTAR SHAPEFILE DE INTERSEÇÃO ---
def exportar_shapes_intersecao(coords_rc_intersecao, transform_geo, crs_geo, output_dir_rf, nome_arquivo_shp):
    """Exporta shapefile com pontos dos pixels da interseção."""
    output_shapefile = os.path.join(output_dir_rf, nome_arquivo_shp)
    print(f"Exportando shapefile de interseção para: {output_shapefile}")

    # Verifica se há coordenadas para exportar
    if coords_rc_intersecao is None or coords_rc_intersecao.size == 0:
        print(f"Aviso: Nenhuma coordenada de interseção fornecida. Shapefile '{nome_arquivo_shp}' não gerado.")
        return None

    lista_shapes = []
    pontos_processados = 0
    try:
        # Converte todas as coordenadas (linha, coluna) para (x, y) de uma vez
        # coords_rc_intersecao é um array Nx2, onde N é o número de pontos
        # Coluna 0 é linha (row), Coluna 1 é coluna (col)
        xs, ys = rasterio.transform.xy(transform_geo, coords_rc_intersecao[:, 0], coords_rc_intersecao[:, 1], offset='center')

        # Cria um dicionário para cada ponto com sua geometria e atributos
        for x, y, (r, c) in zip(xs, ys, coords_rc_intersecao):
            lista_shapes.append({
                "geometry": Point(x, y),
                "classe": CLASSE_ALVO_LABEL, # Adiciona a classe alvo como atributo
                "lin_img": int(r),           # Linha original no raster
                "col_img": int(c)            # Coluna original no raster
            })
            pontos_processados += 1

    except Exception as e:
        print(f"Erro durante a conversão de coordenadas para o shapefile de interseção: {e}")
        traceback.print_exc()
        return None

    # Verifica se geometrias foram criadas
    if not lista_shapes:
        print(f"Aviso: Nenhuma geometria Point criada para interseção. Shapefile '{nome_arquivo_shp}' não gerado.")
        return None

    # Cria e salva o GeoDataFrame
    try:
        valid_crs = crs_geo if crs_geo else None
        if valid_crs is None:
            warnings.warn(f"CRS não definido para interseção. Shapefile '{nome_arquivo_shp}' será gerado sem CRS.")

        gdf = gpd.GeoDataFrame(lista_shapes, crs=valid_crs)
        # Tenta salvar com UTF-8, se falhar, tenta com latin-1
        try:
            gdf.to_file(output_shapefile, driver='ESRI Shapefile', encoding='utf-8')
        except Exception as e_enc:
            warnings.warn(f"Falha ao salvar shape de interseção '{nome_arquivo_shp}' com UTF-8 ({e_enc}). Tentando latin-1.")
            try:
                gdf.to_file(output_shapefile, driver='ESRI Shapefile', encoding='latin-1')
            except Exception as e_enc2:
                print(f"Erro crítico ao salvar shape de interseção '{nome_arquivo_shp}' com latin-1: {e_enc2}")
                return None
        print(f"Shapefile de interseção ({pontos_processados} pontos) exportado: {output_shapefile}")
        return output_shapefile
    except Exception as e:
        print(f"Erro ao criar/salvar GeoDataFrame de interseção: {e}")
        traceback.print_exc()
        return None


# --- BLOCO PRINCIPAL DE EXECUÇÃO ---
if __name__ == "__main__":
    # Cria diretórios de saída (com tratamento de erro)
    try:
        os.makedirs(OUTPUT_DIR_RF, exist_ok=True)
        os.makedirs(OUTPUT_DIR_VALID_RF, exist_ok=True)
        print(f"Diretórios de saída:")
        print(f"  Resultados RF: {OUTPUT_DIR_RF}")
        print(f"  Validação RF:  {OUTPUT_DIR_VALID_RF}")
    except OSError as e:
        print(f"Erro crítico ao criar diretórios de saída: {e}. Verifique os caminhos e permissões.")
        exit() # Interrompe o script se não puder criar diretórios

    # Imprime informações iniciais
    print(f"\nMétodo de Validação Selecionado: {VALIDATION_METHOD}")
    if VALIDATION_METHOD == 'HOLD_OUT': print(f"  Tamanho do Test Set (Hold-Out): {HOLD_OUT_TEST_SIZE*100:.0f}%")
    if VALIDATION_METHOD == 'K_FOLD': print(f"  Número de Folds (K-Fold): {K_FOLD_N_SPLITS}")

    # Registra hora de início
    start_time = datetime.datetime.now()
    brasilia_tz = None
    if pytz:
        try:
            brasilia_tz = pytz.timezone('America/Sao_Paulo')
            print(f"\nInício: {start_time.astimezone(brasilia_tz).strftime('%Y-%m-%d %H:%M:%S %Z%z')}")
        except pytz.UnknownTimeZoneError:
            print("Aviso: Fuso horário 'America/Sao_Paulo' desconhecido. Usando horário local.")
            print(f"\nInício: {start_time.strftime('%Y-%m-%d %H:%M:%S')} (horário local)")
    else:
        print("Biblioteca 'pytz' não encontrada. Usando horário local.")
        print(f"\nInício: {start_time.strftime('%Y-%m-%d %H:%M:%S')} (horário local)")


    # Define os anos e bandas a processar
    anos_para_processar = {'2020': SENTINEL_BANDS_2020, '2023': SENTINEL_BANDS_2023 }

    # Dicionários para guardar resultados por ano
    resultados_validacao_geral = {} # Guarda dicionários de métricas
    resultados_rasters_finais = {}  # Guarda arrays NumPy dos rasters finais
    resultados_perfis_finais = {}   # Guarda dicionários de perfis (metadados)

    # Loop principal para processar cada ano
    for ano, bandas in anos_para_processar.items():
        print(f"\n{'='*15} PROCESSANDO ANO: {ano} {'='*15}")

        # Zera variáveis para o ano atual
        metricas_calculadas, raster_final_path, shape_final_path = None, None, None
        gdf_gt_utilizado, raster_array, raster_profile = None, None, None

        try:
             # Chama a função principal - AGORA RETORNA 6 VALORES
             (metricas_calculadas, raster_final_path, shapes_path_final,
              gdf_gt_utilizado, raster_array, raster_profile) = processar_ano_rf(
                  ano_str=ano,
                  bandas_paths=bandas,
                  shape_limite_path=STUDY_AREA_SHP,
                  shp_ground_truth_path=GROUND_TRUTH_SHP,
                  output_dir_rf=OUTPUT_DIR_RF,
                  col_classe_gt=COLUNA_CLASSE_REF # Passa só a coluna da classe
              )

             # Armazena os resultados deste ano se foram gerados com sucesso
             if metricas_calculadas is not None:
                 resultados_validacao_geral[ano] = metricas_calculadas
                 print(f"Métricas de validação para {ano} armazenadas.")
             else:
                 print(f"AVISO: Métricas de validação para {ano} não foram calculadas/armazenadas.")

             if raster_array is not None and raster_profile is not None:
                 resultados_rasters_finais[ano] = raster_array
                 resultados_perfis_finais[ano] = raster_profile
                 print(f"Raster e perfil final para {ano} armazenados.")
             else:
                 print(f"AVISO: Raster ou perfil final para {ano} não foi gerado/armazenado (necessário para interseção).")

             # Escreve o relatório de validação INDIVIDUAL para este ano (se houver métricas)
             if metricas_calculadas:
                  escrever_relatorio_validacao(
                       metodo_id=f"{ano}_rf", ano_str=ano,
                       shp_gt_path=GROUND_TRUTH_SHP,
                       raster_classificado_path=raster_final_path,
                       dir_output_valid_rf=OUTPUT_DIR_VALID_RF,
                       col_classe_gt=COLUNA_CLASSE_REF,
                       metricas=metricas_calculadas,
                       gdf_gt_usado=gdf_gt_utilizado
                  )
             # Não precisa de 'else' aqui, o aviso já foi dado acima se métricas são None

        except Exception as e_proc_rf:
             # Captura erro GERAL no processamento do ano
             print(f"!!!!!! FALHA GERAL NO PROCESSAMENTO/VALIDAÇÃO PARA O ANO {ano} !!!!!!")
             print(f"Erro: {e_proc_rf}")
             traceback.print_exc()
             print(f"!!!!!! Continuando para o próximo ano (se houver)... !!!!!!")


        print(f"===== PROCESSAMENTO ANO {ano} CONCLUÍDO (ou falhou) =====")

    # --- FIM DO LOOP ---

    # Escreve o relatório comparativo de MÉTRICAS (se houver resultados em resultados_validacao_geral)
    if resultados_validacao_geral:
        escrever_relatorio_comparativo(resultados_validacao_geral, OUTPUT_DIR_VALID_RF)
    else:
        print("\nNenhum resultado de validação foi armazenado. Relatório comparativo de métricas não pode ser gerado.")

    # Calcula e Exporta Interseção dos Resultados (se houver rasters e perfis para ambos os anos)
    print("\n--- Calculando e Exportando Interseção dos Resultados ---")
    # Verifica se temos os dados necessários para os anos chave '2020' e '2023'
    anos_chave = ['2020', '2023']
    if all(ano in resultados_rasters_finais for ano in anos_chave) and \
       all(ano in resultados_perfis_finais for ano in anos_chave):

        raster_2020 = resultados_rasters_finais['2020']
        raster_2023 = resultados_rasters_finais['2023']
        profile_2020 = resultados_perfis_finais['2020']
        profile_2023 = resultados_perfis_finais['2023']

        # Verifica adicionalmente se os rasters e perfis são válidos e compatíveis
        # (Pode já ter sido verificado, mas é uma segurança extra)
        if (raster_2020 is not None and raster_2023 is not None and
            profile_2020 is not None and profile_2023 is not None and
            raster_2020.shape == raster_2023.shape and
            profile_2020.get('crs') == profile_2023.get('crs') and
            # Compara transforms com uma tolerância pequena para evitar problemas de ponto flutuante
            profile_2020.get('transform') is not None and profile_2023.get('transform') is not None and
            profile_2020['transform'].almost_equals(profile_2023['transform'])):

            print("Rasters de 2020 e 2023 encontrados e compatíveis. Calculando interseção...")

            # Cria máscaras booleanas para a classe alvo
            mask_2020 = (raster_2020 == CLASSE_ALVO_LABEL)
            mask_2023 = (raster_2023 == CLASSE_ALVO_LABEL)

            # Calcula a interseção (pixels que são True em ambas as máscaras)
            intersection_mask = mask_2020 & mask_2023

            # Encontra as coordenadas (linha, coluna) dos pixels na interseção
            # np.argwhere retorna um array Nx2 onde N é o número de pontos True
            coords_rc_intersecao = np.argwhere(intersection_mask)
            num_pixels_intersecao = coords_rc_intersecao.shape[0] # Número de linhas = número de pontos

            print(f"Número de pixels classificados como '{CLASSE_ALVO_NOME}' em ambos os anos: {num_pixels_intersecao}")

            # Exporta o shapefile se houver interseção
            if num_pixels_intersecao > 0:
                exportar_shapes_intersecao(
                    coords_rc_intersecao=coords_rc_intersecao,
                    transform_geo=profile_2023['transform'], # Usa um dos perfis como referência
                    crs_geo=profile_2023['crs'],
                    output_dir_rf=OUTPUT_DIR_RF, # Salva junto com os outros shapes
                    # Nome do arquivo inclui a classe alvo e os anos
                    nome_arquivo_shp=f"shapes_intersecao_{CLASSE_ALVO_NOME}_2020_2023.shp"
                )
            else:
                print("Nenhum pixel de interseção encontrado para a classe alvo.")

        else:
            # Mensagem de erro se os rasters/perfis não forem compatíveis
            print("Erro: Rasters/Perfis de 2020 e 2023 são incompatíveis (dimensões, CRS ou transform diferentes). Interseção não calculada.")
            if raster_2020 is not None and raster_2023 is not None:
                 print(f"  Shape 2020: {raster_2020.shape}, Shape 2023: {raster_2023.shape}")
            if profile_2020 is not None and profile_2023 is not None:
                 print(f"  CRS 2020: {profile_2020.get('crs')}, CRS 2023: {profile_2023.get('crs')}")
                 print(f"  Transform 2020: {profile_2020.get('transform')}")
                 print(f"  Transform 2023: {profile_2023.get('transform')}")

    else:
        # Mensagem se faltar resultado de um dos anos
        print("Não foi possível calcular a interseção: resultados (raster/perfil) de 2020 e/ou 2023 não estão disponíveis.")
        print(f"  Resultados disponíveis para os anos: {list(resultados_rasters_finais.keys())}")

    # --- FIM DA LÓGICA DE INTERSEÇÃO ---


    # Registra hora de término e tempo total
    end_time = datetime.datetime.now()
    print(f"\n--- Script Totalmente Concluído ---")
    if brasilia_tz: print(f"Término: {end_time.astimezone(brasilia_tz).strftime('%Y-%m-%d %H:%M:%S %Z%z')}")
    else: print(f"Término: {end_time.strftime('%Y-%m-%d %H:%M:%S')} (horário local)")
    print(f"Tempo total de execução: {end_time - start_time}")
