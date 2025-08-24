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
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


try:
    import pytz
except ImportError:
    pytz = None

# --- CONSTANTES ---

METHODS_TO_PROCESS = ['RF', 'SVM']
VALIDATION_METHOD = 'K_FOLD'
HOLD_OUT_TEST_SIZE = 0.3
K_FOLD_N_SPLITS = 5

SENTINEL_BANDS_2020 = {
    'B02': r"C:\Users\natan\Downloads\mest2-20250822T141805Z-1-001\mest2\2020\R10m\T24KVG_20201007T125311_B02_10m.jp2",
    'B03': r"C:\Users\natan\Downloads\mest2-20250822T141805Z-1-001\mest2\2020\R10m\T24KVG_20201007T125311_B03_10m.jp2",
    'B04': r"C:\Users\natan\Downloads\mest2-20250822T141805Z-1-001\mest2\2020\R10m\T24KVG_20201007T125311_B04_10m.jp2",
    'B08': r"C:\Users\natan\Downloads\mest2-20250822T141805Z-1-001\mest2\2020\R10m\T24KVG_20201007T125311_B08_10m.jp2"
}
SENTINEL_BANDS_2023 = {
    'B02': r"C:\Users\natan\Downloads\mest2-20250822T141805Z-1-001\mest2\2023\R10m\T24KVG_20230927T125309_B02_10m.jp2",
    'B03': r"C:\Users\natan\Downloads\mest2-20250822T141805Z-1-001\mest2\2023\R10m\T24KVG_20230927T125309_B03_10m.jp2",
    'B04': r"C:\Users\natan\Downloads\mest2-20250822T141805Z-1-001\mest2\2023\R10m\T24KVG_20230927T125309_B04_10m.jp2",
    'B08': r"C:\Users\natan\Downloads\mest2-20250822T141805Z-1-001\mest2\2023\R10m\T24KVG_20230927T125309_B08_10m.jp2"
}

STUDY_AREA_SHP = r"C:\Users\natan\Downloads\mest2-20250822T141805Z-1-001\mest2\limite\limite.shp"
OUTPUT_DIR_BASE = r'C:\Users\natan\Downloads\mest2-20250822T141805Z-1-001\mest2\resultados_comparativos_filtrado'
GROUND_TRUTH_SHP = r'C:\Users\natan\Downloads\mest2-20250822T141805Z-1-001\mest2\resultados grouth\pontos4.shp'

# --- Parâmetros dos Modelos ---

RF_N_ESTIMATORS = 100
RF_RANDOM_STATE = 42
RF_CLASS_WEIGHT = 'balanced'

SVM_APPLY_SCALING = True
SVM_KERNEL = 'rbf'
SVM_C = 1.0
SVM_GAMMA = 'scale'
SVM_CLASS_WEIGHT = 'balanced'
SVM_RANDOM_STATE = 42

APLICAR_FILTRO_TAMANHO = True
GRUPO_MIN = 3
GRUPO_MAX = 4

# --- COLUNAS E CLASSES NO GROUND TRUTH ---
COLUNA_ANO_REF = None
COLUNA_CLASSE_REF = 'NIVEL_III'

CLASSES_LABELS = [0, 1]
CLASSE_ALVO_NOME = 'Sapucaia'
CLASSE_ALVO_LABEL = 1

INTERNAL_NODATA = -9999
OUTPUT_NODATA = 255

# --- FUNÇÕES AUXILIARES ---

def identificar_grupos_conectados_mask(input_mask, min_size=2, max_size=1000000):
    """Identifica grupos conectados, retorna máscara, lista de coordenadas de cada grupo e contagens."""
    input_mask = input_mask.astype(bool)
    if not np.any(input_mask):
        return np.zeros_like(input_mask, dtype=bool), [], 0, {}

    structure = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=bool)
    array_rotulado, num_total_componentes = label(input_mask, structure=structure)

    lista_de_grupos_coords = []
    final_mask = np.zeros_like(input_mask, dtype=bool)
    num_grupos_validos = 0
    group_size_counts = {}

    if num_total_componentes > 0:
        component_ids, tamanhos = np.unique(array_rotulado, return_counts=True)
        component_ids_sem_fundo = component_ids[1:]
        tamanhos_sem_fundo = tamanhos[1:]
        mascara_tamanho_valido = (tamanhos_sem_fundo >= min_size) & (tamanhos_sem_fundo <= max_size)
        ids_tamanho_certo = component_ids_sem_fundo[mascara_tamanho_valido]
        tamanhos_dos_grupos_validos = tamanhos_sem_fundo[mascara_tamanho_valido]
        num_grupos_validos = len(ids_tamanho_certo)

        if num_grupos_validos > 0:
            tamanhos_unicos, contagens = np.unique(tamanhos_dos_grupos_validos, return_counts=True)
            group_size_counts = {int(t): int(c) for t, c in zip(tamanhos_unicos, contagens)}
            final_mask = np.isin(array_rotulado, ids_tamanho_certo)
            print(f"Filtro de tamanho encontrou {num_grupos_validos} grupos. Detalhes (tamanho: contagem): {group_size_counts}")
            for group_id in ids_tamanho_certo:
                coords_rc = np.argwhere(array_rotulado == group_id)
                if coords_rc.size > 0:
                    lista_de_grupos_coords.append(coords_rc)
    else:
        print("Nenhum grupo encontrado na máscara inicial para aplicar filtro.")

    return final_mask, lista_de_grupos_coords, num_grupos_validos, group_size_counts

def exportar_shapes_finais(transform_geo, crs_geo, todos_grupos_coords, output_dir, sufixo_ano_metodo):
    """Exporta shapefiles de pixels e centroides, tratando cada grupo individualmente."""
    print(f"Exportando shapefiles para {sufixo_ano_metodo}...")
    output_shapefile_pontos = os.path.join(output_dir, f'shapes_alvo_pixels_{sufixo_ano_metodo}.shp')
    output_shapefile_centroides = os.path.join(output_dir, f'shapes_alvo_centroides_{sufixo_ano_metodo}.shp')

    if not todos_grupos_coords:
        print(f"Aviso: Nenhuma coordenada de grupo fornecida. Shapefiles não gerados.")
        return None, None

    lista_todos_pontos = []
    lista_centroides = []
    ano_shape = sufixo_ano_metodo.split('_')[0]
    metodo_shape = sufixo_ano_metodo.split('_')[-1]
    print(f"Processando {len(todos_grupos_coords)} grupo(s) de coordenadas para gerar centroides individuais...")

    for i, grupo_coords_rc in enumerate(todos_grupos_coords):
        group_id = i + 1
        if grupo_coords_rc.size == 0: continue
        xs, ys = rasterio.transform.xy(transform_geo, grupo_coords_rc[:, 0], grupo_coords_rc[:, 1], offset="center")
        for x, y, (lin, col) in zip(xs, ys, grupo_coords_rc):
            lista_todos_pontos.append({
                "geometry": Point(x, y), "metodo": metodo_shape, "ano": ano_shape,
                "group_id": group_id, "lin_img": int(lin), "col_img": int(col)
            })
        if xs.size > 0:
            centroide_geom = Point(np.mean(xs), np.mean(ys))
            lista_centroides.append({
                'geometry': centroide_geom, 'group_id': group_id, 'num_pixels': len(xs),
                'metodo': metodo_shape, 'ano': ano_shape
            })

    if not lista_todos_pontos:
        print(f"Aviso: Nenhuma geometria de PONTO criada. Shapefiles não gerados.")
        return None, None

    try:
        valid_crs = crs_geo if crs_geo else None
        if valid_crs is None: warnings.warn(f"CRS não definido para {sufixo_ano_metodo}.")

        gdf_pontos = gpd.GeoDataFrame(lista_todos_pontos, crs=valid_crs)
        gdf_pontos['x_coord'] = gdf_pontos.geometry.x
        gdf_pontos['y_coord'] = gdf_pontos.geometry.y
        gdf_pontos.to_file(output_shapefile_pontos, driver='ESRI Shapefile', encoding='utf-8')
        print(f"Shapefile de PIXELS ({len(gdf_pontos)} pontos) exportado: {output_shapefile_pontos}")

        if lista_centroides:
            gdf_centroides = gpd.GeoDataFrame(lista_centroides, crs=valid_crs)
            gdf_centroides['x_coord'] = gdf_centroides.geometry.x
            gdf_centroides['y_coord'] = gdf_centroides.geometry.y
            gdf_centroides.to_file(output_shapefile_centroides, driver='ESRI Shapefile', encoding='utf-8')
            print(f"Shapefile de CENTROIDES ({len(gdf_centroides)} grupos) exportado: {output_shapefile_centroides}")
        else:
            output_shapefile_centroides = None
            print("Nenhum centroide para exportar.")

        return output_shapefile_pontos, output_shapefile_centroides
    except Exception as e:
        print(f"Erro ao criar/salvar GeoDataFrames para {sufixo_ano_metodo}: {e}")
        traceback.print_exc()
        return None, None

def exportar_tif_classificacao(mascara_final_numerica, out_profile, output_dir, nome_sufixo_metodo):
    output_tif = os.path.join(output_dir, f'raster_classificacao_{nome_sufixo_metodo}.tif')
    if mascara_final_numerica is None: return None
    mascara_uint8 = mascara_final_numerica.astype(np.uint8)
    print(f"Exportando TIF de classificação para {nome_sufixo_metodo} ({np.sum(mascara_uint8 == CLASSE_ALVO_LABEL)} pixels alvo)...")
    try:
        profile_final = out_profile.copy()
        profile_final.update(dtype=rasterio.uint8, count=1, nodata=OUTPUT_NODATA, compress='lzw', driver='GTiff')
        profile_final.pop('blockxsize', None); profile_final.pop('blockysize', None); profile_final.pop('tiled', None)
        with rasterio.open(output_tif, 'w', **profile_final) as dst:
            dst.write(mascara_uint8, 1)
        print(f"TIF de classificação {nome_sufixo_metodo} exportado.")
        return output_tif
    except Exception as e:
        print(f"Erro crítico ao exportar TIF {nome_sufixo_metodo}: {e}")
        traceback.print_exc()
        return None

# --- FUNÇÃO DE PROCESSAMENTO PRINCIPAL ---
def processar_ano_metodo(ano_str, metodo_str, bandas_paths, shape_limite_path, shp_ground_truth_path, output_dir_metodo, col_classe_gt):
    metodo_str_upper = metodo_str.upper()
    print(f"\n--- Iniciando Processamento [{metodo_str_upper}] e Validação ({VALIDATION_METHOD}) para {ano_str} ---")
    metodo_id = f"{ano_str}_{metodo_str.lower()}"
    sumario_path = os.path.join(output_dir_metodo, f'sumario_{metodo_id}.txt')

    raster_classificado_path_final, shapes_path_final, centroid_path_final = None, None, None
    X_all, y_all = [], []
    pixels_in_geom, total_pixels_preditos = 0, 0
    classification_raster_final_map, out_profile_final_map = None, None
    pixels_alvo_inicial, pixels_alvo_final = 0, 0
    num_grupos_filtrados_final = 0
    contagem_grupos_por_tamanho = {}
    metricas_validacao, gdf_gt_valid = None, None

    try:
        # 1. Leitura e Preparação dos Dados Geoespaciais
        print("Lendo shapefile de limite...")
        limite_gdf = gpd.read_file(shape_limite_path)
        if limite_gdf.empty: raise ValueError("Shapefile de limite vazio.")

        try:
            ref_raster_path = bandas_paths['B04']
            if not os.path.exists(ref_raster_path):
                raise FileNotFoundError(f"Arquivo da Banda B04 não encontrado para {ano_str}: {ref_raster_path}")
            print(f"Abrindo raster de referência: {ref_raster_path}")
            with rasterio.open(ref_raster_path) as src_ref:
                raster_crs = src_ref.crs; raster_profile = src_ref.profile
                raster_width, raster_height = src_ref.width, src_ref.height
            print(f"CRS do Raster: {raster_crs}")
        except Exception as e_ref: raise ValueError(f"Erro ao processar raster de referência: {e_ref}")

        if limite_gdf.crs != raster_crs:
            print(f"Reprojetando limite de {limite_gdf.crs} para {raster_crs}...")
            limite_gdf = limite_gdf.to_crs(raster_crs)
        limite_geom = limite_gdf.geometry.tolist()

        sources = {}
        try:
            print("Abrindo arquivos das bandas Sentinel...")
            for band, path in bandas_paths.items():
                 if not os.path.exists(path): raise FileNotFoundError(f"Arquivo da banda {band} não encontrado para {ano_str}: {path}")
                 sources[band] = rasterio.open(path)

            out_window = geometry_window(sources['B04'], limite_geom, pad_x=1, pad_y=1)
            out_window = out_window.intersection(Window(0, 0, raster_width, raster_height))
            if not (out_window.width > 0 and out_window.height > 0):
                raise ValueError(f"Área de estudo (limite) parece estar fora da extensão do raster para {ano_str}.")
            out_transform = rasterio.windows.transform(out_window, sources['B04'].transform)

            out_profile = raster_profile.copy()
            out_profile.update(height=out_window.height, width=out_window.width, transform=out_transform, nodata=INTERNAL_NODATA, driver='GTiff', count=len(sources))
            out_profile.pop('blockxsize', None); out_profile.pop('blockysize', None); out_profile.pop('tiled', None)

            print(f"Lendo dados das bandas {ano_str} ({int(out_window.width)}x{int(out_window.height)} pixels)...")
            band_data = {band: src.read(1, window=out_window, boundless=True, fill_value=INTERNAL_NODATA) for band, src in sources.items()}
            b02_data, b03_data, b04_data, b08_data = band_data['B02'], band_data['B03'], band_data['B04'], band_data['B08']
        finally:
            for src in sources.values():
                if src and not src.closed: src.close()
            print("Arquivos das bandas Sentinel fechados.")

        mask_geom = geometry_mask(limite_geom, out_shape=b04_data.shape, transform=out_transform, invert=True, all_touched=True)
        pixels_in_geom = int(np.sum(mask_geom))
        if pixels_in_geom == 0: raise StopIteration(f"Nenhum pixel do raster coincide com a área de estudo para {ano_str}.")
        
        valid_data_mask_pred = mask_geom & (b02_data != INTERNAL_NODATA) & (b03_data != INTERNAL_NODATA) & (b04_data != INTERNAL_NODATA) & (b08_data != INTERNAL_NODATA)
        if not np.any(valid_data_mask_pred): raise StopIteration(f"Nenhum pixel com dados válidos encontrado na área de estudo para {ano_str}.")

        # 2. Preparar Dados de Treinamento/Validação
        print(f"\nPreparando dados GT do shapefile: {os.path.basename(shp_ground_truth_path)}...")
        gdf_gt = gpd.read_file(shp_ground_truth_path)
        if gdf_gt.empty: raise ValueError("Shapefile de Ground Truth vazio.")
        if col_classe_gt not in gdf_gt.columns: raise ValueError(f"Coluna '{col_classe_gt}' não encontrada no shapefile GT.")
        if gdf_gt.crs != raster_crs: gdf_gt = gdf_gt.to_crs(raster_crs)

        coords_gt_xy = [(p.x, p.y) for p in gdf_gt.geometry]
        rows_gt, cols_gt = rasterio.transform.rowcol(out_transform, [c[0] for c in coords_gt_xy], [c[1] for c in coords_gt_xy])

        valid_indices_gt, X_all_list = [], []
        for i, (r, c) in enumerate(zip(rows_gt, cols_gt)):
            if 0 <= r < b04_data.shape[0] and 0 <= c < b04_data.shape[1]:
                band_values = [b02_data[r, c], b03_data[r, c], b04_data[r, c], b08_data[r, c]]
                if all(val != INTERNAL_NODATA for val in band_values):
                    valid_indices_gt.append(i)
                    X_all_list.append(band_values)
        
        if not valid_indices_gt: raise ValueError(f"Nenhum ponto GT válido encontrado na área de estudo para {ano_str}.")

        X_all = np.array(X_all_list)
        gdf_gt_valid = gdf_gt.iloc[valid_indices_gt].copy()
        classe_map = {CLASSE_ALVO_NOME: CLASSE_ALVO_LABEL}
        gdf_gt_valid['Ref_Num'] = gdf_gt_valid[col_classe_gt].str.strip().map(classe_map).fillna(0).astype(int)
        y_all = gdf_gt_valid['Ref_Num'].values

        print(f"Total de amostras GT válidas: {X_all.shape[0]}")
        print(f"  Distribuição GT (0=NãoAlvo, 1=Alvo): {dict(zip(*np.unique(y_all, return_counts=True)))}")
        if len(np.unique(y_all)) < 2: warnings.warn(f"ATENÇÃO: Apenas uma classe GT presente. Validação pode ser limitada.")

        # 3. Treinamento e Validação
        model_final = None
        if VALIDATION_METHOD == 'K_FOLD':
            print(f"\n--- Executando Validação K-FOLD ({K_FOLD_N_SPLITS} folds) ---")
            if X_all.shape[0] < K_FOLD_N_SPLITS : raise ValueError(f"Não há dados suficientes ({X_all.shape[0]}) para {K_FOLD_N_SPLITS}-Fold CV.")
            stratify_option = y_all if len(np.unique(y_all)) > 1 else None
            
            skf = StratifiedKFold(n_splits=K_FOLD_N_SPLITS, shuffle=True, random_state=RF_RANDOM_STATE)
            y_true_all_folds, y_pred_all_folds = [], []
            for fold_num, (train_index, test_index) in enumerate(skf.split(X_all, stratify_option), 1):
                print(f"\nProcessando Fold {fold_num}/{K_FOLD_N_SPLITS}...")
                X_train_fold, X_test_fold = X_all[train_index], X_all[test_index]
                y_train_fold, y_test_fold = y_all[train_index], y_all[test_index]
                
                if metodo_str_upper == 'SVM' and SVM_APPLY_SCALING:
                    scaler_fold = StandardScaler().fit(X_train_fold)
                    X_train_fold = scaler_fold.transform(X_train_fold)
                    X_test_fold = scaler_fold.transform(X_test_fold)

                print(f"  Treinando modelo {metodo_str_upper} no fold {fold_num}...")
                if metodo_str_upper == 'RF':
                    model_fold = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE, class_weight=RF_CLASS_WEIGHT, n_jobs=-1)
                elif metodo_str_upper == 'SVM':
                    model_fold = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, class_weight=SVM_CLASS_WEIGHT, random_state=SVM_RANDOM_STATE)
                model_fold.fit(X_train_fold, y_train_fold)

                y_pred_fold = model_fold.predict(X_test_fold)
                y_true_all_folds.extend(y_test_fold)
                y_pred_all_folds.extend(y_pred_fold)

            print(f"\nCalculando métricas agregadas da Validação Cruzada K-Fold...")
            metricas_validacao = calcular_metricas(y_true_all_folds, y_pred_all_folds, CLASSES_LABELS)
            if metricas_validacao:
                 metricas_validacao['metodo_validacao'] = f'{K_FOLD_N_SPLITS}-Fold Cross-Validation'
                 metricas_validacao['pontos_teste'] = len(y_true_all_folds)
        
        # Treinamento do modelo final com todos os dados 
        X_all_final = X_all.copy()
        scaler_final = None
        if metodo_str_upper == 'SVM' and SVM_APPLY_SCALING:
            scaler_final = StandardScaler().fit(X_all_final)
            X_all_final = scaler_final.transform(X_all_final)

        print(f"\nTreinando modelo {metodo_str_upper} final com TODOS os dados GT...")
        if metodo_str_upper == 'RF':
            model_final = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE, class_weight=RF_CLASS_WEIGHT, n_jobs=-1)
        elif metodo_str_upper == 'SVM':
            model_final = SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, class_weight=SVM_CLASS_WEIGHT, random_state=SVM_RANDOM_STATE, probability=True)
        
        if model_final is None: raise ValueError(f"Método '{metodo_str_upper}' inválido.")
        model_final.fit(X_all_final, y_all)
        model_final.scaler = scaler_final 

        # 4. Predição Final e Pós-Processamento
        print("\nGerando mapa de classificação FINAL...")
        rows_pred, cols_pred = np.where(valid_data_mask_pred)
        X_pred_map = np.vstack([b02_data[rows_pred, cols_pred], b03_data[rows_pred, cols_pred],
                                b04_data[rows_pred, cols_pred], b08_data[rows_pred, cols_pred]]).T
        total_pixels_preditos = X_pred_map.shape[0]
        if hasattr(model_final, 'scaler') and model_final.scaler is not None:
            X_pred_map = model_final.scaler.transform(X_pred_map)

        predictions_map = model_final.predict(X_pred_map)
        classification_raster_initial_finalmap = np.full(b04_data.shape, OUTPUT_NODATA, dtype=np.uint8)
        classification_raster_initial_finalmap[valid_data_mask_pred] = 0
        classification_raster_initial_finalmap[rows_pred, cols_pred] = predictions_map.astype(np.uint8)
        pixels_alvo_inicial = int(np.sum(classification_raster_initial_finalmap == CLASSE_ALVO_LABEL))
        print(f"Pixels Alvo ANTES do filtro (Modelo Final): {pixels_alvo_inicial}")

        coords_para_shape_final = []
        classification_raster_final_map = classification_raster_initial_finalmap.copy()
        if APLICAR_FILTRO_TAMANHO and pixels_alvo_inicial > 0:
            alvo_mask = (classification_raster_initial_finalmap == CLASSE_ALVO_LABEL)
            final_mask_bool, grupos_coords, num_grupos, contagem_tamanhos = identificar_grupos_conectados_mask(
                alvo_mask, min_size=GRUPO_MIN, max_size=GRUPO_MAX
            )
            num_grupos_filtrados_final = num_grupos
            contagem_grupos_por_tamanho = contagem_tamanhos
            classification_raster_final_map = np.full(b04_data.shape, OUTPUT_NODATA, dtype=np.uint8)
            classification_raster_final_map[valid_data_mask_pred] = 0
            classification_raster_final_map[final_mask_bool] = CLASSE_ALVO_LABEL
            pixels_alvo_final = int(np.sum(final_mask_bool))
            print(f"Pixels Alvo APÓS filtro (Modelo Final): {pixels_alvo_final}")
            coords_para_shape_final = grupos_coords
        else:
            pixels_alvo_final = pixels_alvo_inicial
            if pixels_alvo_final > 0:
                 coords_alvo_rc_final = np.argwhere(classification_raster_final_map == CLASSE_ALVO_LABEL)
                 if coords_alvo_rc_final.size > 0:
                      coords_para_shape_final = [coords_alvo_rc_final]

        # 5. Exportar Resultados Finais
        out_profile_final_map = out_profile.copy()
        out_profile_final_map.update(count=1, dtype=rasterio.uint8, nodata=OUTPUT_NODATA)
        raster_classificado_path_final = exportar_tif_classificacao(classification_raster_final_map, out_profile_final_map, output_dir_metodo, metodo_id)
        if coords_para_shape_final:
             shapes_path_final, centroid_path_final = exportar_shapes_finais(
                 transform_geo=out_profile_final_map['transform'], crs_geo=out_profile_final_map['crs'],
                 todos_grupos_coords=coords_para_shape_final, output_dir=output_dir_metodo, sufixo_ano_metodo=metodo_id
             )
        else:
            shapes_path_final, centroid_path_final = None, None

    except StopIteration as stop_msg:
        print(f"Processamento {metodo_str_upper} {ano_str} interrompido: {stop_msg}")
    except Exception as e:
        print(f"Erro Crítico no Processamento/Validação {metodo_str_upper} {ano_str}: {e}")
        traceback.print_exc()
    finally:
        print(f"\nEscrevendo sumário {metodo_str_upper} para {ano_str} em: {sumario_path}")
        try:
            path_tif_str = os.path.basename(raster_classificado_path_final) if raster_classificado_path_final else 'N/A'
            path_shp_str = os.path.basename(shapes_path_final) if shapes_path_final else 'N/A'
            path_shp_centroid_str = os.path.basename(centroid_path_final) if centroid_path_final else 'N/A'

            with open(sumario_path, 'w', encoding='utf-8') as f_sum:
                f_sum.write(f"Sumário da Classificação e Validação [{metodo_str_upper}] - Ano {ano_str}\n")
                f_sum.write(f"Processado em: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f_sum.write("="*40 + "\n"); f_sum.write(f"Método: {metodo_str_upper}\n")
                f_sum.write("Pós-processamento:\n")
                if APLICAR_FILTRO_TAMANHO: f_sum.write(f"  - Filtro Tamanho Grupo: SIM (Min: {GRUPO_MIN}, Max: {GRUPO_MAX} pixels)\n")
                else: f_sum.write(f"  - Filtro Tamanho Grupo: NÃO\n")
                f_sum.write("="*40 + "\n"); f_sum.write("Predição e Filtragem (Modelo Final):\n")
                f_sum.write(f"  - Pixels Alvo ANTES do filtro: {pixels_alvo_inicial}\n")
                f_sum.write(f"  - Pixels Alvo FINAIS (pós-filtro): {pixels_alvo_final}\n")
                if APLICAR_FILTRO_TAMANHO:
                    f_sum.write(f"  - Total de Grupos Encontrados (pós-filtro): {num_grupos_filtrados_final}\n")
                    if num_grupos_filtrados_final > 0:
                        detalhes_str = "; ".join([f"{size} pixels: {count} grupo(s)" for size, count in contagem_grupos_por_tamanho.items()])
                        f_sum.write(f"  - Detalhamento por Tamanho: {detalhes_str}\n")
                f_sum.write("="*40 + "\n"); f_sum.write("Resultados Finais Exportados:\n")
                f_sum.write(f"  - Raster Classificado Final: {path_tif_str}\n")
                f_sum.write(f"  - Shapefile Pixels Alvo Finais: {path_shp_str}\n")
                f_sum.write(f"  - Shapefile Centroides Alvo Finais: {path_shp_centroid_str}\n")
        except Exception as e_sum:
            print(f"Erro ao escrever sumário {metodo_str_upper} {ano_str}: {e_sum}")

    return (metricas_validacao, raster_classificado_path_final, (shapes_path_final, centroid_path_final),
            gdf_gt_valid, classification_raster_final_map, out_profile_final_map)

def calcular_metricas(y_true, y_pred, labels):
    if len(y_true) == 0 or len(y_pred) == 0:
        return None
    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
        fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
        tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
        report_text = classification_report(y_true, y_pred, labels=labels, target_names=[f"Classe {l}" for l in labels], zero_division=0)
        report_dict = classification_report(y_true, y_pred, labels=labels, target_names=[f"Classe {l}" for l in labels], zero_division=0, output_dict=True)
        target_label_str = f'Classe {CLASSE_ALVO_LABEL}'
        precision_alvo = report_dict.get(target_label_str, {}).get('precision', 0)
        recall_alvo = report_dict.get(target_label_str, {}).get('recall', 0)
        f1_alvo = report_dict.get(target_label_str, {}).get('f1-score', 0)
        metrics = {
            'pontos_matriz': len(y_true), 'acuracia_geral': accuracy_score(y_true, y_pred),
            'acuracia_balanceada': balanced_accuracy_score(y_true, y_pred, adjusted=False),
            'kappa': cohen_kappa_score(y_true, y_pred, labels=labels),
            'iou_alvo': jaccard_score(y_true, y_pred, pos_label=CLASSE_ALVO_LABEL, average='binary', zero_division=0),
            'precisao_alvo': precision_alvo, 'recall_alvo': recall_alvo, 'f1_alvo': f1_alvo,
            'cm_tn': tn, 'cm_fp': fp, 'cm_fn': fn, 'cm_tp': tp,
            'classification_report_text': report_text
        }
        return metrics
    except Exception as e:
        print(f"Erro ao calcular métricas: {e}"); return None

def escrever_relatorio_validacao(metodo_id, ano_str, metodo_str, shp_gt_path, raster_classificado_path, dir_output_valid, col_classe_gt, metricas, gdf_gt_usado):
    if metricas is None:
        print(f"Métricas inválidas para {metodo_id}. Relatório não gerado.")
        return
    metodo_val_desc = metricas.get('metodo_validacao', 'N/A')
    output_txt = os.path.join(dir_output_valid, f"validacao_{metodo_id}_{VALIDATION_METHOD}.txt")
    with open(output_txt, "w", encoding='utf-8') as f:
        f.write(f"Relatório de Validação [{metodo_str.upper()}] - Ano {ano_str}\n")
        f.write(f"Método de Validação: {metodo_val_desc}\n")
        f.write(f"Total Pontos GT Válidos: {len(gdf_gt_usado) if gdf_gt_usado is not None else 'N/A'}\n")
        f.write(f"Total Pontos na Matriz: {metricas.get('pontos_matriz', 'N/A')}\n")
        f.write("\nMatriz de Confusão:\n")
        cm_tn, cm_fp, cm_fn, cm_tp = metricas.get('cm_tn', 'N/A'), metricas.get('cm_fp', 'N/A'), metricas.get('cm_fn', 'N/A'), metricas.get('cm_tp', 'N/A')
        f.write(f"  TN={cm_tn}, FP={cm_fp}, FN={cm_fn}, TP={cm_tp}\n")
        f.write("\nMétricas Gerais:\n")
        f.write(f"  Acurácia Geral: {metricas.get('acuracia_geral', 'N/A'):.4f}\n")
        f.write(f"  Kappa: {metricas.get('kappa', 'N/A'):.4f}\n")
        f.write(f"\nMétricas Classe Alvo ('{CLASSE_ALVO_NOME}'):\n")
        f.write(f"  IoU (Jaccard): {metricas.get('iou_alvo', 'N/A'):.4f}\n")
        f.write(f"  Precisão: {metricas.get('precisao_alvo', 'N/A'):.4f}\n")
        f.write(f"  Recall: {metricas.get('recall_alvo', 'N/A'):.4f}\n")
        f.write(f"  F1-Score: {metricas.get('f1_alvo', 'N/A'):.4f}\n")
        f.write("\nRelatório Detalhado:\n")
        f.write(metricas.get('classification_report_text', 'N/A'))
    print(f"Relatório de validação salvo: {output_txt}")

# --- FUNÇÃO PARA ESCREVER RELATÓRIO COMPARATIVO ---
def escrever_relatorio_comparativo(resultados_validacao, dir_output_valid):
    """ Escreve um relatório comparando as métricas de validação entre todas as execuções. """
    run_ids = sorted(list(resultados_validacao.keys()))
    if len(run_ids) < 1:
        print("Aviso: Nenhum resultado de validação para gerar relatório comparativo.")
        return
    
    output_txt = os.path.join(dir_output_valid, f"comparacao_validacao_geral_{VALIDATION_METHOD}.txt")
    print(f"\n--- Escrevendo Relatório Comparativo de Validação ({VALIDATION_METHOD}) ---")

    try:
        os.makedirs(dir_output_valid, exist_ok=True)
        with open(output_txt, "w", encoding='utf-8') as f:
            f.write(f"Relatório Comparativo de Validação\n")
            metodo_val = resultados_validacao[run_ids[0]].get('metodo_validacao', f'N/A ({VALIDATION_METHOD})')
            f.write(f"Método de Validação Utilizado: {metodo_val}\n")
            f.write(f"Gerado em: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Shapefile Ground Truth Base: {os.path.basename(GROUND_TRUTH_SHP)}\n")
            f.write(f"!! AVISO: O MESMO conjunto de pontos GT foi usado para validar todos os anos !!\n")
            f.write("="*80 + "\n")
            
            header_line = f"{'Métrica':<25}"
            for run_id in run_ids: header_line += f"| {run_id.upper():^15} "
            f.write(header_line + "\n"); f.write("-" * len(header_line.strip()) + "\n");

            # Função auxiliar corrigida para reconhecer números do NumPy
            def write_metric_line(metric_key, display_name, precision=4):
                line = f"{display_name:<25}"
                for run_id in run_ids:
                    metricas_run = resultados_validacao.get(run_id, {})
                    valor = metricas_run.get(metric_key, 'N/A')
                    # Adicionado np.number ao teste de instância
                    if isinstance(valor, (int, float, np.number)):
                         fmt = f".{precision}f" if precision > 0 else "d"
                         line += f"| {valor:^15{fmt}} "
                    else:
                         line += f"| {'N/A':^15} "
                f.write(line + "\n")

            write_metric_line('pontos_teste', 'Pontos na Validação', 0)
            f.write("-" * len(header_line.strip()) + "\n")
            write_metric_line('acuracia_geral', 'Acurácia Geral')
            write_metric_line('acuracia_balanceada', 'Acurácia Balanceada')
            write_metric_line('kappa', 'Kappa')
            f.write("-" * len(header_line.strip()) + "\n")
            f.write(f"Classe Alvo: {CLASSE_ALVO_NOME} ({CLASSE_ALVO_LABEL})\n")
            write_metric_line('iou_alvo', '  IoU (Jaccard)')
            write_metric_line('precisao_alvo', '  Precisão')
            write_metric_line('recall_alvo', '  Recall (Sensibilidade)')
            write_metric_line('f1_alvo', '  F1-Score')
            f.write("-" * len(header_line.strip()) + "\n")
            f.write("Matriz Confusão (Alvo):\n")
            # Chamadas para Matriz
            write_metric_line('cm_tp', '  Verd. Positivos (TP)', 0)
            write_metric_line('cm_fp', '  Falsos Positivos (FP)', 0)
            write_metric_line('cm_fn', '  Falsos Negativos (FN)', 0)
            write_metric_line('cm_tn', '  Verd. Negativos (TN)', 0)
            f.write("="*80 + "\n")
        print(f"Relatório comparativo de validação salvo: {output_txt}")
    except Exception as e:
        print(f"Erro inesperado ao escrever relatório comparativo: {e}"); traceback.print_exc()

def exportar_shapes_intersecao_grupos(todos_grupos_coords, transform_geo, crs_geo, output_dir, sufixo_metodo):
    """Exporta shapefiles de pixels e centroides para grupos de interseção."""
    print(f"Exportando shapefiles de grupos de interseção para o método {sufixo_metodo}...")
    output_shapefile_pontos = os.path.join(output_dir, f'shapes_intersecao_pixels_{sufixo_metodo}.shp')
    output_shapefile_centroides = os.path.join(output_dir, f'shapes_intersecao_centroides_{sufixo_metodo}.shp')

    if not todos_grupos_coords:
        print("Aviso: Nenhuma coordenada de grupo de interseção fornecida. Shapefiles de interseção não gerados.")
        return

    lista_todos_pontos, lista_centroides = [], []
    ano_intersecao = '2020_2023' # Representa a interseção entre os anos

    for i, grupo_coords_rc in enumerate(todos_grupos_coords):
        group_id = i + 1
        if grupo_coords_rc.size == 0: continue
        xs, ys = rasterio.transform.xy(transform_geo, grupo_coords_rc[:, 0], grupo_coords_rc[:, 1], offset="center")
        
        # Coleta coordenadas para os pixels
        for x, y, (lin, col) in zip(xs, ys, grupo_coords_rc):
            lista_todos_pontos.append({
                "geometry": Point(x, y),
                "metodo": sufixo_metodo,
                "ano": ano_intersecao,
                "group_id": group_id,
                "lin_img": int(lin),
                "col_img": int(col)
            })
            
        # Calcula e coleta o centroide do grupo
        if xs.size > 0:
            lista_centroides.append({
                'geometry': Point(np.mean(xs), np.mean(ys)),
                'metodo': sufixo_metodo,
                'ano': ano_intersecao,
                'group_id': group_id,
                'num_pixels': len(xs)
            })
    
    try:
        valid_crs = crs_geo if crs_geo else None

        if lista_todos_pontos:
            gdf_pontos = gpd.GeoDataFrame(lista_todos_pontos, crs=valid_crs)
            gdf_pontos['x_coord'] = gdf_pontos.geometry.x
            gdf_pontos['y_coord'] = gdf_pontos.geometry.y
            gdf_pontos.to_file(output_shapefile_pontos, driver='ESRI Shapefile', encoding='utf-8')
            print(f"Shapefile de PIXELS de interseção ({len(gdf_pontos)}) exportado: {output_shapefile_pontos}")
        else:
            print(f"Nenhum pixel de interseção encontrado para o método {sufixo_metodo}.")
            
        if lista_centroides:
            gdf_centroides = gpd.GeoDataFrame(lista_centroides, crs=valid_crs)
            gdf_centroides['x_coord'] = gdf_centroides.geometry.x
            gdf_centroides['y_coord'] = gdf_centroides.geometry.y
            gdf_centroides.to_file(output_shapefile_centroides, driver='ESRI Shapefile', encoding='utf-8')
            print(f"Shapefile de CENTROIDES de interseção ({len(gdf_centroides)}) exportado: {output_shapefile_centroides}")
        else:
            print(f"Nenhum centroide de interseção encontrado para o método {sufixo_metodo}.")

    except Exception as e:
        print(f"Erro ao criar/salvar shapefiles de interseção para {sufixo_metodo}: {e}")
        traceback.print_exc()

# --- BLOCO PRINCIPAL DE EXECUÇÃO ---
if __name__ == "__main__":
    output_dir_rf = os.path.join(OUTPUT_DIR_BASE, 'rf_outputs')
    output_dir_svm = os.path.join(OUTPUT_DIR_BASE, 'svm_outputs')
    output_dir_valid = os.path.join(OUTPUT_DIR_BASE, 'validacao')
    os.makedirs(output_dir_rf, exist_ok=True)
    os.makedirs(output_dir_svm, exist_ok=True)
    os.makedirs(output_dir_valid, exist_ok=True)

    start_time = datetime.datetime.now()
    print(f"\nInício: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    anos_para_processar = {'2020': SENTINEL_BANDS_2020, '2023': SENTINEL_BANDS_2023 }
    resultados_validacao_geral = {}
    resultados_rasters_finais = {}
    resultados_perfis_finais = {}

    for ano, bandas in anos_para_processar.items():
        for metodo in METHODS_TO_PROCESS:
            print(f"\n{'='*15} PROCESSANDO ANO: {ano} | MÉTODO: {metodo.upper()} {'='*15}")
            output_dir_atual = output_dir_rf if metodo.upper() == 'RF' else output_dir_svm
            run_id = f"{ano}_{metodo.lower()}"
            try:
                 metricas, raster_path, shapes_paths, gdf_gt, raster_array, raster_profile = processar_ano_metodo(
                      ano, metodo, bandas, STUDY_AREA_SHP, GROUND_TRUTH_SHP, output_dir_atual, COLUNA_CLASSE_REF
                  )
                 if metricas:
                     resultados_validacao_geral[run_id] = metricas
                     escrever_relatorio_validacao(run_id, ano, metodo, GROUND_TRUTH_SHP, raster_path, output_dir_valid, COLUNA_CLASSE_REF, metricas, gdf_gt)
                 if raster_array is not None: resultados_rasters_finais[run_id] = raster_array
                 if raster_profile is not None: resultados_perfis_finais[run_id] = raster_profile
                 
            except Exception as e_proc:
                 print(f"!!!!!! FALHA GERAL NO PROCESSAMENTO PARA {run_id}: {e_proc} !!!!!!"); traceback.print_exc()

    if resultados_validacao_geral:
        escrever_relatorio_comparativo(resultados_validacao_geral, output_dir_valid)

    print("\n--- Calculando e Exportando Interseção dos Resultados ---")
    for metodo in METHODS_TO_PROCESS:
        metodo_lower = metodo.lower()
        print(f"\n-- Verificando interseção para o método: {metodo.upper()} --")
        run_id_2020, run_id_2023 = f'2020_{metodo_lower}', f'2023_{metodo_lower}'
        if run_id_2020 in resultados_rasters_finais and run_id_2023 in resultados_rasters_finais:
            raster_2020, raster_2023 = resultados_rasters_finais[run_id_2020], resultados_rasters_finais[run_id_2023]
            profile_2020, profile_2023 = resultados_perfis_finais[run_id_2020], resultados_perfis_finais[run_id_2023]
            
            if raster_2020 is not None and raster_2023 is not None and profile_2020 is not None and profile_2023 is not None:
                intersection_mask = (raster_2020 == CLASSE_ALVO_LABEL) & (raster_2023 == CLASSE_ALVO_LABEL)
                num_pixels_intersecao = np.sum(intersection_mask)
                print(f"  Pixels '{CLASSE_ALVO_NOME}' em ambos os anos ({metodo.upper()}): {num_pixels_intersecao}")
                if num_pixels_intersecao > 0:
                    _, grupos_coords_intersecao, num_grupos, _ = identificar_grupos_conectados_mask(intersection_mask)
                    print(f"  Encontrados {num_grupos} grupos de interseção.")
                    if num_grupos > 0:
                        output_dir_intersecao = output_dir_rf if metodo.upper() == 'RF' else output_dir_svm
                        exportar_shapes_intersecao_grupos(
                            todos_grupos_coords=grupos_coords_intersecao,
                            transform_geo=profile_2023['transform'], crs_geo=profile_2023['crs'],
                            output_dir=output_dir_intersecao, sufixo_metodo=metodo_lower
                        )
        else:
            print(f"Não foi possível calcular a interseção para {metodo.upper()}.")

    end_time = datetime.datetime.now()
    print(f"\n--- Script Totalmente Concluído ---")
    print(f"Tempo total de execução: {end_time - start_time}")
