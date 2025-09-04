# -*- coding: utf-8 -*-
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4' 

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import geometry_mask, geometry_window
from scipy.ndimage import label
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import warnings
import datetime
import traceback
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (confusion_matrix, accuracy_score, cohen_kappa_score,
                           classification_report, balanced_accuracy_score,
                           jaccard_score, precision_score, recall_score, f1_score)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


try:
    import pytz
except ImportError:
    pytz = None

# --- CONSTANTES ---

METHODS_TO_PROCESS = ['RF', 'SVM', 'KNN', 'VOTING', 'STACKING']
VALIDATION_METHOD = 'K_FOLD'
HOLD_OUT_TEST_SIZE = 0.3
K_FOLD_N_SPLITS = 5

SENTINEL_BANDS_2020 = {
    'B02': r"C:\mest3\2020\R10m\T24KVG_20201007T125311_B02_10m.jp2",
    'B03': r"C:\mest3\2020\R10m\T24KVG_20201007T125311_B03_10m.jp2",
    'B04': r"C:\mest3\2020\R10m\T24KVG_20201007T125311_B04_10m.jp2",
    'B08': r"C:\mest3\2020\R10m\T24KVG_20201007T125311_B08_10m.jp2"
}
SENTINEL_BANDS_2023 = {
    'B02': r"C:\mest3\2023\R10m\T24KVG_20230927T125309_B02_10m.jp2",
    'B03': r"C:\mest3\2023\R10m\T24KVG_20230927T125309_B03_10m.jp2",
    'B04': r"C:\mest3\2023\R10m\T24KVG_20230927T125309_B04_10m.jp2",
    'B08': r"C:\mest3\2023\R10m\T24KVG_20230927T125309_B08_10m.jp2"
}

STUDY_AREA_SHP = r"C:\mest3\limite\limite.shp"
OUTPUT_DIR_BASE = r'C:\mest3\resultados_comparativos_filtrado'
GROUND_TRUTH_SHP = r'C:\mest3\verdade de campo\pontos.shp'

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
SVM_PROBABILITY = True 

KNN_APPLY_SCALING = True
KNN_N_NEIGHBORS = 5 # Número de vizinhos a considerar
KNN_WEIGHTS = 'distance' # 'uniform' para pesos iguais, 'distance' para pesos por distância
KNN_METRIC = 'euclidean'

# --- Parâmetros do Voting Classifier ---
VOTING_TYPE = 'soft'
VOTING_WEIGHTS = [0.4, 0.6] # Pesos para [RF, SVM]

# --- Parâmetros do Stacking Classifier ---
STACKING_FINAL_ESTIMATOR = LogisticRegression()

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

def escrever_relatorio_validacao(metodo_id, anos_str, metodo_str, shp_gt_path, dir_output_valid, col_classe_gt, metricas, gdf_gt_usado):
    if metricas is None:
        print(f"Métricas inválidas para {metodo_id}. Relatório não gerado.")
        return
    metodo_val_desc = metricas.get('metodo_validacao', 'N/A')
    output_txt = os.path.join(dir_output_valid, f"validacao_{metodo_id}_{VALIDATION_METHOD}.txt")
    with open(output_txt, "w", encoding='utf-8') as f:
        f.write(f"Relatório de Validação [{metodo_str.upper()}]\n")
        f.write(f"Dados de treinamento/validação dos anos: {anos_str}\n")
        f.write(f"Método de Validação: {metodo_val_desc}\n")
        f.write(f"Total Pontos GT Válidos (combinados): {len(gdf_gt_usado) if gdf_gt_usado is not None else 'N/A'}\n")
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
            f.write(f"!! AVISO: A validação foi feita em um dataset COMBINADO de múltiplos anos !!\n")
            f.write("="*80 + "\n")
            
            header_line = f"{'Métrica':<25}"
            for run_id in run_ids: header_line += f"| {run_id.upper():^15} "
            f.write(header_line + "\n"); f.write("-" * len(header_line.strip()) + "\n");

            def write_metric_line(metric_key, display_name, precision=4):
                line = f"{display_name:<25}"
                for run_id in run_ids:
                    metricas_run = resultados_validacao.get(run_id, {})
                    valor = metricas_run.get(metric_key, 'N/A')
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
    ano_intersecao = '2020_2023' # Interseção entre os anos

    for i, grupo_coords_rc in enumerate(todos_grupos_coords):
        group_id = i + 1
        if grupo_coords_rc.size == 0: continue
        xs, ys = rasterio.transform.xy(transform_geo, grupo_coords_rc[:, 0], grupo_coords_rc[:, 1], offset="center")
        
        for x, y, (lin, col) in zip(xs, ys, grupo_coords_rc):
            lista_todos_pontos.append({
                "geometry": Point(x, y), "metodo": sufixo_metodo, "ano": ano_intersecao,
                "group_id": group_id, "lin_img": int(lin), "col_img": int(col)
            })
            
        if xs.size > 0:
            lista_centroides.append({
                'geometry': Point(np.mean(xs), np.mean(ys)), 'metodo': sufixo_metodo,
                'ano': ano_intersecao, 'group_id': group_id, 'num_pixels': len(xs)
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


# --- FUNÇÕES DE LÓGICA PRINCIPAL ---

def coletar_dados_treinamento_combinado(anos_para_processar, shape_limite_path, shp_ground_truth_path, col_classe_gt):
    """
    Coleta dados de treinamento de múltiplos anos e os combina em um único dataset.
    """
    print("\n--- Iniciando Coleta de Dados de Treinamento Combinado ---")
    X_all_list, y_all = [], []
    gdf_gt_valid_global = None

    print("Lendo shapefile de limite e ground truth...")
    limite_gdf = gpd.read_file(shape_limite_path)
    gdf_gt = gpd.read_file(shp_ground_truth_path)
    if limite_gdf.empty or gdf_gt.empty:
        raise ValueError("Shapefile de limite ou Ground Truth vazio.")

    ref_ano = list(anos_para_processar.keys())[0]
    ref_raster_path = anos_para_processar[ref_ano]['B04']
    with rasterio.open(ref_raster_path) as src_ref:
        raster_crs = src_ref.crs
        limite_gdf = limite_gdf.to_crs(raster_crs)
        gdf_gt = gdf_gt.to_crs(raster_crs)
        limite_geom = limite_gdf.geometry.tolist()
        out_window = geometry_window(src_ref, limite_geom, pad_x=1, pad_y=1)
        out_window = out_window.intersection(Window(0, 0, src_ref.width, src_ref.height))
        out_transform = rasterio.windows.transform(out_window, src_ref.transform)

    coords_gt_xy = [(p.x, p.y) for p in gdf_gt.geometry]
    rows_gt, cols_gt = rasterio.transform.rowcol(out_transform, [c[0] for c in coords_gt_xy], [c[1] for c in coords_gt_xy])
    
    classe_map = {CLASSE_ALVO_NOME: CLASSE_ALVO_LABEL}
    y_labels_potenciais = gdf_gt[col_classe_gt].str.strip().map(classe_map).fillna(0).astype(int)

    for ano_str, bandas_paths in anos_para_processar.items():
        print(f"\nExtraindo amostras para o ano: {ano_str}")
        try:
            sources = {band: rasterio.open(path) for band, path in bandas_paths.items()}
            band_data = {band: src.read(1, window=out_window, boundless=True, fill_value=INTERNAL_NODATA) for band, src in sources.items()}
            b02, b03, b04, b08 = band_data['B02'], band_data['B03'], band_data['B04'], band_data['B08']
        finally:
            for src in sources.values():
                if src and not src.closed: src.close()

        valid_indices_ano, X_ano_list, y_ano_list = [], [], []
        for i, (r, c) in enumerate(zip(rows_gt, cols_gt)):
            if 0 <= r < b04.shape[0] and 0 <= c < b04.shape[1]:
                band_values = [b02[r, c], b03[r, c], b04[r, c], b08[r, c]]
                if all(val != INTERNAL_NODATA for val in band_values):
                    X_ano_list.append(band_values)
                    y_ano_list.append(y_labels_potenciais[i])
        
        print(f"  Amostras válidas encontradas em {ano_str}: {len(X_ano_list)}")
        X_all_list.extend(X_ano_list)
        if not y_all:
            y_all.extend(y_ano_list)
        else:
             y_all.extend(y_ano_list)

    if not X_all_list:
        raise ValueError("Nenhuma amostra de treinamento válida encontrada em nenhum dos anos.")

    X_all_final = np.array(X_all_list)
    y_all_final = np.array(y_all)
    
    print(f"\nTotal de amostras combinadas: {X_all_final.shape[0]}")
    print(f"Distribuição de classes (0=NãoAlvo, 1=Alvo): {dict(zip(*np.unique(y_all_final, return_counts=True)))}")

    return X_all_final, y_all_final, gdf_gt

def treinar_e_validar_modelo(X_all, y_all, metodo_str, gdf_gt_usado):
    """
    Treina e valida um modelo com o dataset combinado.
    """
    metodo_str_upper = metodo_str.upper()
    print(f"\n--- Iniciando Treinamento e Validação [{metodo_str_upper}] ({VALIDATION_METHOD}) ---")
    
    metricas_validacao = None
    model_final = None

    if VALIDATION_METHOD == 'K_FOLD':
        print(f"Executando Validação K-FOLD ({K_FOLD_N_SPLITS} folds)...")
        if X_all.shape[0] < K_FOLD_N_SPLITS:
            raise ValueError(f"Não há dados suficientes ({X_all.shape[0]}) para {K_FOLD_N_SPLITS}-Fold CV.")
        
        skf = StratifiedKFold(n_splits=K_FOLD_N_SPLITS, shuffle=True, random_state=RF_RANDOM_STATE)
        y_true_all_folds, y_pred_all_folds = [], []

        for fold_num, (train_index, test_index) in enumerate(skf.split(X_all, y_all), 1):
            X_train_fold, X_test_fold = X_all[train_index], X_all[test_index]
            y_train_fold, y_test_fold = y_all[train_index], y_all[test_index]
            
            model_fold = None
            if metodo_str_upper == 'RF':
                model_fold = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE, class_weight=RF_CLASS_WEIGHT, n_jobs=-1)
                model_fold.fit(X_train_fold, y_train_fold)
                y_pred_fold = model_fold.predict(X_test_fold)

            elif metodo_str_upper == 'SVM':
                model_fold = Pipeline([
                    ('scaler', StandardScaler()),
                    ('svm', SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, class_weight=SVM_CLASS_WEIGHT, random_state=SVM_RANDOM_STATE, probability=SVM_PROBABILITY))
                ])
                model_fold.fit(X_train_fold, y_train_fold)
                y_pred_fold = model_fold.predict(X_test_fold)

            elif metodo_str_upper == 'KNN':
                model_fold = Pipeline([
                    ('scaler', StandardScaler()),
                    ('knn', KNeighborsClassifier(n_neighbors=KNN_N_NEIGHBORS, weights=KNN_WEIGHTS, metric=KNN_METRIC))
                ])
                model_fold.fit(X_train_fold, y_train_fold)
                y_pred_fold = model_fold.predict(X_test_fold)

            elif metodo_str_upper == 'VOTING':
                clf1 = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE, class_weight=RF_CLASS_WEIGHT, n_jobs=-1)
                clf2 = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, class_weight=SVM_CLASS_WEIGHT, random_state=SVM_RANDOM_STATE, probability=SVM_PROBABILITY))])
                model_fold = VotingClassifier(estimators=[('rf', clf1), ('svm', clf2)], voting=VOTING_TYPE, weights=VOTING_WEIGHTS if VOTING_TYPE == 'soft' else None, n_jobs=-1)
                model_fold.fit(X_train_fold, y_train_fold)
                y_pred_fold = model_fold.predict(X_test_fold)
            
            elif metodo_str_upper == 'STACKING':
                estimators = [
                    ('rf', RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE, class_weight=RF_CLASS_WEIGHT, n_jobs=-1)),
                    ('svm', Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, class_weight=SVM_CLASS_WEIGHT, random_state=SVM_RANDOM_STATE, probability=SVM_PROBABILITY))]))
                ]
                model_fold = StackingClassifier(estimators=estimators, final_estimator=STACKING_FINAL_ESTIMATOR, cv=3)
                model_fold.fit(X_train_fold, y_train_fold)
                y_pred_fold = model_fold.predict(X_test_fold)

            y_true_all_folds.extend(y_test_fold)
            y_pred_all_folds.extend(y_pred_fold)

        print("Calculando métricas agregadas da Validação Cruzada K-Fold...")
        metricas_validacao = calcular_metricas(y_true_all_folds, y_pred_all_folds, CLASSES_LABELS)
        if metricas_validacao:
            metricas_validacao['metodo_validacao'] = f'{K_FOLD_N_SPLITS}-Fold Cross-Validation'
            metricas_validacao['pontos_teste'] = len(y_true_all_folds)

    # Treinamento do modelo final
    print(f"\nTreinando modelo {metodo_str_upper} final com TODOS os dados combinados...")
    if metodo_str_upper == 'RF':
        model_final = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE, class_weight=RF_CLASS_WEIGHT, n_jobs=-1)
        model_final.fit(X_all, y_all)
    
    elif metodo_str_upper == 'SVM':
        model_final = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, class_weight=SVM_CLASS_WEIGHT, random_state=SVM_RANDOM_STATE, probability=SVM_PROBABILITY))])
        model_final.fit(X_all, y_all)

    elif metodo_str_upper == 'KNN':
        model_final = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=KNN_N_NEIGHBORS, weights=KNN_WEIGHTS, metric=KNN_METRIC))
        ])
        model_final.fit(X_all, y_all)

    elif metodo_str_upper == 'VOTING':
        clf1 = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE, class_weight=RF_CLASS_WEIGHT, n_jobs=-1)
        clf2 = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, class_weight=SVM_CLASS_WEIGHT, random_state=SVM_RANDOM_STATE, probability=SVM_PROBABILITY))])
        model_final = VotingClassifier(estimators=[('rf', clf1), ('svm', clf2)], voting=VOTING_TYPE, weights=VOTING_WEIGHTS if VOTING_TYPE == 'soft' else None, n_jobs=-1)
        model_final.fit(X_all, y_all)
    
    elif metodo_str_upper == 'STACKING':
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE, class_weight=RF_CLASS_WEIGHT, n_jobs=-1)),
            ('svm', Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, class_weight=SVM_CLASS_WEIGHT, random_state=SVM_RANDOM_STATE, probability=SVM_PROBABILITY))]))
        ]
        model_final = StackingClassifier(estimators=estimators, final_estimator=STACKING_FINAL_ESTIMATOR, cv=5)
        model_final.fit(X_all, y_all)

    if model_final is None: raise ValueError(f"Método '{metodo_str_upper}' inválido.")
    model_final.scaler = None
    
    return model_final, metricas_validacao

def aplicar_modelo_e_exportar_resultados(ano_str, metodo_str, bandas_paths, shape_limite_path, model, output_dir_metodo):
    """
    Aplica um modelo já treinado a uma imagem de um ano específico e exporta os resultados.
    """
    print(f"\n--- Aplicando Modelo [{metodo_str.upper()}] para o Ano {ano_str} ---")
    metodo_id = f"{ano_str}_{metodo_str.lower()}"
    sumario_path = os.path.join(output_dir_metodo, f'sumario_{metodo_id}.txt')
    
    raster_classificado_path, shapes_path, centroid_path = None, None, None
    classification_raster_final, out_profile_final = None, None
    pixels_alvo_inicial, pixels_alvo_final = 0, 0
    num_grupos_filtrados = 0
    contagem_grupos_por_tamanho = {}

    try:
        limite_gdf = gpd.read_file(shape_limite_path)
        ref_raster_path = bandas_paths['B04']
        with rasterio.open(ref_raster_path) as src_ref:
            raster_crs = src_ref.crs
            limite_gdf = limite_gdf.to_crs(raster_crs)
            limite_geom = limite_gdf.geometry.tolist()
            out_window = geometry_window(src_ref, limite_geom, pad_x=1, pad_y=1)
            out_window = out_window.intersection(Window(0, 0, src_ref.width, src_ref.height))
            out_transform = rasterio.windows.transform(out_window, src_ref.transform)
            out_profile = src_ref.profile.copy()

        sources = {band: rasterio.open(path) for band, path in bandas_paths.items()}
        band_data = {band: src.read(1, window=out_window, boundless=True, fill_value=INTERNAL_NODATA) for band, src in sources.items()}
        for src in sources.values(): src.close()
        b02, b03, b04, b08 = band_data['B02'], band_data['B03'], band_data['B04'], band_data['B08']
        
        mask_geom = geometry_mask(limite_geom, out_shape=b04.shape, transform=out_transform, invert=True, all_touched=True)
        valid_data_mask_pred = mask_geom & (b02 != INTERNAL_NODATA) & (b03 != INTERNAL_NODATA) & (b04 != INTERNAL_NODATA) & (b08 != INTERNAL_NODATA)

        print("Gerando mapa de classificação...")
        rows_pred, cols_pred = np.where(valid_data_mask_pred)
        X_pred_map = np.vstack([b02[rows_pred, cols_pred], b03[rows_pred, cols_pred], b04[rows_pred, cols_pred], b08[rows_pred, cols_pred]]).T
        
        predictions_map = model.predict(X_pred_map)

        classification_raster_initial = np.full(b04.shape, OUTPUT_NODATA, dtype=np.uint8)
        classification_raster_initial[valid_data_mask_pred] = 0
        classification_raster_initial[rows_pred, cols_pred] = predictions_map.astype(np.uint8)
        pixels_alvo_inicial = int(np.sum(classification_raster_initial == CLASSE_ALVO_LABEL))
        print(f"Pixels Alvo ANTES do filtro: {pixels_alvo_inicial}")

        coords_para_shape = []
        classification_raster_final = classification_raster_initial.copy()
        if APLICAR_FILTRO_TAMANHO and pixels_alvo_inicial > 0:
            alvo_mask = (classification_raster_initial == CLASSE_ALVO_LABEL)
            final_mask_bool, grupos_coords, num_grupos, contagem_tamanhos = identificar_grupos_conectados_mask(alvo_mask, min_size=GRUPO_MIN, max_size=GRUPO_MAX)
            num_grupos_filtrados = num_grupos
            contagem_grupos_por_tamanho = contagem_tamanhos
            classification_raster_final = np.full(b04.shape, OUTPUT_NODATA, dtype=np.uint8)
            classification_raster_final[valid_data_mask_pred] = 0
            classification_raster_final[final_mask_bool] = CLASSE_ALVO_LABEL
            pixels_alvo_final = int(np.sum(final_mask_bool))
            print(f"Pixels Alvo APÓS filtro: {pixels_alvo_final}")
            coords_para_shape = grupos_coords
        else:
            pixels_alvo_final = pixels_alvo_inicial
            if pixels_alvo_final > 0:
                coords_alvo_rc = np.argwhere(classification_raster_final == CLASSE_ALVO_LABEL)
                if coords_alvo_rc.size > 0: coords_para_shape = [coords_alvo_rc]

        out_profile_final = out_profile.copy()
        out_profile_final.update(height=out_window.height, width=out_window.width, transform=out_transform, count=1, dtype=rasterio.uint8, nodata=OUTPUT_NODATA)
        
        raster_classificado_path = exportar_tif_classificacao(classification_raster_final, out_profile_final, output_dir_metodo, metodo_id)
        if coords_para_shape:
            shapes_path, centroid_path = exportar_shapes_finais(transform_geo=out_profile_final['transform'], crs_geo=out_profile_final['crs'], todos_grupos_coords=coords_para_shape, output_dir=output_dir_metodo, sufixo_ano_metodo=metodo_id)
    
    except Exception as e:
        print(f"Erro Crítico ao aplicar modelo para {metodo_id}: {e}")
        traceback.print_exc()

    finally:
        print(f"Escrevendo sumário para {metodo_id} em: {sumario_path}")
        path_tif_str = os.path.basename(raster_classificado_path) if raster_classificado_path else 'N/A'
        path_shp_str = os.path.basename(shapes_path) if shapes_path else 'N/A'
        path_shp_centroid_str = os.path.basename(centroid_path) if centroid_path else 'N/A'
        with open(sumario_path, 'w', encoding='utf-8') as f_sum:
            f_sum.write(f"Sumário da Classificação [{metodo_str.upper()}] - Ano {ano_str}\n")
            f_sum.write(f"Processado em: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f_sum.write("="*40 + "\n"); f_sum.write(f"Método: {metodo_str.upper()}\n")
            if metodo_str.upper() == 'VOTING':
                f_sum.write(f"  - Tipo de Votação: {VOTING_TYPE}\n")
                if VOTING_TYPE == 'soft': f_sum.write(f"  - Pesos (RF, SVM): {VOTING_WEIGHTS}\n")
            elif metodo_str.upper() == 'STACKING':
                f_sum.write(f"  - Meta-Modelo: Regressão Logística\n")
            f_sum.write("Pós-processamento:\n")
            if APLICAR_FILTRO_TAMANHO: f_sum.write(f"  - Filtro Tamanho Grupo: SIM (Min: {GRUPO_MIN}, Max: {GRUPO_MAX} pixels)\n")
            else: f_sum.write(f"  - Filtro Tamanho Grupo: NÃO\n")
            f_sum.write("="*40 + "\n"); f_sum.write("Predição e Filtragem:\n")
            f_sum.write(f"  - Pixels Alvo ANTES do filtro: {pixels_alvo_inicial}\n")
            f_sum.write(f"  - Pixels Alvo FINAIS (pós-filtro): {pixels_alvo_final}\n")
            if APLICAR_FILTRO_TAMANHO:
                f_sum.write(f"  - Total de Grupos Encontrados (pós-filtro): {num_grupos_filtrados}\n")
                if num_grupos_filtrados > 0:
                    detalhes_str = "; ".join([f"{size} pixels: {count} grupo(s)" for size, count in contagem_grupos_por_tamanho.items()])
                    f_sum.write(f"  - Detalhamento por Tamanho: {detalhes_str}\n")
            f_sum.write("="*40 + "\n"); f_sum.write("Resultados Exportados:\n")
            f_sum.write(f"  - Raster Classificado: {path_tif_str}\n")
            f_sum.write(f"  - Shapefile Pixels Alvo: {path_shp_str}\n")
            f_sum.write(f"  - Shapefile Centroides Alvo: {path_shp_centroid_str}\n")

    return classification_raster_final, out_profile_final


# --- BLOCO PRINCIPAL DE EXECUÇÃO ---
if __name__ == "__main__":
    output_dir_rf = os.path.join(OUTPUT_DIR_BASE, 'rf_outputs')
    output_dir_svm = os.path.join(OUTPUT_DIR_BASE, 'svm_outputs')
    output_dir_knn = os.path.join(OUTPUT_DIR_BASE, 'knn_outputs') # Alterado de kmn para knn
    output_dir_voting = os.path.join(OUTPUT_DIR_BASE, 'voting_outputs')
    output_dir_stacking = os.path.join(OUTPUT_DIR_BASE, 'stacking_outputs')
    output_dir_valid = os.path.join(OUTPUT_DIR_BASE, 'validacao')
    os.makedirs(output_dir_rf, exist_ok=True)
    os.makedirs(output_dir_svm, exist_ok=True)
    os.makedirs(output_dir_knn, exist_ok=True) # Alterado de kmn para knn
    os.makedirs(output_dir_voting, exist_ok=True)
    os.makedirs(output_dir_stacking, exist_ok=True)
    os.makedirs(output_dir_valid, exist_ok=True)

    start_time = datetime.datetime.now()
    print(f"\nInício: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    anos_para_processar = {'2020': SENTINEL_BANDS_2020, '2023': SENTINEL_BANDS_2023}
    resultados_validacao_geral = {}
    resultados_rasters_finais = {}
    resultados_perfis_finais = {}

    try:
        X_combinado, y_combinado, gdf_gt_completo = coletar_dados_treinamento_combinado(anos_para_processar, STUDY_AREA_SHP, GROUND_TRUTH_SHP, COLUNA_CLASSE_REF)
        
        for metodo in METHODS_TO_PROCESS:
            print(f"\n{'='*15} PROCESSANDO MÉTODO: {metodo.upper()} {'='*15}")
            
            output_dir_atual = None
            if metodo.upper() == 'RF': output_dir_atual = output_dir_rf
            elif metodo.upper() == 'SVM': output_dir_atual = output_dir_svm
            elif metodo.upper() == 'KNN': output_dir_atual = output_dir_knn # Alterado de kmn para knn
            elif metodo.upper() == 'VOTING': output_dir_atual = output_dir_voting
            elif metodo.upper() == 'STACKING': output_dir_atual = output_dir_stacking
            else: continue
            
            modelo_final, metricas = treinar_e_validar_modelo(X_combinado, y_combinado, metodo, gdf_gt_completo)

            if metricas:
                resultados_validacao_geral[f"2020_{metodo.lower()}"] = metricas
                resultados_validacao_geral[f"2023_{metodo.lower()}"] = metricas
                escrever_relatorio_validacao(metodo_id=metodo.lower(), anos_str=', '.join(anos_para_processar.keys()), metodo_str=metodo, shp_gt_path=GROUND_TRUTH_SHP, dir_output_valid=output_dir_valid, col_classe_gt=COLUNA_CLASSE_REF, metricas=metricas, gdf_gt_usado=gdf_gt_completo)

            for ano, bandas in anos_para_processar.items():
                run_id = f"{ano}_{metodo.lower()}"
                raster_array, raster_profile = aplicar_modelo_e_exportar_resultados(ano, metodo, bandas, STUDY_AREA_SHP, modelo_final, output_dir_atual)
                if raster_array is not None: resultados_rasters_finais[run_id] = raster_array
                if raster_profile is not None: resultados_perfis_finais[run_id] = raster_profile

    except Exception as e_proc:
        print(f"!!!!!! FALHA GERAL NO PROCESSAMENTO: {e_proc} !!!!!!"); traceback.print_exc()

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
                        output_dir_intersecao = None
                        if metodo.upper() == 'RF': output_dir_intersecao = output_dir_rf
                        elif metodo.upper() == 'SVM': output_dir_intersecao = output_dir_svm
                        elif metodo.upper() == 'KNN': output_dir_intersecao = output_dir_knn 
                        elif metodo.upper() == 'VOTING': output_dir_intersecao = output_dir_voting
                        elif metodo.upper() == 'STACKING': output_dir_intersecao = output_dir_stacking
                        
                        if output_dir_intersecao:
                            exportar_shapes_intersecao_grupos(todos_grupos_coords=grupos_coords_intersecao, transform_geo=profile_2023['transform'], crs_geo=profile_2023['crs'], output_dir=output_dir_intersecao, sufixo_metodo=metodo_lower)
        else:
            print(f"Não foi possível calcular a interseção para {metodo.upper()}.")

    end_time = datetime.datetime.now()
    print(f"\n--- Script Totalmente Concluído ---")
    print(f"Tempo total de execução: {end_time - start_time}")
