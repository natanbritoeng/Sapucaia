Python 3.13.2 (tags/v3.13.2:4f8bb39, Feb  4 2025, 15:23:48) [MSC v.1942 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> 
===================== RESTART: E:\mestrado\mest3\codigo.py =====================

Início: 2025-08-27 10:09:22

--- Iniciando Coleta de Dados de Treinamento Combinado ---
Lendo shapefile de limite e ground truth...

Extraindo amostras para o ano: 2020
  Amostras válidas encontradas em 2020: 300

Extraindo amostras para o ano: 2023
  Amostras válidas encontradas em 2023: 300

Total de amostras combinadas: 600
Distribuição de classes (0=NãoAlvo, 1=Alvo): {np.int64(0): np.int64(400), np.int64(1): np.int64(200)}

=============== PROCESSANDO MÉTODO: RF ===============

--- Iniciando Treinamento e Validação [RF] (K_FOLD) ---
Executando Validação K-FOLD (5 folds)...
Calculando métricas agregadas da Validação Cruzada K-Fold...

Treinando modelo RF final com TODOS os dados combinados...
Relatório de validação salvo: E:\mestrado\mest3\resultados_comparativos_filtrado\validacao\validacao_rf_K_FOLD.txt

--- Aplicando Modelo [RF] para o Ano 2020 ---
Gerando mapa de classificação...
Pixels Alvo ANTES do filtro: 158251
Filtro de tamanho encontrou 5517 grupos. Detalhes (tamanho: contagem): {3: 3628, 4: 1889}
Pixels Alvo APÓS filtro: 18440
Exportando TIF de classificação para 2020_rf (18440 pixels alvo)...
TIF de classificação 2020_rf exportado.
Exportando shapefiles para 2020_rf...
Processando 5517 grupo(s) de coordenadas para gerar centroides individuais...
Shapefile de PIXELS (18440 pontos) exportado: E:\mestrado\mest3\resultados_comparativos_filtrado\rf_outputs\shapes_alvo_pixels_2020_rf.shp
Shapefile de CENTROIDES (5517 grupos) exportado: E:\mestrado\mest3\resultados_comparativos_filtrado\rf_outputs\shapes_alvo_centroides_2020_rf.shp
Escrevendo sumário para 2020_rf em: E:\mestrado\mest3\resultados_comparativos_filtrado\rf_outputs\sumario_2020_rf.txt

--- Aplicando Modelo [RF] para o Ano 2023 ---
Gerando mapa de classificação...
Pixels Alvo ANTES do filtro: 228613
Filtro de tamanho encontrou 10182 grupos. Detalhes (tamanho: contagem): {3: 6764, 4: 3418}
Pixels Alvo APÓS filtro: 33964
Exportando TIF de classificação para 2023_rf (33964 pixels alvo)...
TIF de classificação 2023_rf exportado.
Exportando shapefiles para 2023_rf...
Processando 10182 grupo(s) de coordenadas para gerar centroides individuais...
Shapefile de PIXELS (33964 pontos) exportado: E:\mestrado\mest3\resultados_comparativos_filtrado\rf_outputs\shapes_alvo_pixels_2023_rf.shp
Shapefile de CENTROIDES (10182 grupos) exportado: E:\mestrado\mest3\resultados_comparativos_filtrado\rf_outputs\shapes_alvo_centroides_2023_rf.shp
Escrevendo sumário para 2023_rf em: E:\mestrado\mest3\resultados_comparativos_filtrado\rf_outputs\sumario_2023_rf.txt

=============== PROCESSANDO MÉTODO: SVM ===============

--- Iniciando Treinamento e Validação [SVM] (K_FOLD) ---
Executando Validação K-FOLD (5 folds)...
Calculando métricas agregadas da Validação Cruzada K-Fold...

Treinando modelo SVM final com TODOS os dados combinados...
Relatório de validação salvo: E:\mestrado\mest3\resultados_comparativos_filtrado\validacao\validacao_svm_K_FOLD.txt

--- Aplicando Modelo [SVM] para o Ano 2020 ---
Gerando mapa de classificação...
Pixels Alvo ANTES do filtro: 95153
Filtro de tamanho encontrou 2646 grupos. Detalhes (tamanho: contagem): {3: 1710, 4: 936}
Pixels Alvo APÓS filtro: 8874
Exportando TIF de classificação para 2020_svm (8874 pixels alvo)...
TIF de classificação 2020_svm exportado.
Exportando shapefiles para 2020_svm...
Processando 2646 grupo(s) de coordenadas para gerar centroides individuais...
Shapefile de PIXELS (8874 pontos) exportado: E:\mestrado\mest3\resultados_comparativos_filtrado\svm_outputs\shapes_alvo_pixels_2020_svm.shp
Shapefile de CENTROIDES (2646 grupos) exportado: E:\mestrado\mest3\resultados_comparativos_filtrado\svm_outputs\shapes_alvo_centroides_2020_svm.shp
Escrevendo sumário para 2020_svm em: E:\mestrado\mest3\resultados_comparativos_filtrado\svm_outputs\sumario_2020_svm.txt

--- Aplicando Modelo [SVM] para o Ano 2023 ---
Gerando mapa de classificação...
Pixels Alvo ANTES do filtro: 200326
Filtro de tamanho encontrou 6014 grupos. Detalhes (tamanho: contagem): {3: 3904, 4: 2110}
Pixels Alvo APÓS filtro: 20152
Exportando TIF de classificação para 2023_svm (20152 pixels alvo)...
TIF de classificação 2023_svm exportado.
Exportando shapefiles para 2023_svm...
Processando 6014 grupo(s) de coordenadas para gerar centroides individuais...
Shapefile de PIXELS (20152 pontos) exportado: E:\mestrado\mest3\resultados_comparativos_filtrado\svm_outputs\shapes_alvo_pixels_2023_svm.shp
Shapefile de CENTROIDES (6014 grupos) exportado: E:\mestrado\mest3\resultados_comparativos_filtrado\svm_outputs\shapes_alvo_centroides_2023_svm.shp
Escrevendo sumário para 2023_svm em: E:\mestrado\mest3\resultados_comparativos_filtrado\svm_outputs\sumario_2023_svm.txt

--- Escrevendo Relatório Comparativo de Validação (K_FOLD) ---
Relatório comparativo de validação salvo: E:\mestrado\mest3\resultados_comparativos_filtrado\validacao\comparacao_validacao_geral_K_FOLD.txt

--- Calculando e Exportando Interseção dos Resultados ---

-- Verificando interseção para o método: RF --
  Pixels 'Sapucaia' em ambos os anos (RF): 612
Filtro de tamanho encontrou 175 grupos. Detalhes (tamanho: contagem): {2: 112, 3: 55, 4: 8}
  Encontrados 175 grupos de interseção.
Exportando shapefiles de grupos de interseção para o método rf...
Shapefile de PIXELS de interseção (421) exportado: E:\mestrado\mest3\resultados_comparativos_filtrado\rf_outputs\shapes_intersecao_pixels_rf.shp
Shapefile de CENTROIDES de interseção (175) exportado: E:\mestrado\mest3\resultados_comparativos_filtrado\rf_outputs\shapes_intersecao_centroides_rf.shp

-- Verificando interseção para o método: SVM --
  Pixels 'Sapucaia' em ambos os anos (SVM): 579
Filtro de tamanho encontrou 186 grupos. Detalhes (tamanho: contagem): {2: 93, 3: 77, 4: 16}
  Encontrados 186 grupos de interseção.
Exportando shapefiles de grupos de interseção para o método svm...
Shapefile de PIXELS de interseção (481) exportado: E:\mestrado\mest3\resultados_comparativos_filtrado\svm_outputs\shapes_intersecao_pixels_svm.shp
Shapefile de CENTROIDES de interseção (186) exportado: E:\mestrado\mest3\resultados_comparativos_filtrado\svm_outputs\shapes_intersecao_centroides_svm.shp

--- Script Totalmente Concluído ---
Tempo total de execução: 0:07:01.775459
