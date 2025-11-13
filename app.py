import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score, log_loss, confusion_matrix, roc_curve, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# =========================================================================
# CONFIGURAÇÕES E CONSTANTES
# =========================================================================

st.set_page_config(layout="wide", page_title="Modelagem de Cancelamento de Reservas")

DATA_URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotel_bookings.csv"
SAMPLE_SIZE = 40000  # Usando o mesmo tamanho da análise original para velocidade

# Features selecionadas via RFE na análise original (10 features)
RFE_FEATURES = [
    'required_car_parking_spaces', 'country_AGO', 'country_PRT', 'reserved_room_type_A', 
    'assigned_room_type_A', 'assigned_room_type_I', 'deposit_type_No Deposit', 
    'deposit_type_Non Refund', 'deposit_type_Refundable', 'customer_type_Group'
]

# =========================================================================
# FUNÇÕES DE PRÉ-PROCESSAMENTO (CACHEADAS)
# =========================================================================

@st.cache_data
def load_and_preprocess_data(sample_n):
    """Carrega, amostra e pré-processa os dados (limpeza, OHE, escalonamento, SMOTE)."""
    
    st.info(f"Carregando e pré-processando {sample_n} registros (cacheado)...")
    
    # 1. Carregamento e Amostragem
    try:
        df = pd.read_csv(DATA_URL)
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return None, None, None, None, None, None

    if sample_n < len(df):
        df = df.sample(sample_n, random_state=42).reset_index(drop=True)
        
    # 2. Limpeza e Imputação Básica
    cols_to_drop = ['reservation_status_date', 'reservation_status', 'agent', 'company']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df['children'] = df['children'].fillna(0)
    df['country'] = df['country'].fillna('Unknown')
    df['is_canceled'] = df['is_canceled'].astype(int)
    
    # 3. Split treino/teste
    X = df.drop(columns=['is_canceled'])
    y = df['is_canceled']
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    # 4. Definição de Features Numéricas e Categóricas
    num_feats = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_feats = X.select_dtypes(include=['object','category']).columns.tolist()

    # 5. Pipeline de Pré-processamento (Escalonamento + OHE)
    num_pipeline = Pipeline([('imputer', st.cache_data(SimpleImputer)(strategy='median')), 
                             ('scaler', st.cache_data(StandardScaler)())])
    cat_pipeline = Pipeline([('imputer', st.cache_data(SimpleImputer)(strategy='most_frequent')),
                             ('onehot', st.cache_data(OneHotEncoder)(handle_unknown='ignore', sparse_output=False))])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_feats),
        ('cat', cat_pipeline, cat_feats)
    ], remainder='passthrough')

    X_train_enc = preprocessor.fit_transform(X_train_raw)
    X_test_enc = preprocessor.transform(X_test_raw)
    
    # 6. Mapeamento de Features (após OHE)
    all_feature_names = num_feats + list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_feats))
    
    # 7. Subsetting pelas RFE_FEATURES (Importante para modelos leves)
    rfe_indices = [all_feature_names.index(f) for f in RFE_FEATURES if f in all_feature_names]
    
    X_train_rfe = X_train_enc[:, rfe_indices]
    X_test_rfe = X_test_enc[:, rfe_indices]
    
    # 8. SMOTE (Balanceamento)
    smote = st.cache_data(SMOTE)(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_rfe, y_train)

    st.success("Dados prontos!")
    return X_train_bal, y_train_bal, X_test_rfe, y_test, RFE_FEATURES

# =========================================================================
# FUNÇÃO DE TREINAMENTO E AVALIAÇÃO
# =========================================================================

def train_and_evaluate_model(model_name, X_train, y_train, X_test, y_test, params):
    """Treina o modelo e calcula métricas essenciais."""
    
    st.write(f"### Modelo: {model_name}")
    st.json(params)
    
    if model_name == 'Regressão Logística':
        model = LogisticRegression(solver='liblinear', random_state=42, **params)
    elif model_name == 'KNN':
        model = KNeighborsClassifier(**params)
    elif model_name == 'SVM':
        # Nota: SVM é muito lento, usaremos um limite menor de max_iter para fins de demonstração
        model = SVC(probability=True, random_state=42, cache_size=500, **params)

    t0 = time.time()
    try:
        model.fit(X_train, y_train)
        train_time = time.time() - t0
    except Exception as e:
        st.error(f"Erro no treinamento: {e}. Tente ajustar os parâmetros.")
        return None, None
        
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Métricas
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    logloss = log_loss(y_test, y_proba)
    
    metrics = {
        'AUC': round(auc, 4),
        'F1-Score': round(f1, 4),
        'Log-Loss': round(logloss, 4),
        'Precisão': round(classification_report(y_test, y_pred, output_dict=True)['1']['precision'], 4),
        'Recall': round(classification_report(y_test, y_pred, output_dict=True)['1']['recall'], 4),
        'Tempo de Treino (s)': round(train_time, 2)
    }
    
    return metrics, y_proba

# =========================================================================
# INTERFACE STREAMLIT
# =========================================================================

def main():
    st.title("🎯 Dashboard de Predição de Cancelamento")
    st.subheader("Comparação de Modelos de Classificação (RL, KNN, SVM)")

    # Carregar e Pré-processar Dados
    X_train, y_train, X_test, y_test, feature_names = load_and_preprocess_data(SAMPLE_SIZE)
    
    if X_train is None:
        return

    st.sidebar.header("1. Seleção e Parâmetros")
    
    # Seleção de Algoritmo
    model_choice = st.sidebar.selectbox(
        "Selecione o Algoritmo",
        ('Regressão Logística', 'KNN', 'SVM')
    )
    
    params = {}
    
    # Configuração de Parâmetros
    if model_choice == 'Regressão Logística':
        st.sidebar.markdown("**Regressão Logística (RL)**: Modelo interpretável, linear.")
        params['penalty'] = st.sidebar.selectbox("Regularização (Penalty)", ('l2', 'l1'), index=0)
        params['C'] = st.sidebar.slider("Força de Regularização (C)", 0.01, 10.0, 1.0, 0.01)
        
    elif model_choice == 'KNN':
        st.sidebar.markdown("**K-Nearest Neighbors (KNN)**: Modelo baseado em distância, sensível ao escalonamento.")
        params['n_neighbors'] = st.sidebar.slider("Número de Vizinhos (k)", 1, 30, 5)
        params['metric'] = st.sidebar.selectbox("Métrica de Distância", ('minkowski', 'manhattan'))
        
    elif model_choice == 'SVM':
        st.sidebar.markdown("**Support Vector Machine (SVM)**: Modelo baseado em fronteira ótima, bom para não-linearidade (Kernel RBF).")
        params['kernel'] = st.sidebar.selectbox("Kernel", ('linear', 'rbf'))
        params['C'] = st.sidebar.slider("Parâmetro de Penalidade (C)", 0.01, 10.0, 1.0, 0.01)
        if params['kernel'] == 'rbf':
            params['gamma'] = st.sidebar.selectbox("Gamma", ('scale', 'auto', 0.01, 0.1))

    # Botão de Treinamento
    if st.sidebar.button("Treinar Modelo Selecionado"):
        
        # Treinar o modelo e calcular métricas
        with st.spinner(f"Treinando {model_choice}..."):
            metrics, y_proba = train_and_evaluate_model(model_choice, X_train, y_train, X_test, y_test, params)
        
        if metrics:
            st.session_state['current_model'] = {
                'name': model_choice,
                'metrics': metrics,
                'y_proba': y_proba,
                'params': params
            }
        
    # =========================================================================
    # VISUALIZAÇÃO DE RESULTADOS
    # =========================================================================

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'current_model' in st.session_state:
        current = st.session_state['current_model']
        
        st.header("2. Resultados do Modelo Atual")

        # Exibir Métricas em Tabela
        st.dataframe(pd.DataFrame([current['metrics']]).T.rename(columns={0: 'Valor'}), use_container_width=True)

        col_roc, col_cm = st.columns(2)
        
        with col_roc:
            # Curva ROC
            fig, ax = plt.subplots(figsize=(6, 4))
            fpr, tpr, _ = roc_curve(y_test, current['y_proba'])
            auc_val = current['metrics']['AUC']
            ax.plot(fpr, tpr, label=f'{current["name"]} (AUC={auc_val:.4f})')
            ax.plot([0, 1], [0, 1], 'r--')
            ax.set_xlabel('FPR (Taxa de Falso Positivo)')
            ax.set_ylabel('TPR (Taxa de Verdadeiro Positivo)')
            ax.set_title(f'Curva ROC - {current["name"]}')
            ax.legend()
            st.pyplot(fig)
            
        with col_cm:
            # Matriz de Confusão (usando threshold padrão de 0.5)
            y_pred = (current['y_proba'] >= 0.5).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Não Cancela (0)', 'Cancela (1)'],
                        yticklabels=['Não Cancela (0)', 'Cancela (1)'], ax=ax_cm)
            ax_cm.set_xlabel('Previsão'); ax_cm.set_ylabel('Real')
            ax_cm.set_title('Matriz de Confusão (Threshold 0.5)')
            st.pyplot(fig_cm)
            
        # Adicionar ao Histórico
        if st.button("Salvar Resultados para Comparação"):
            st.session_state['history'].append(current)
            st.success(f"Resultados de {current['name']} salvos para comparação!")
            
    # =========================================================================
    # COMPARATIVO E INTERPRETAÇÃO AUTOMÁTICA
    # =========================================================================

    if st.session_state['history']:
        st.header("3. Ranking e Comparação de Modelos Salvos")
        
        comparison_data = []
        for model_data in st.session_state['history']:
            row = model_data['metrics']
            row['Modelo'] = model_data['name'] + ' ' + str(model_data['params'])
            comparison_data.append(row)
            
        df_comp = pd.DataFrame(comparison_data).set_index('Modelo')
        
        # Ranking pelo Log-Loss (menor é melhor)
        best_model_loss = df_comp['Log-Loss'].idxmin()
        
        # Ranking pelo AUC (maior é melhor)
        best_model_auc = df_comp['AUC'].idxmax()
        
        st.subheader("Tabela Comparativa")
        st.dataframe(df_comp.sort_values(by='Log-Loss'), use_container_width=True)
        
        st.subheader("Interpretação Automática (Ranking)")
        
        st.markdown(f"""
        Com base nos modelos treinados até o momento:
        
        * **Melhor Modelo (Probabilidade - Log-Loss):** **{best_model_loss}**. Este modelo faz as previsões de probabilidade mais precisas, sendo ideal para definir estratégias de *overbooking* com base no risco.
        * **Melhor Modelo (Poder Discriminatório - AUC):** **{best_model_auc}**. Este modelo tem a melhor capacidade de distinguir entre reservas que irão e não irão cancelar.
        
        **Recomendação:** Priorize o modelo com o **menor Log-Loss** (atualmente **{best_model_loss}**), pois a minimização da função de perda é crucial para confiar na probabilidade de cancelamento (risco) na tomada de decisão operacional.
        """)

if __name__ == '__main__':
    main()
