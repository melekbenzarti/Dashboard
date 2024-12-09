import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Translations dictionary
TRANSLATIONS = {
    'en': {
        'title': 'SMS Campaign Dashboard',
        'select_sector': 'Select a Sector',
        'sector': 'Sector',
        'human_analysis': 'Human Analysis',
        'ml_analysis': 'ML Analysis',
        'promo_sms_metrics_human': 'Promotional SMS Metrics (Human Analysis)',
        'promo_sms_metrics_ml': 'Promotional SMS Metrics (ML Analysis)',
        'click_rate': 'Click Rate',
        'stop_rate': 'Stop Rate',
        'visit_in_store': 'Visit in Store',
        'visit_criteria_table': 'Visit Criteria Table'
    },
    'fr': {
        'title': 'Tableau de Bord des Campagnes SMS',
        'select_sector': 'Sélectionner un Secteur',
        'sector': 'Secteur',
        'human_analysis': 'Analyse Humaine',
        'ml_analysis': 'Analyse IA',
        'promo_sms_metrics_human': 'Métriques SMS Promotionnels (Analyse Humaine)',
        'promo_sms_metrics_ml': 'Métriques SMS Promotionnels (Analyse IA)',
        'click_rate': 'Taux de Clic',
        'stop_rate': 'Taux de Stop',
        'visit_in_store': 'Visites en Magasin',
        'visit_criteria_table': 'Visite la Table de Critères'
    }
}

# Initialize language in session state if not already set
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# Function to set language
def set_language(lang):
    st.session_state.language = lang

# Language selection buttons
st.sidebar.header('Language / Langue')
col1, col2 = st.sidebar.columns(2)
with col1:
    st.button('English', on_click=set_language, args=('en',))
with col2:
    st.button('Français', on_click=set_language, args=('fr',))

# Get current language translations
def t(key):
    return TRANSLATIONS[st.session_state.language][key]

# Load datasets
human_data_path = "Updated_EDITEED_Data.xlsx"
ml_data_path = "DataBase_with_timing_predictions_final2.csv"

# Load and preprocess the data
edited_data = pd.read_excel(human_data_path)
edited_data['full_date'] = pd.to_datetime(edited_data['full_date'])
edited_data['hour'] = edited_data['full_date'].dt.hour
edited_data['day_of_week'] = edited_data['full_date'].dt.day_name()

ml_data = pd.read_csv(ml_data_path)
ml_data['predicted_best_time'] = pd.to_datetime(ml_data['predicted_best_time'])
ml_data['hour'] = ml_data['predicted_best_time'].dt.hour
ml_data['day_of_week'] = ml_data['predicted_best_time'].dt.day_name()

# Aggregated data
human_aggregated_data = edited_data.groupby(['Secteur', 'label_predicted', 'day_of_week', 'hour']).agg({
    'Tx de clic unique': 'mean',
    'Tx de stop': 'mean',
    'tx délivrabilité': 'mean',
    'CPV': 'mean',
    'Visites en magasin': 'mean'
}).reset_index()

ml_aggregated_data = ml_data.groupby(['Secteur', 'label_predicted', 'day_of_week', 'hour', 'predicted_best_time']).agg({
    'Tx de clic unique': 'mean',
    'Tx de stop': 'mean',
    'tx délivrabilité': 'mean',
    'CPV': 'mean',
    'Visites en magasin': 'mean'
}).reset_index()

# Separate data
human_promo_data = human_aggregated_data[human_aggregated_data['label_predicted'] == 'promotion']
ml_promo_data = ml_aggregated_data[ml_aggregated_data['label_predicted'] == 'promotion']

# Streamlit interface
st.title(t('title'))

# Sidebar for sector selection
sectors = human_aggregated_data['Secteur'].unique()
selected_sector = st.sidebar.selectbox(t('select_sector'), sectors)
# Hyperlink to Criteria Table
st.sidebar.markdown(f"""
[{t('visit_criteria_table')}](https://table-de-criteres.streamlit.app/?embed_options=dark_theme)
""", unsafe_allow_html=True)

# Function to generate plots
def plot_metric(data, sector, metric, ylabel, title):
    plt.figure(figsize=(12, 6))
    subset = data[data['Secteur'] == sector]
    sns.lineplot(data=subset, x='hour', y=metric, marker='o')
    plt.title(title)
    plt.xlabel("Hour")
    plt.ylabel(ylabel)
    st.pyplot(plt)

# Display metrics for Human Analysis
st.header(f"{t('sector')}: {selected_sector}")
st.subheader(t('human_analysis'))

if selected_sector:
    st.subheader(t('promo_sms_metrics_human'))
    plot_metric(
        human_promo_data, selected_sector, 'Tx de clic unique',
        t('click_rate'), f"{t('promo_sms_metrics_human')} - {selected_sector}"
    )
    plot_metric(
        human_promo_data, selected_sector, 'Tx de stop',
        t('stop_rate'), f"{t('promo_sms_metrics_human')} - {selected_sector}"
    )
    plot_metric(
        human_promo_data, selected_sector, 'CPV',
        t('visit_in_store'), f"{t('promo_sms_metrics_human')} - {selected_sector}"
    )

# Display metrics for ML Analysis
st.subheader(t('ml_analysis'))

if selected_sector:
    st.subheader(t('promo_sms_metrics_ml'))
    plot_metric(
        ml_promo_data, selected_sector, 'Tx de clic unique',
        t('click_rate'), f"{t('promo_sms_metrics_ml')} - {selected_sector}"
    )
    plot_metric(
        ml_promo_data, selected_sector, 'Tx de stop',
        t('stop_rate'), f"{t('promo_sms_metrics_ml')} - {selected_sector}"
    )
    plot_metric(
        ml_promo_data, selected_sector, 'CPV',
        t('visit_in_store'), f"{t('promo_sms_metrics_ml')} - {selected_sector}"
    )

#streamlit run dashboard.py