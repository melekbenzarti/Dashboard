from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

# Translations Dictionary
TRANSLATIONS = {
    'en': {
        'title': 'SMS Analysis Dashboard',
        'criteria_table': 'Visit the Criteria Table',
        'methodology': 'Analysis Methodology',
        'methodology_desc': 'This dashboard presents two approaches to SMS optimization:',
        'human_analysis': 'Human Analysis: Manually selected optimal SMS sending times based on historical data.',
        'ml_analysis': 'Machine Learning Analysis: Predicted optimal SMS sending times using advanced predictive modeling.',
        'sector_select': 'Select a sector',
        'human_graphs': 'Human Analysis Graphs',
        'promo_click_rate': 'Promotional SMS: Click Rate per Hour',
        'info_stop_rate': 'Informational SMS: Stop Rate per Hour',
        'store_visits': 'Store Visits and CPV',
        'ml_graphs': 'Machine Learning Analysis Graphs',
        'comparison': 'Comparative Analysis'
    },
    'fr': {
        'title': 'Tableau de Bord d\'Analyse SMS',
        'criteria_table': 'Visiter le Tableau de Critères',
        'methodology': 'Méthodologie d\'Analyse',
        'methodology_desc': 'Ce tableau de bord présente deux approches d\'optimisation des SMS :',
        'human_analysis': 'Analyse Humaine : Sélection manuelle des meilleurs moments d\'envoi de SMS basée sur des données historiques.',
        'ml_analysis': 'Analyse par Apprentissage Automatique : Prédiction des meilleurs moments d\'envoi de SMS en utilisant une modélisation prédictive avancée.',
        'sector_select': 'Sélectionner un secteur',
        'human_graphs': 'Graphiques d\'Analyse Humaine',
        'promo_click_rate': 'SMS Promotionnels : Taux de Clic par Heure',
        'info_stop_rate': 'SMS Informationnels : Taux d\'Arrêt par Heure',
        'store_visits': 'Visites en Magasin et CPV',
        'ml_graphs': 'Graphiques d\'Analyse par Apprentissage Automatique',
        'comparison': 'Analyse Comparative'
    }
}

# Load both datasets
human_data_path = "Updated_EDITEED_Data.xlsx"
ml_data_path = "ML_Predicted_Data.xlsx"  # Assuming this is your new ML-predicted dataset

# Load human analysis data
edited_data = pd.read_excel(human_data_path)
edited_data['full_date'] = pd.to_datetime(edited_data['full_date'])
edited_data['hour'] = edited_data['full_date'].dt.hour
edited_data['day_of_week'] = edited_data['full_date'].dt.day_name()

# Load ML-predicted data
ml_data = pd.read_excel(ml_data_path)
ml_data['full_date'] = pd.to_datetime(ml_data['full_date'])
ml_data['hour'] = ml_data['full_date'].dt.hour
ml_data['day_of_week'] = ml_data['full_date'].dt.day_name()

# Aggregate human analysis data
human_aggregated_data = edited_data.groupby(['Secteur', 'label_predicted', 'day_of_week', 'hour']).agg({
    'Tx de clic unique': 'mean',
    'Tx de stop': 'mean',
    'tx délivrabilité': 'mean',
    'CPV': 'mean',
    'Visites en magasin': 'mean'
}).reset_index()

# Aggregate ML-predicted data
ml_aggregated_data = ml_data.groupby(['Secteur', 'label_predicted', 'day_of_week', 'hour', 'predicted_best_time']).agg({
    'Tx de clic unique': 'mean',
    'Tx de stop': 'mean',
    'tx délivrabilité': 'mean',
    'CPV': 'mean',
    'Visites en magasin': 'mean'
}).reset_index()

# Separate data for different SMS types and analysis methods
human_promo_data = human_aggregated_data[human_aggregated_data['label_predicted'] == 'promotion']
human_info_data = human_aggregated_data[human_aggregated_data['label_predicted'] == 'information']
human_store_visit_data = human_aggregated_data[~human_aggregated_data['Visites en magasin'].isna()]

ml_promo_data = ml_aggregated_data[ml_aggregated_data['label_predicted'] == 'promotion']
ml_info_data = ml_aggregated_data[ml_aggregated_data['label_predicted'] == 'information']
ml_store_visit_data = ml_aggregated_data[~ml_aggregated_data['Visites en magasin'].isna()]

# Extract unique sectors
unique_sectors = sorted(human_aggregated_data['Secteur'].unique())

# Function to generate a matplotlib graph image
def create_graph(data, metric, sector, analysis_type, sms_type, ylabel, language):
    plt.figure(figsize=(14, 8))
    
    # If ML data, color-code based on predicted best time
    if 'predicted_best_time' in data.columns:
        for predicted_time in data['predicted_best_time'].unique():
            subset = data[(data['Secteur'] == sector) & (data['predicted_best_time'] == predicted_time)]
            sns.lineplot(data=subset, x='hour', y=metric, marker='o', 
                         label=f"{ylabel} - Predicted Time {predicted_time}")
    else:
        subset = data[data['Secteur'] == sector]
        sns.lineplot(data=subset, x='hour', y=metric, marker='o', label=ylabel)
    
    # Translated title
    title_translations = {
        'en': f"{analysis_type} - {sms_type}: {ylabel} per hour for the {sector} sector",
        'fr': f"{analysis_type} - {sms_type} : {ylabel} par heure pour le secteur {sector}"
    }
    plt.title(title_translations.get(language, title_translations['en']))
    
    plt.xlabel("Hour of the day" if language == 'en' else "Heure de la journée")
    plt.ylabel(ylabel)
    plt.legend()

    # Save the graph into a BytesIO object
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png', bbox_inches='tight')
    img_io.seek(0)

    # Convert the image to base64 for displaying in Dash
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

# Function to generate an interpretation of the graph
def generate_interpretation(data, metric, analysis_type, sms_type, ylabel, language):
    mean_value = data[metric].mean()
    max_value = data[metric].max()
    min_value = data[metric].min()

    # Prepare translation keys
    translations = {
        'en': {
            'analysis': f"{analysis_type} Analysis for {sms_type.lower()}:",
            'avg': f"- The average {ylabel.lower()} value is {mean_value:.2f}.",
            'max': f"- The maximum observed value is {max_value:.2f}.",
            'min': f"- The minimum observed value is {min_value:.2f}.",
            'best_times': "Predicted Best Times:"
        },
        'fr': {
            'analysis': f"Analyse {analysis_type} pour {sms_type.lower()} :",
            'avg': f"- La valeur moyenne de {ylabel.lower()} est de {mean_value:.2f}.",
            'max': f"- La valeur maximale observée est de {max_value:.2f}.",
            'min': f"- La valeur minimale observée est de {min_value:.2f}.",
            'best_times': "Moments Prédits les Meilleurs :"
        }
    }

    # Select the right translation set
    t = translations.get(language, translations['en'])

    interpretation = f"{t['analysis']}\n"
    interpretation += f"{t['avg']}\n"
    interpretation += f"{t['max']}\n"
    interpretation += f"{t['min']}\n"
    
    # If ML data, add predicted best times
    if 'predicted_best_time' in data.columns:
        best_times = data.groupby('predicted_best_time')[metric].mean()
        interpretation += f"\n{t['best_times']}\n"
        for time, value in best_times.items():
            # Translate "Average" / "Moyenne"
            avg_text = "Average" if language == 'en' else "Moyenne"
            interpretation += f"- Predicted Time {time}: {avg_text} {ylabel} = {value:.2f}\n"
    
    return interpretation

# Function to generate comparison interpretation
def generate_comparison_interpretation(human_data, ml_data, metric, sector, sms_type, language):
    human_mean = human_data[human_data['Secteur'] == sector][metric].mean()
    ml_mean = ml_data[ml_data['Secteur'] == sector][metric].mean()
    
    # Prepare translation keys
    translations = {
        'en': {
            'title': f"Comparison for {sector} - {sms_type}:",
            'human_mean': "- Human Analysis Mean:",
            'ml_mean': "- ML Prediction Mean:",
            'improvement': "- Improvement:",
            'decline': "- Decline:",
            'no_change': "- No significant change"
        },
        'fr': {
            'title': f"Comparaison pour {sector} - {sms_type} :",
            'human_mean': "- Moyenne de l'Analyse Humaine :",
            'ml_mean': "- Moyenne de la Prédiction ML :",
            'improvement': "- Amélioration :",
            'decline': "- Déclin :",
            'no_change': "- Aucun changement significatif"
        }
    }

    # Select the right translation set
    t = translations.get(language, translations['en'])

    comparison = f"{t['title']}\n"
    comparison += f"{t['human_mean']} {human_mean:.2f}\n"
    comparison += f"{t['ml_mean']} {ml_mean:.2f}\n"
    
    # Calculate improvement or change
    if ml_mean > human_mean:
        improvement = ((ml_mean - human_mean) / human_mean) * 100
        comparison += f"{t['improvement']} +{improvement:.2f}%\n"
    elif ml_mean < human_mean:
        decline = ((human_mean - ml_mean) / human_mean) * 100
        comparison += f"{t['decline']} -{decline:.2f}%\n"
    else:
        comparison += f"{t['no_change']}\n"
    
    return comparison

# Initialize the Dash app
app = Dash(__name__)

# Layout of the app
def create_layout(language='en'):
    t = TRANSLATIONS[language]
    
    return html.Div([
        # Language Selection Buttons
        html.Div([
            dcc.Store(id='language-store', data={'language': language}),
            html.Button('English', id='lang-en-btn', n_clicks=0, style={'margin-right': '10px'}),
            html.Button('Français', id='lang-fr-btn', n_clicks=0)
        ], style={'text-align': 'right', 'margin': '10px'}),

        html.H1(t['title'], style={'text-align': 'center', 'id': 'dashboard-title'}),

        # Add a link to the Criteria Table
        html.Div([
            html.A(t['criteria_table'], 
                   href="https://table-de-criteres.streamlit.app/?embed_options=dark_theme", 
                   target="_blank", 
                   style={'font-size': '18px', 'color': 'blue', 'text-decoration': 'underline'})
        ], style={'text-align': 'center', 'margin-bottom': '20px'}),

        # Introduction and Methodology
        html.Div([
            html.H2(t['methodology']),
            html.P(t['methodology_desc']),
            html.Ul([
                html.Li(t['human_analysis']),
                html.Li(t['ml_analysis'])
            ])
        ], style={'margin': '20px 0', 'padding': '10px', 'border': '1px solid #ddd'}),

        # Sector Dropdown
        html.Div([
            html.Label(t['sector_select']),
            dcc.Dropdown(
                id='sector-dropdown',
                options=[{'label': sector, 'value': sector} for sector in unique_sectors],
                value=unique_sectors[0]
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'margin-bottom': '20px'}),

        # Human Analysis Graphs
        html.Div([
            html.H2(t['human_graphs'], style={'text-align': 'center'}),
            html.Div([
                html.Div([
                    html.H3(t['promo_click_rate']), 
                    html.Img(id='human-promotion-graph', style={'width': '100%', 'height': '500px'}), 
                    html.P(id='human-promotion-interpretation')
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    html.H3(t['info_stop_rate']), 
                    html.Img(id='human-informational-graph', style={'width': '100%', 'height': '500px'}), 
                    html.P(id='human-informational-interpretation')
                ], style={'width': '48%', 'display': 'inline-block'}),
            ]),
            html.Div([
                html.H3(t['store_visits']), 
                html.Img(id='human-store-visit-graph', style={'width': '100%', 'height': '500px'}), 
                html.P(id='human-store-visit-interpretation')
            ])
        ]),

        # Machine Learning Analysis Graphs
        html.Div([
            html.H2(t['ml_graphs'], style={'text-align': 'center'}),
            html.Div([
                html.Div([
                    html.H3(t['promo_click_rate']), 
                    html.Img(id='ml-promotion-graph', style={'width': '100%', 'height': '500px'}), 
                    html.P(id='ml-promotion-interpretation')
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    html.H3(t['info_stop_rate']), 
                    html.Img(id='ml-informational-graph', style={'width': '100%', 'height': '500px'}), 
                    html.P(id='ml-inform
