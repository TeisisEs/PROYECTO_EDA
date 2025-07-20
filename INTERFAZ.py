import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
from eda_manager import EDAManager
import tempfile
from ydata_profiling import ProfileReport
#_________________________________________________________________________________________
eda_manager = EDAManager()

def load_file(file):
    return eda_manager.load_data(file)

def toggle_info(check_state):
    return eda_manager.get_data_info() if check_state else ""

def toggle_describe(check_state):
    return eda_manager.get_statistical_summary() if check_state else pd.DataFrame()

def toggle_nulos(check_state):
    result = eda_manager.analyze_missing_values() if check_state else ""
    return result if isinstance(result, str) else result

def toggle_correlacion(check_state):
    return eda_manager.create_correlation_matrix() if check_state else None

def toggle_distribuciones(check_state):
    return eda_manager.create_distribution_plots() if check_state else None

def toggle_boxplots(check_state):
    return eda_manager.create_boxplots() if check_state else None

def toggle_categoricas(check_state):
    return eda_manager.create_categorical_plots() if check_state else None

def handle_grafico_change(tipo):
    return eda_manager.create_custom_plot(tipo) if tipo else None
#_________________________________________________________________________________________
#REPORTES AUTOMATICOS PYTHON
def generar_reportes():
    """Genera reporte EDA con ydata_profiling"""
    if eda_manager.df.empty:
        gr.Warning("Primero carga un dataset")
        return None
    
    try:
        # Configuración optimizada para mejor rendimiento
        config = {
            "progress_bar": False,
            "samples": {"head": 1000, "tail": 1000},  # Limita muestras
            "correlations": {
                "pearson": {"calculate": True},
                "spearman": {"calculate": False},
                "kendall": {"calculate": False}
            },
            "missing_diagrams": {
                "heatmap": False,
                "dendrogram": False
            }
        }

        with gr.Progress() as progress:
            progress(0.3, desc="Generando reporte...")
            
            # Generar el reporte con configuración optimizada
            report = ProfileReport(
                eda_manager.df,
                title="Reporte EDA Automático",
                minimal=False,
                config=config
            )
            
            progress(0.8, desc="Guardando reporte...")
            
            # Guardar en un archivo temporal
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                report.to_file(tmp.name)
                return tmp.name
            
    except Exception as e:
        gr.Warning(f"Error al generar el reporte: {str(e)}")
        return None
#_________________________________________________________________________________________

def create_interface():
    """Crea y configura la interfaz de Gradio"""
    with gr.Blocks(
        title="EDA  -  Análisis Exploratorio", 
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        """
    ) as demo:
        
        # Header principal
        gr.HTML("""
        <div class="main-header">
             EDA - Análisis Exploratorio 
        </div>
        <p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
            Herramienta profesional para análisis exploratorio de datos con visualizaciones avanzadas
        </p>
        """)
        
        # Sección de carga de archivo
        with gr.Row():
            with gr.Column(scale=2):
                file_input = gr.File(
                    label="Cargar Dataset", 
                    file_types=[".csv", ".txt", ".xlsx", ".xls"],
                    height=100
                )
            with gr.Column(scale=3):
                status_text = gr.Textbox(
                    label="Estado del Dataset", 
                    interactive=False,
                    lines=4
                )
        
        # Vista previa del dataset
        gr.Markdown("### Vista Previa del Dataset")
        df_output = gr.Dataframe(
            label="Primeras 100 filas del dataset",
            wrap=True
        )
        
        file_input.change(fn=load_file, inputs=file_input, outputs=[df_output, status_text])
        
        # Análisis básico con pestañas
        with gr.Tabs():
            with gr.TabItem("Información Básica"):
                with gr.Row():
                    check_info = gr.Checkbox(label="Información del Dataset", value=False)
                    check_desc = gr.Checkbox(label="Estadísticas", value=False)
                    check_nulos = gr.Checkbox(label="Análisis de Valores Faltantes", value=False)
                
                with gr.Row():
                    with gr.Column():
                        info_output = gr.Textbox(
                            label="Información del Dataset", 
                            lines=15,
                            max_lines=20
                        )
                    with gr.Column():
                        nulos_output = gr.Textbox(
                            label="Análisis de Valores Faltantes",
                            lines=15
                        )
                
                desc_output = gr.Dataframe(
                    label="Estadísticas Extendidas",
                )
 #_________________________________________________________________________________________
            # AQUI VOY A MODIFICAR LA INTERFACE            
            with gr.TabItem("Gráficos"):
                gr.Markdown("### Selecciona el Grafico a Crear:")
                
                with gr.Row():
                    tipo_grafico = gr.Radio(
                        ["Histograma", "Gráfico de barras", "Dispersión"], 
                        label="Tipo de gráfico",
                        value="Histograma"
                    )
                    
                with gr.Row():
                    x_col = gr.Dropdown(
                        label="Columna X (obligatoria)",
                        interactive=True
                    )
                    y_col = gr.Dropdown(
                        label="Columna Y (solo para dispersión)",
                        interactive=True,
                        visible=False  # Inicialmente oculto
                    )
                
                grafico_output = gr.Image(label="Gráfico Personalizado", height=500)

                # Actualizar dropdowns al cargar archivo
                def update_columns(df):
                    cols = df.columns.tolist() if isinstance(df, pd.DataFrame) else []
                    return [
                        gr.update(choices=cols, value=cols[0] if cols else None),
                        gr.update(choices=cols, value=cols[1] if len(cols) > 1 else None)
                    ]
                
                file_input.change(
                    fn=lambda f: update_columns(eda_manager.df),
                    inputs=file_input,
                    outputs=[x_col, y_col]
                )
                
                # Mostrar/ocultar columna Y según el tipo de gráfico
                def toggle_y_col(plot_type):
                    return gr.update(visible=plot_type == "Dispersión")
                
                tipo_grafico.change(
                    fn=toggle_y_col,
                    inputs=tipo_grafico,
                    outputs=y_col
                )
                
                # Manejador principal del gráfico
                def update_plot(plot_type, x_col, y_col):
                    return eda_manager.create_custom_plot(plot_type, x_col, y_col)
                
                tipo_grafico.change(
                    fn=update_plot,
                    inputs=[tipo_grafico, x_col, y_col],
                    outputs=grafico_output
                )
                x_col.change(
                    fn=update_plot,
                    inputs=[tipo_grafico, x_col, y_col],
                    outputs=grafico_output
                )
                y_col.change(
                    fn=update_plot,
                    inputs=[tipo_grafico, x_col, y_col],
                    outputs=grafico_output
                )
#_________________________________________________________________________________________
            with gr.TabItem(" Reportes"):
                gr.Markdown("### Genera reportes completos de EDA:")
                
                with gr.Row():
                    btn_reportes = gr.Button(
                        " Generar Reportes Completos", 
                        variant="primary",
                        size="lg"
                    )
                
                gr.Markdown("*Los reportes incluirán análisis completo en Python y R*")
                
                with gr.Row():
                    eda_python_file = gr.File(label=" Reporte EDA - Python")
                    eda_r_file = gr.File(
                        label=" Reporte EDA - R (Próximamente)",
                        interactive=False, #Para que no lo puedan clickear aun
                        visible=False  # Oculto por ahora
                        )
        
        # Event handlers
        check_info.change(fn=toggle_info, inputs=check_info, outputs=info_output)
        check_desc.change(fn=toggle_describe, inputs=check_desc, outputs=desc_output)
        check_nulos.change(fn=toggle_nulos, inputs=check_nulos, outputs=nulos_output)
        tipo_grafico.change(fn=handle_grafico_change, inputs=tipo_grafico, outputs=grafico_output)
        
        #FALTA PULIR LA FUNCION DE GENERAR REPORTE AUTOMATICO CON R
        #btn_reportes.click(fn=generar_reportes, outputs=[eda_python_file, eda_r_file])
        btn_reportes.click(fn=generar_reportes, outputs=[eda_python_file])
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 3rem; padding: 2rem; background-color: #f8f9fa; border-radius: 10px;">
            <p style="color: #666; font-size: 0.9rem;">
                 EDA  v2.0 - Desarrollado para análisis exploratorio <br>
                 Tip: Comienza cargando tu dataset y selecciona los análisis 
            </p>
        </div>
        """)
    
    return demo