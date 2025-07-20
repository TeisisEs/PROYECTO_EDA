"""CONTIENE LA CLASE PRINCIPAL LA CLASE PRINCIPAL DEL ANALISIS"""

import logging
import warnings
import io
import tempfile
from typing import Tuple, Union, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración inicial
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

class EDAManager:
    """Clase principal para manejar el análisis exploratorio de datos"""
    
    def __init__(self):
        self.df = pd.DataFrame()
        self.file_info = {}
    
    def load_data(self, file) -> Tuple[pd.DataFrame, str]:
        """Carga datos desde archivo con validación"""
        try:
            if file is None:
                return pd.DataFrame(), " ALERTA: No se seleccionó ningún archivo"
            
            file_path = file.name
            file_size = file.size if hasattr(file, 'size') else 0
            
            # Validar tamaño de archivo (máximo 50MB)
            if file_size > 50 * 1024 * 1024:
                return pd.DataFrame(), " ALERTA: El archivo es demasiado grande (máximo 50MB)"
            
            # Cargar según extensión
            if file_path.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(file_path, nrows=10000)
            elif file_path.endswith('.csv'):
                sep = self._detect_csv_separator(file_path)
                self.df = pd.read_csv(file_path, sep=sep, nrows=10000, encoding='utf-8')
            else:
                return pd.DataFrame(), " Formato no soportado. Use CSV o Excel"
            
            self._update_file_info(file_path, file_size)
            
            success_msg = (f"✅ Archivo cargado exitosamente:\n"
                          f" {self.file_info['rows']} filas × {self.file_info['columns']} columnas\n"
                          f" Tamaño: {self.file_info['size_mb']} MB\n"
                          f" Memoria: {self.file_info['memory_usage']} MB")
            
            return self.df.head(100), success_msg
            
        except UnicodeDecodeError:
            try:
                self.df = pd.read_csv(file_path, encoding='latin1', nrows=10000)
                return self.df.head(100), "✅ Archivo cargado con codificación Latin1"
            except Exception as e:
                return pd.DataFrame(), f" Error de codificación: {str(e)}"
        except Exception as e:
            logging.error(f"Error loading file: {str(e)}")
            return pd.DataFrame(), f" Error al cargar archivo: {str(e)}"
    
    def _detect_csv_separator(self, file_path: str) -> str:
        """Detecta automáticamente el separador del CSV"""
        with open(file_path, 'r', encoding='utf-8') as f:
            sample = f.read(1024)
            if ';' in sample and ',' not in sample.split('\n')[0]:
                return ';'
            return ','
    
    def _update_file_info(self, file_path: str, file_size: int):
        """Actualiza la información del archivo cargado"""
        self.file_info = {
            'filename': file_path.split('/')[-1],
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'size_mb': round(file_size / (1024*1024), 2),
            'memory_usage': round(self.df.memory_usage(deep=True).sum() / (1024*1024), 2)
        }
    
    def get_data_info(self) -> str:
        """Información detallada del DataFrame"""
        if self.df.empty:
            return "No hay datos cargados"
        
        buffer = io.StringIO()
        self.df.info(buf=buffer, verbose=True, show_counts=True)
        info_text = buffer.getvalue()
        
        additional_info = self._get_additional_stats()
        return info_text + additional_info
    
    def _get_additional_stats(self) -> str:
        """Genera estadísticas adicionales para el reporte"""
        stats = [
            f"\n{'='*50}\n📈 ESTADÍSTICAS ADICIONALES:\n",
            f"• Filas duplicadas: {self.df.duplicated().sum()}",
            f"• Columnas numéricas: {len(self.df.select_dtypes(include=np.number).columns)}",
            f"• Columnas categóricas: {len(self.df.select_dtypes(include=['object', 'category']).columns)}",
            f"• Columnas de fecha: {len(self.df.select_dtypes(include=['datetime64']).columns)}"
        ]
        return '\n'.join(stats)
    
    def get_statistical_summary(self) -> pd.DataFrame:
        """Resumen estadístico mejorado"""
        if self.df.empty:
            return pd.DataFrame()
        
        numeric_df = self.df.select_dtypes(include=np.number)
        if numeric_df.empty:
            return pd.DataFrame()
        
        summary = numeric_df.describe(percentiles=[.05, .25, .5, .75, .95]).T
        summary = self._add_extra_stats(summary, numeric_df)
        return summary.round(3)
    
    def _add_extra_stats(self, summary: pd.DataFrame, numeric_df: pd.DataFrame) -> pd.DataFrame:
        """Agrega estadísticas adicionales al resumen"""
        summary['missing'] = numeric_df.isnull().sum()
        summary['missing_pct'] = (summary['missing'] / len(numeric_df) * 100).round(2)
        summary['skewness'] = numeric_df.skew().round(3)
        summary['kurtosis'] = numeric_df.kurtosis().round(3)
        return summary
    
    def analyze_missing_values(self) -> Union[pd.DataFrame, str]:
        """Análisis completo de valores faltantes"""
        if self.df.empty:
            return "No hay datos cargados"
        
        missing_count = self.df.isnull().sum()
        missing_pct = (missing_count / len(self.df) * 100).round(2)
        
        missing_df = pd.DataFrame({
            'Columna': missing_count.index,
            'Valores_Faltantes': missing_count.values,
            'Porcentaje': missing_pct.values,
            'Tipo_Dato': [str(self.df[col].dtype) for col in missing_count.index]
        })
        
        missing_df = missing_df[missing_df['Valores_Faltantes'] > 0]
        return "✅ No hay valores faltantes en el dataset" if missing_df.empty else missing_df.sort_values('Porcentaje', ascending=False)
    
    def create_correlation_matrix(self) -> Optional[str]:
        """Crea y guarda una matriz de correlación"""
        numeric_df = self.df.select_dtypes(include=np.number)
        if numeric_df.empty or len(numeric_df.columns) < 2:
            return None
        
        if len(numeric_df.columns) > 20:
            numeric_df = numeric_df.iloc[:, :20]
        
        plt.figure(figsize=(12, 10))
        correlation_matrix = numeric_df.corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', 
                   center=0, square=True, fmt='.2f', 
                   cbar_kws={'label': 'Coeficiente de Correlación'},
                   linewidths=0.5)
        
        plt.title('Matriz de Correlación entre Variables Numéricas', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        return self._save_plot()
    
    def create_distribution_plots(self) -> Optional[str]:
        """Histogramas y distribuciones mejoradas"""
        numeric_df = self.df.select_dtypes(include=np.number)
        if numeric_df.empty:
            return None
        
        n_cols = min(4, len(numeric_df.columns))
        n_rows = (len(numeric_df.columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = self._flatten_axes(axes, n_rows, n_cols)
        
        for i, col in enumerate(numeric_df.columns):
            if i < len(axes):
                self._plot_histogram_with_density(axes[i], numeric_df[col].dropna(), col)
        
        self._hide_empty_axes(axes, len(numeric_df.columns))
        plt.suptitle('Distribuciones de Variables Numéricas', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return self._save_plot()
    
    def _flatten_axes(self, axes, n_rows: int, n_cols: int):
        """Aplana el array de axes según la configuración de subplots"""
        if n_rows == 1 and n_cols == 1:
            return [axes]
        if n_rows == 1:
            return axes
        return axes.ravel()
    
    def _plot_histogram_with_density(self, ax, data, col_name: str):
        """Dibuja histograma con curva de densidad en el axis dado"""
        ax.hist(data, bins=30, alpha=0.7, edgecolor='black', 
               color='skyblue', density=True)
        
        if len(data) > 10:
            from scipy import stats
            x = np.linspace(data.min(), data.max(), 100)
            kde = stats.gaussian_kde(data)
            ax.plot(x, kde(x), 'r-', linewidth=2, alpha=0.8)
        
        mean_val = data.mean()
        ax.axvline(mean_val, color='red', linestyle='--', 
                  alpha=0.8, label=f'Media: {mean_val:.2f}')
        
        ax.set_title(f'Distribución de {col_name}', fontweight='bold')
        ax.set_xlabel(col_name)
        ax.set_ylabel('Densidad')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _hide_empty_axes(self, axes, num_plots: int):
        """Oculta los axes no utilizados"""
        for j in range(num_plots, len(axes)):
            axes[j].axis('off')
    
    def create_boxplots(self) -> Optional[str]:
        """Crea boxplots normalizados para detección de outliers"""
        numeric_df = self.df.select_dtypes(include=np.number)
        if numeric_df.empty:
            return None
        
        cols_to_plot = numeric_df.columns[:10]
        plt.figure(figsize=(max(15, len(cols_to_plot) * 2), 8))
        
        normalized_data, labels = self._prepare_boxplot_data(numeric_df, cols_to_plot)
        if normalized_data:
            self._draw_boxplots(normalized_data, labels)
        
        plt.title('Boxplots Normalizados - Detección de Outliers', 
                 fontsize=16, fontweight='bold')
        plt.ylabel('Valores Normalizados (Z-score)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return self._save_plot()
    
    def _prepare_boxplot_data(self, numeric_df: pd.DataFrame, cols_to_plot) -> tuple:
        """Prepara datos normalizados para boxplots"""
        normalized_data = []
        labels = []
        for col in cols_to_plot:
            data = numeric_df[col].dropna()
            if len(data) > 0:
                normalized = (data - data.mean()) / data.std()
                normalized_data.append(normalized)
                labels.append(f'{col}\n(n={len(data)})')
        return normalized_data, labels
    
    def _draw_boxplots(self, data, labels):
        """Dibuja los boxplots con estilos personalizados"""
        bp = plt.boxplot(data, labels=labels, patch_artist=True,
                       showfliers=True, flierprops={'marker': 'o', 'markersize': 3})
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    def create_categorical_plots(self) -> Optional[str]:
        """Crea gráficos de distribución para variables categóricas"""
        categorical_df = self.df.select_dtypes(include=['object', 'category'])
        if categorical_df.empty:
            return None
        
        valid_cols = self._filter_categorical_columns(categorical_df)
        if not valid_cols:
            return None
        
        n_cols = 2
        n_rows = (len(valid_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
        axes = self._flatten_axes(axes, n_rows, n_cols)
        
        colors = plt.cm.Set3(np.linspace(0, 1, 15))
        
        for i, col in enumerate(valid_cols):
            if i < len(axes):
                self._plot_categorical_distribution(axes[i], categorical_df[col], col, colors)
        
        self._hide_empty_axes(axes, len(valid_cols))
        plt.suptitle('Distribuciones de Variables Categóricas', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return self._save_plot()
    
    def _filter_categorical_columns(self, categorical_df: pd.DataFrame) -> list:
        """Filtra columnas categóricas con valores adecuados para graficar"""
        valid_cols = []
        for col in categorical_df.columns[:6]:
            unique_vals = categorical_df[col].nunique()
            if 2 <= unique_vals <= 15:
                valid_cols.append(col)
        return valid_cols
    
    def _plot_categorical_distribution(self, ax, series, col_name: str, colors):
        """Dibuja gráfico de barras para distribución categórica"""
        value_counts = series.value_counts().head(10)
        bars = ax.bar(range(len(value_counts)), value_counts.values, 
                     color=colors[:len(value_counts)], alpha=0.8, 
                     edgecolor='black', linewidth=0.5)
        
        ax.set_title(f'Distribución de {col_name}', fontweight='bold')
        ax.set_ylabel('Frecuencia')
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Agregar valores en las barras
        for bar, val in zip(bars, value_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(value_counts.values)*0.01,
                   str(val), ha='center', va='bottom', fontweight='bold')
            
   # __________________________________________________________________________________________________
    
    def create_custom_plot(self, plot_type: str, x_col: str = None, y_col: str = None) -> Optional[str]:
   
        if self.df.empty or x_col is None:
            return None

        try:
            plt.figure(figsize=(12, 8))
            
            if plot_type == "Histograma":
                self._plot_histogram(x_col)
            elif plot_type == "Gráfico de barras":
                self._plot_bar_chart(x_col)
            elif plot_type == "Dispersión" and y_col:
                self._plot_scatter(x_col, y_col)
            else:
                return None
                
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            return self._save_plot()
            
        except Exception as e:
            logging.error(f"Error al crear gráfico: {str(e)}")
            return None
  #  ___________________________________________________________________________________________________
  #ESTA ONDA ESTABA DANDO PROBLEMAS 
    def _plot_histogram(self, col_name: str):
        """Dibuja histograma para una columna específica"""
        if col_name not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[col_name]):
            return
            
        data = self.df[col_name].dropna()
        if len(data) == 0:
            return
            
        plt.hist(data, bins=min(30, len(data)), alpha=0.7, edgecolor='black', 
                color='skyblue', density=True)
        plt.title(f'Histograma de {col_name}', fontsize=14, fontweight='bold')
        plt.xlabel(col_name)
        plt.ylabel('Densidad')
    
        mean_val = data.mean()
        plt.axvline(mean_val, color='red', linestyle='--', 
                alpha=0.8, label=f'Media: {mean_val:.2f}')
    
    # Añadir curva de densidad si hay suficientes datos
        if len(data) > 10:
            from scipy import stats
            kde = stats.gaussian_kde(data)
            x = np.linspace(data.min(), data.max(), 100)
            plt.plot(x, kde(x), 'r-', linewidth=2, alpha=0.8)
            
        plt.legend()
        plt.grid(True, alpha=0.3)
    
       
    def _plot_bar_chart(self, col_name: str):
        """Gráfico de barras para columna categórica"""
        if col_name not in self.df.columns:
            return
            
        value_counts = self.df[col_name].value_counts().head(15)
        bars = plt.bar(range(len(value_counts)), value_counts.values, 
                    color='skyblue', edgecolor='black', alpha=0.8)
        
        plt.title(f'Gráfico de Barras - {col_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Categorías')
        plt.ylabel('Frecuencia')
        plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
        
        for bar, val in zip(bars, value_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(value_counts.values)*0.01,
                str(val), ha='center', va='bottom')

    def _plot_scatter(self, x_col: str, y_col: str):
        """Gráfico de dispersión entre dos columnas"""
        if x_col not in self.df.columns or y_col not in self.df.columns:
            return
            
        x_data = self.df[x_col].dropna()
        y_data = self.df[y_col].dropna()
        
        # Asegurar misma longitud
        min_len = min(len(x_data), len(y_data))
        x_data = x_data[:min_len]
        y_data = y_data[:min_len]
        
        plt.scatter(x_data, y_data, alpha=0.6, s=30, 
                edgecolors='black', linewidth=0.5)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'Dispersión: {x_col} vs {y_col}', fontsize=14, fontweight='bold')
        
        # Línea de regresión (opcional)
        try:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            plt.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2)
            
            corr = np.corrcoef(x_data, y_data)[0,1]
            plt.text(0.05, 0.95, f'Correlación: {corr:.3f}', 
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        except:
            pass
    #HASTA LLEGA EL PROBELMA PARA SELECCIONAR EL GRAFUCO XD 
    # ___________________________________________________________________________________________________
    def _save_plot(self) -> str:
        """Guarda el plot actual como imagen y retorna la ruta"""
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        plt.close()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(buf.read())
            return tmp.name