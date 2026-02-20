import os
import re
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ======================== НАСТРАИВАЕМЫЕ ПАРАМЕТРЫ ========================
CONFIG = {
    'xlim_days': 1000,  # Ось Х по умолчанию (сутки)
    'ylim_rate': 1.5,  # Ось Y по умолчанию (темп)
    'max_outliers_ratio': 0.3,  # Максимальная доля выбросов (30%)
    'min_points': 5,  # Минимальное количество точек для аппроксимации
    'n_initial_points': 3,  # Количество начальных точек для совпадения
    'n_final_points': 3,  # Количество конечных точек для совпадения
    'convergence_threshold': 0.85,  # Порог сходимости (R^2) - снизил для реальных данных
    'max_iterations': 15,  # Максимальное число итераций исключения выбросов
    'parallel_workers': max(1, cpu_count() - 1),  # Количество параллельных процессов
    'output_base_dir': 'results',  # Базовая директория для результатов
    'max_days': 1000,  # Максимальное количество дней для анализа
}


# ========================================================================

class ApproximationMethods:
    """Класс с методами аппроксимации, начинающимися с 1 в нулевые сутки"""

    @staticmethod
    def exponential(x, a, b):
        """Экспоненциальная: y = exp(-a*x) + (1-exp(-a*x))*b"""
        return np.exp(-a * x) + (1 - np.exp(-a * x)) * b

    @staticmethod
    def logarithmic(x, a, b):
        """Логарифмическая с насыщением: y = 1 - a*ln(1+b*x)"""
        return 1 - a * np.log1p(b * x)

    @staticmethod
    def power_law(x, a, b):
        """Степенная: y = (1 + a*x)^(-b)"""
        return (1 + a * x) ** (-b)

    @staticmethod
    def hyperbolic(x, a, b):
        """Гиперболическая: y = 1 / (1 + a*x**b)"""
        return 1 / (1 + a * x ** b)

    @staticmethod
    def rational(x, a, b):
        """Рациональная: y = (1 + a*x) / (1 + b*x)"""
        return (1 + a * x) / (1 + b * x)

    @staticmethod
    def arps_harmonic(x, a):
        """Арпса гармоническая: y = 1 / (1 + a*x)"""
        return 1 / (1 + a * x)

    @staticmethod
    def arps_hyperbolic(x, a, b):
        """Арпса гиперболическая: y = (1 + b*a*x)^(-1/b)"""
        return (1 + b * a * x) ** (-1 / b)


def get_approximation_methods():
    """Возвращает список методов аппроксимации"""
    return [
        {
            'name': 'Экспоненциальная',
            'func': ApproximationMethods.exponential,
            'bounds': ([0, 0], [10, 2]),
            'p0': [0.1, 0.1]
        },
        {
            'name': 'Логарифмическая',
            'func': ApproximationMethods.logarithmic,
            'bounds': ([0, 0], [2, 2]),
            'p0': [0.1, 0.1]
        },
        {
            'name': 'Степенная',
            'func': ApproximationMethods.power_law,
            'bounds': ([0, 0], [2, 5]),
            'p0': [0.1, 0.5]
        },
        {
            'name': 'Гиперболическая',
            'func': ApproximationMethods.hyperbolic,
            'bounds': ([0, 0], [5, 2]),
            'p0': [0.1, 0.8]
        },
        {
            'name': 'Рациональная',
            'func': ApproximationMethods.rational,
            'bounds': ([0, 0], [5, 5]),
            'p0': [0.1, 0.1]
        },
        {
            'name': 'Гармоническая Арпса',
            'func': ApproximationMethods.arps_harmonic,
            'bounds': ([0], [10]),
            'p0': [0.1],
            'is_one_param': True
        },
        {
            'name': 'Гиперболическая Арпса',
            'func': ApproximationMethods.arps_hyperbolic,
            'bounds': ([0, 0.01], [5, 2]),
            'p0': [0.1, 0.5]
        }
    ]


def read_excel_file(filename):
    """
    Чтение Excel файла с правильной структурой:
    - Первая колонка: названия скважин
    - Первая строка: дни (сутки)
    - Ячейка A1: заголовок "Скважина / Сутки"
    """
    try:
        # Читаем весь файл без заголовков
        df_raw = pd.read_excel(filename, header=None, index_col=None)

        logger.info(f"Файл прочитан, размер: {df_raw.shape}")
        logger.info(f"Первая ячейка: {df_raw.iloc[0, 0]}")

        # Извлекаем дни из первой строки (начиная со второй колонки)
        days = df_raw.iloc[0, 1:].values
        # Преобразуем дни в числа
        days = pd.to_numeric(days, errors='coerce')

        # Извлекаем названия скважин из первой колонки (начиная со второй строки)
        well_names = df_raw.iloc[1:, 0].values

        # Создаем DataFrame с правильной структурой: строки - дни, колонки - скважины
        data_dict = {}
        for i, well_name in enumerate(well_names):
            if pd.notna(well_name):  # Пропускаем пустые названия
                well_name = str(well_name).strip()
                # Извлекаем данные для скважины (строка i+1, колонки 1:)
                well_data = df_raw.iloc[i + 1, 1:].values
                # Преобразуем в числа
                well_data = pd.to_numeric(well_data, errors='coerce')
                # Создаем Series с индексами-днями
                data_dict[well_name] = pd.Series(well_data, index=days)

        # Создаем DataFrame
        df = pd.DataFrame(data_dict)

        # Удаляем колонки с NaN в названиях
        df = df.dropna(axis=1, how='all')

        # Сортируем дни по возрастанию
        df = df.sort_index()

        # Ограничиваем количество дней
        if CONFIG['max_days'] > 0:
            df = df[df.index <= CONFIG['max_days']]

        logger.info(f"Преобразовано: {df.shape[1]} скважин, {df.shape[0]} дней")
        logger.info(f"Диапазон дней: {df.index.min()} - {df.index.max()}")
        logger.info(f"Первые 5 скважин: {list(df.columns)[:5]}")

        return df

    except Exception as e:
        logger.error(f"Ошибка чтения файла: {e}")
        logger.error("Проверьте структуру Excel файла:")
        logger.error("  - Первая строка: дни (0, 1, 2, ...)")
        logger.error("  - Первая колонка: названия скважин (10-01, 10-02, ...)")
        logger.error("  - Ячейка A1: заголовок 'Скважина / Сутки'")
        raise


def enforce_boundary_conditions(x_data, y_data, n_initial=3, n_final=3):
    """Принудительное обеспечение условий: начало в 1, концы совпадают"""
    x_data = np.array(x_data, dtype=float)
    y_data = np.array(y_data, dtype=float)

    # Удаляем NaN и бесконечности
    mask = np.isfinite(x_data) & np.isfinite(y_data) & (y_data > 0)
    x_data = x_data[mask]
    y_data = y_data[mask]

    if len(x_data) < 3:
        return x_data, y_data, np.ones_like(x_data)

    # Нормализуем данные: первый темп должен быть 1
    if y_data[0] != 1.0 and y_data[0] > 0:
        y_data = y_data / y_data[0]

    # Убеждаемся, что y=1 при x=0 (добавляем фиктивную точку)
    if x_data[0] > 0:
        x_data = np.insert(x_data, 0, 0)
        y_data = np.insert(y_data, 0, 1.0)

    # Усиливаем вес начальных и конечных точек
    weights = np.ones_like(x_data, dtype=float)
    if len(weights) > n_initial:
        weights[:n_initial] *= 5  # Больший вес начальным точкам
    if len(weights) > n_final:
        weights[-n_final:] *= 3  # Больший вес конечным точкам

    return x_data, y_data, weights


def fit_curve_with_constraints(x_data, y_data, method, n_initial=3, n_final=3):
    """Подгонка кривой с учетом граничных условий"""
    x_data, y_data, weights = enforce_boundary_conditions(x_data, y_data, n_initial, n_final)

    if len(x_data) < 3:
        return {'success': False, 'error': 'Недостаточно точек'}

    try:
        if method.get('is_one_param', False):
            popt, _ = curve_fit(method['func'], x_data, y_data,
                                p0=method['p0'],
                                bounds=method['bounds'],
                                sigma=1 / weights if len(weights) == len(x_data) else None,
                                maxfev=5000,
                                nan_policy='omit')
        else:
            popt, _ = curve_fit(method['func'], x_data, y_data,
                                p0=method['p0'],
                                bounds=method['bounds'],
                                sigma=1 / weights if len(weights) == len(x_data) else None,
                                maxfev=5000,
                                nan_policy='omit')

        y_pred = method['func'](x_data, *popt)

        # Маска для удаления добавленной точки x=0 для расчета R2
        if x_data[0] == 0 and len(x_data) > 1:
            mask = x_data > 0
            if mask.sum() > 1:
                r2 = r2_score(y_data[mask], y_pred[mask])
            else:
                r2 = r2_score(y_data, y_pred)
        else:
            r2 = r2_score(y_data, y_pred)

        return {
            'success': True,
            'popt': popt,
            'r2': r2,
            'y_pred': y_pred,
            'method_name': method['name'],
            'func': method['func']
        }
    except Exception as e:
        return {'success': False, 'error': str(e), 'method_name': method['name']}


def iterative_outlier_removal(x_data, y_data, method, config):
    """Итеративное исключение выбросов"""
    x_clean = np.array(x_data, dtype=float).copy()
    y_clean = np.array(y_data, dtype=float).copy()
    outliers_values = []
    best_r2 = -np.inf
    best_state = None

    for iteration in range(config['max_iterations']):
        if len(x_clean) < config['min_points']:
            break

        fit_result = fit_curve_with_constraints(x_clean, y_clean, method,
                                                config['n_initial_points'],
                                                config['n_final_points'])

        if not fit_result['success']:
            break

        current_r2 = fit_result['r2']

        # Сохраняем лучшее состояние
        if current_r2 > best_r2:
            best_r2 = current_r2
            best_state = {
                'x_clean': x_clean.copy(),
                'y_clean': y_clean.copy(),
                'fit_result': fit_result,
                'outliers': outliers_values.copy()
            }

        residuals = np.abs(y_clean - fit_result['y_pred'][:len(y_clean)])
        max_residual_idx = np.argmax(residuals)
        max_residual_value = residuals[max_residual_idx]

        # Критерий остановки
        if (current_r2 >= config['convergence_threshold'] or
                len(outliers_values) >= len(y_data) * config['max_outliers_ratio'] or
                max_residual_value < 0.01 or
                len(x_clean) <= config['min_points']):
            break

        # Исключаем точку с максимальным отклонением
        outliers_values.append((x_clean[max_residual_idx], y_clean[max_residual_idx]))
        x_clean = np.delete(x_clean, max_residual_idx)
        y_clean = np.delete(y_clean, max_residual_idx)

    # Возвращаем лучшее найденное состояние
    if best_state is not None:
        return {
            'x_clean': best_state['x_clean'],
            'y_clean': best_state['y_clean'],
            'outliers': best_state['outliers'],
            'n_outliers': len(best_state['outliers']),
            'fit_result': best_state['fit_result'],
            'iteration': iteration + 1
        }
    else:
        return {
            'x_clean': x_clean,
            'y_clean': y_clean,
            'outliers': outliers_values,
            'n_outliers': len(outliers_values),
            'fit_result': fit_result if 'fit_result' in locals() else {'success': False},
            'iteration': iteration + 1
        }


def process_well(args):
    """Обработка одной скважины"""
    well_name, data_series, config = args

    try:
        # Получаем индексы (дни) и значения
        x_data = data_series.index.values
        y_data = data_series.values

        # Удаляем NaN
        mask = pd.notna(y_data) & pd.notna(x_data)
        x_data = x_data[mask]
        y_data = y_data[mask]

        if len(x_data) < config['min_points']:
            logger.warning(f"Скважина {well_name}: недостаточно данных ({len(x_data)} точек)")
            return None

        # Нормализуем данные: первый темп должен быть 1
        if y_data[0] > 0:
            y_data = y_data / y_data[0]

        methods = get_approximation_methods()
        best_result = None
        best_r2 = -np.inf
        best_method_name = None
        best_outliers_info = None

        # Пробуем все методы аппроксимации
        for method in methods:
            try:
                outlier_result = iterative_outlier_removal(x_data, y_data, method, config)

                if outlier_result['fit_result']['success']:
                    r2 = outlier_result['fit_result']['r2']

                    if r2 > best_r2:
                        best_r2 = r2
                        best_result = outlier_result['fit_result']
                        best_method_name = method['name']
                        best_outliers_info = outlier_result
            except Exception as e:
                logger.debug(f"Метод {method['name']} для {well_name} не сработал: {e}")
                continue

        if best_result is not None:
            # Генерируем уравнение
            equation = generate_equation(best_method_name, best_result['popt'])

            # Создаем функцию аппроксимации для этой скважины
            methods_dict = {m['name']: m['func'] for m in methods}

            return {
                'well_name': str(well_name),
                'x_original': x_data.tolist(),
                'y_original': y_data.tolist(),
                'x_clean': best_outliers_info['x_clean'].tolist(),
                'y_clean': best_outliers_info['y_clean'].tolist(),
                'x_fit': best_outliers_info['x_clean'].tolist(),
                'y_fit': best_result['y_pred'][:len(best_outliers_info['x_clean'])].tolist(),
                'outliers': best_outliers_info['outliers'],
                'r2': best_r2,
                'method': best_method_name,
                'equation': equation,
                'n_outliers': best_outliers_info['n_outliers'],
                'n_points_original': len(x_data),
                'n_points_clean': len(best_outliers_info['x_clean']),
                'popt': best_result['popt'].tolist(),
                'func': methods_dict[best_method_name]
            }
        else:
            logger.warning(f"Скважина {well_name}: не удалось подобрать аппроксимацию")
            return None

    except Exception as e:
        logger.error(f"Ошибка при обработке скважины {well_name}: {e}")
        return None


def generate_equation(method_name, popt):
    """Генерирует строку с уравнением"""
    if method_name == 'Экспоненциальная':
        return f"y = exp(-{popt[0]:.4f}*x) + (1-exp(-{popt[0]:.4f}*x))*{popt[1]:.4f}"
    elif method_name == 'Логарифмическая':
        return f"y = 1 - {popt[0]:.4f}*ln(1+{popt[1]:.4f}*x)"
    elif method_name == 'Степенная':
        return f"y = (1 + {popt[0]:.4f}*x)^(-{popt[1]:.4f})"
    elif method_name == 'Гиперболическая':
        return f"y = 1/(1 + {popt[0]:.4f}*x^{popt[1]:.4f})"
    elif method_name == 'Рациональная':
        return f"y = (1 + {popt[0]:.4f}*x)/(1 + {popt[1]:.4f}*x)"
    elif method_name == 'Гармоническая Арпса':
        return f"y = 1/(1 + {popt[0]:.4f}*x)"
    elif method_name == 'Гиперболическая Арпса':
        return f"y = (1 + {popt[1]:.4f}*{popt[0]:.4f}*x)^(-1/{popt[1]:.4f})"
    else:
        return f"y = f(x, params={popt})"


def create_output_directory():
    """Создает директорию для результатов текущего запуска"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(CONFIG['output_base_dir'], f"run_{timestamp}")
    plots_dir = os.path.join(output_dir, "well_plots")
    os.makedirs(plots_dir, exist_ok=True)
    return output_dir, plots_dir


def save_rates_excel(results, output_dir):
    """Сохраняет файл Темпы_апроксимированные.xlsx"""
    output_file = os.path.join(output_dir, "Темпы_апроксимированные.xlsx")

    # Находим максимальный день
    max_day = 0
    for result in results:
        if result is not None and result['x_fit']:
            max_day = max(max_day, max(result['x_fit']))

    max_day = min(max_day, CONFIG['xlim_days'])
    days = np.arange(0, max_day + 1, 1)

    # Создаем DataFrame с днями
    df_result = pd.DataFrame({'Сутки': days})

    # Добавляем данные по каждой скважине
    for result in results:
        if result is not None:
            well_name = result['well_name']
            try:
                # Интерполируем аппроксимированные значения на все дни
                if len(result['x_fit']) > 1:
                    f = interp1d(result['x_fit'], result['y_fit'],
                                 kind='linear', fill_value='extrapolate',
                                 bounds_error=False)
                    y_interp = f(days)
                    # Ограничиваем значения
                    y_interp = np.clip(y_interp, 0, CONFIG['ylim_rate'])
                    df_result[well_name] = y_interp
            except Exception as e:
                logger.warning(f"Не удалось интерполировать данные для {well_name}: {e}")

    df_result.to_excel(output_file, index=False)
    logger.info(f"Сохранен файл: {output_file}")


def save_analytical_results(results, output_dir):
    """Сохраняет файл Аналитические результаты.xlsx"""
    output_file = os.path.join(output_dir, "Аналитические результаты.xlsx")

    # Статистика по каждой скважине
    well_stats = []

    for result in results:
        if result is not None:
            well_stats.append({
                'Скважина': result['well_name'],
                'Метод аппроксимации': result['method'],
                'Уравнение': result['equation'],
                'R^2': round(result['r2'], 4),
                'Исходных точек': result['n_points_original'],
                'Исключено выбросов': result['n_outliers'],
                'Доля выбросов': f"{result['n_outliers'] / max(result['n_points_original'], 1) * 100:.1f}%",
                'Точек после очистки': result['n_points_clean']
            })

    df_wells = pd.DataFrame(well_stats)

    # Общая статистика
    if not df_wells.empty:
        total_stats = pd.DataFrame([{
            'Показатель': 'Всего скважин',
            'Значение': len(well_stats)
        }, {
            'Показатель': 'Средний R^2',
            'Значение': round(df_wells['R^2'].mean(), 4)
        }, {
            'Показатель': 'Медианный R^2',
            'Значение': round(df_wells['R^2'].median(), 4)
        }, {
            'Показатель': 'Мин R^2',
            'Значение': round(df_wells['R^2'].min(), 4)
        }, {
            'Показатель': 'Макс R^2',
            'Значение': round(df_wells['R^2'].max(), 4)
        }, {
            'Показатель': 'Средняя доля выбросов',
            'Значение': f"{df_wells['Доля выбросов'].apply(lambda x: float(x.strip('%'))).mean():.1f}%"
        }])
    else:
        total_stats = pd.DataFrame()

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_wells.to_excel(writer, sheet_name='По скважинам', index=False)
        if not total_stats.empty:
            total_stats.to_excel(writer, sheet_name='Общая статистика', index=False)

    logger.info(f"Сохранен файл: {output_file}")
    return df_wells


def plot_all_wells(results, plots_dir, config):
    """Создает графики по каждой скважине"""
    for result in results:
        if result is None:
            continue

        fig, ax = plt.subplots(figsize=(12, 8))

        # Исходные данные
        if result['x_original'] and result['y_original']:
            ax.scatter(result['x_original'], result['y_original'],
                       color='gray', alpha=0.5, label='Исходные данные', s=20)

        # Данные после исключения выбросов
        if result['x_clean'] and result['y_clean']:
            ax.scatter(result['x_clean'], result['y_clean'],
                       color='blue', alpha=0.7, label='После очистки', s=30)

        # Аппроксимированная кривая
        if result['x_fit'] and len(result['x_fit']) > 1:
            x_smooth = np.linspace(0, min(config['xlim_days'], max(result['x_fit'])), 200)
            y_smooth = result['func'](x_smooth, *result['popt'])
            ax.plot(x_smooth, y_smooth, 'r-', linewidth=2,
                    label=f"{result['method']}\n{result['equation']}\nR² = {result['r2']:.4f}")

        # Выбросы
        if result['outliers']:
            outliers_x = [o[0] for o in result['outliers']]
            outliers_y = [o[1] for o in result['outliers']]
            ax.scatter(outliers_x, outliers_y, color='red',
                       marker='x', s=100, label=f'Выбросы ({len(result["outliers"])})')

        ax.set_xlabel('Сутки', fontsize=12)
        ax.set_ylabel('Темп (нормированный)', fontsize=12)
        ax.set_title(f"Скважина: {result['well_name']}", fontsize=14)
        ax.set_xlim(0, config['xlim_days'])
        ax.set_ylim(0, config['ylim_rate'])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)

        # Очищаем имя файла от недопустимых символов
        safe_name = re.sub(r'[^\w\s-]', '', str(result['well_name']))
        filename = f"well_{safe_name}.png"
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

    logger.info(f"Сохранены графики по скважинам в: {plots_dir}")


def plot_grouped_curves(results, output_dir, config):
    """Создает групповые графики аппроксимированных кривых"""

    # Группируем по методам аппроксимации
    methods_groups = {}
    for result in results:
        if result is not None:
            method = result['method']
            if method not in methods_groups:
                methods_groups[method] = []
            methods_groups[method].append(result)

    if not methods_groups:
        logger.warning("Нет данных для построения групповых графиков")
        return

    common_x = np.linspace(0, config['xlim_days'], 500)

    # 1. Графики по каждой группе
    n_methods = len(methods_groups)
    n_cols = min(4, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig1, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_methods == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (method, group_results) in enumerate(methods_groups.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Тонкие линии для каждой скважины
        all_y = []
        for result in group_results:
            try:
                y_smooth = result['func'](common_x, *result['popt'])
                ax.plot(common_x, y_smooth, 'b-', alpha=0.3, linewidth=0.5)
                all_y.append(y_smooth)
            except:
                continue

        # Средняя линия
        if all_y:
            mean_y = np.mean(all_y, axis=0)
            ax.plot(common_x, mean_y, 'r-', linewidth=2.5, label='Среднее')

        ax.set_xlabel('Сутки')
        ax.set_ylabel('Темп')
        ax.set_title(f"{method}\n({len(group_results)} скважин)")
        ax.set_xlim(0, config['xlim_days'])
        ax.set_ylim(0, config['ylim_rate'])
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Скрываем неиспользуемые подграфики
    for idx in range(len(methods_groups), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grouped_curves.png"), dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # 2. Общий график - средние линии всех групп
    fig2, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.tab20(np.linspace(0, 1, len(methods_groups)))

    for idx, (method, group_results) in enumerate(methods_groups.items()):
        all_y = []
        for result in group_results:
            try:
                y_smooth = result['func'](common_x, *result['popt'])
                all_y.append(y_smooth)
            except:
                continue

        if all_y:
            mean_y = np.mean(all_y, axis=0)
            ax.plot(common_x, mean_y, color=colors[idx], linewidth=2.5,
                    label=f"{method} (n={len(group_results)})")

    ax.set_xlabel('Сутки', fontsize=12)
    ax.set_ylabel('Темп', fontsize=12)
    ax.set_title('Средние аппроксимированные кривые по группам', fontsize=14)
    ax.set_xlim(0, config['xlim_days'])
    ax.set_ylim(0, config['ylim_rate'])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    plt.savefig(os.path.join(output_dir, "all_groups_mean.png"), dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # 3. Пучок тонких линий всех скважин с группировкой по цветам
    fig3, ax = plt.subplots(figsize=(14, 8))

    for idx, (method, group_results) in enumerate(methods_groups.items()):
        for result in group_results:
            try:
                y_smooth = result['func'](common_x, *result['popt'])
                ax.plot(common_x, y_smooth, color=colors[idx], alpha=0.2, linewidth=0.5)
            except:
                continue

    # Легенда для цветов
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[idx], alpha=0.5,
                             label=f"{method} (n={len(group_results)})")
                       for idx, (method, group_results) in enumerate(methods_groups.items())]

    ax.set_xlabel('Сутки', fontsize=12)
    ax.set_ylabel('Темп', fontsize=12)
    ax.set_title('Все аппроксимированные кривые (по группам)', fontsize=14)
    ax.set_xlim(0, config['xlim_days'])
    ax.set_ylim(0, config['ylim_rate'])
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_elements, loc='best', fontsize=10)

    plt.savefig(os.path.join(output_dir, "all_curves_grouped.png"), dpi=150, bbox_inches='tight')
    plt.close(fig3)

    # 4. Аналитические графики
    fig4, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Распределение R^2
    r2_values = [r['r2'] for r in results if r is not None]
    if r2_values:
        ax1 = axes[0, 0]
        ax1.hist(r2_values, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(r2_values), color='red', linestyle='--',
                    label=f'Среднее: {np.mean(r2_values):.3f}')
        ax1.set_xlabel('R²')
        ax1.set_ylabel('Количество скважин')
        ax1.set_title('Распределение качества аппроксимации')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Доля выбросов
    outlier_ratios = [r['n_outliers'] / max(r['n_points_original'], 1) * 100
                      for r in results if r is not None]
    if outlier_ratios:
        ax2 = axes[0, 1]
        ax2.hist(outlier_ratios, bins=20, color='coral', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(outlier_ratios), color='red', linestyle='--',
                    label=f'Среднее: {np.mean(outlier_ratios):.1f}%')
        ax2.set_xlabel('Доля выбросов, %')
        ax2.set_ylabel('Количество скважин')
        ax2.set_title('Распределение доли выбросов')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Популярность методов
    methods_count = {}
    for r in results:
        if r is not None:
            methods_count[r['method']] = methods_count.get(r['method'], 0) + 1

    if methods_count:
        ax3 = axes[1, 0]
        ax3.bar(methods_count.keys(), methods_count.values(), color='seagreen', alpha=0.7)
        ax3.set_xlabel('Метод аппроксимации')
        ax3.set_ylabel('Количество скважин')
        ax3.set_title('Частота использования методов')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)

    # R^2 по методам
    method_r2 = {}
    for r in results:
        if r is not None:
            if r['method'] not in method_r2:
                method_r2[r['method']] = []
            method_r2[r['method']].append(r['r2'])

    if method_r2:
        ax4 = axes[1, 1]
        box_data = [method_r2[m] for m in methods_count.keys()]
        ax4.boxplot(box_data, labels=methods_count.keys())
        ax4.set_xlabel('Метод аппроксимации')
        ax4.set_ylabel('R²')
        ax4.set_title('Распределение R² по методам')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "analytical_plots.png"), dpi=150, bbox_inches='tight')
    plt.close(fig4)

    logger.info(f"Сохранены групповые и аналитические графики")


def main():
    """Основная функция"""
    start_time = time.time()
    stage_times = {}

    try:
        # Этап 1: Чтение данных
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("НАЧАЛО ВЫПОЛНЕНИЯ СКРИПТА")
        logger.info("=" * 60)

        input_file = "Темпы.xlsx"
        if not os.path.exists(input_file):
            logger.error(f"Файл {input_file} не найден!")
            logger.info("Создайте файл Темпы.xlsx в текущей директории")
            return

        # Используем специальную функцию для чтения файла
        df = read_excel_file(input_file)

        stage_times['чтение_данных'] = time.time() - t0

        # Этап 2: Подготовка данных для параллельной обработки
        t0 = time.time()

        well_data_list = []
        for well_name in df.columns:
            data_series = df[well_name].dropna()
            if len(data_series) >= CONFIG['min_points']:
                well_data_list.append((well_name, data_series, CONFIG))

        logger.info(f"Подготовлено {len(well_data_list)} скважин для обработки")
        stage_times['подготовка_данных'] = time.time() - t0

        if len(well_data_list) == 0:
            logger.error("Нет скважин с достаточным количеством данных!")
            return

        # Этап 3: Параллельная обработка скважин
        t0 = time.time()
        n_workers = min(CONFIG['parallel_workers'], len(well_data_list))
        logger.info(f"Запуск параллельной обработки на {n_workers} ядрах...")

        # Используем параллельную обработку
        with Pool(processes=n_workers) as pool:
            results = pool.map(process_well, well_data_list)

        results = [r for r in results if r is not None]
        logger.info(f"Успешно обработано {len(results)} из {len(well_data_list)} скважин")

        if len(results) == 0:
            logger.error("Нет успешно обработанных скважин!")
            return

        stage_times['обработка_скважин'] = time.time() - t0

        # Этап 4: Создание выходной директории и сохранение результатов
        t0 = time.time()

        output_dir, plots_dir = create_output_directory()
        logger.info(f"Создана директория результатов: {output_dir}")

        # Сохраняем результаты
        save_rates_excel(results, output_dir)
        df_stats = save_analytical_results(results, output_dir)

        stage_times['сохранение_таблиц'] = time.time() - t0

        # Этап 5: Создание графиков
        t0 = time.time()

        plot_all_wells(results, plots_dir, CONFIG)
        plot_grouped_curves(results, output_dir, CONFIG)

        stage_times['создание_графиков'] = time.time() - t0

        # Вывод статистики по времени
        total_time = time.time() - start_time

        logger.info("=" * 60)
        logger.info("ВРЕМЯ ВЫПОЛНЕНИЯ ЭТАПОВ:")
        logger.info("-" * 60)
        for stage, duration in stage_times.items():
            logger.info(f"  {stage:25s}: {duration:6.2f} сек ({duration / total_time * 100:5.1f}%)")
        logger.info("-" * 60)
        logger.info(f"  {'ОБЩЕЕ ВРЕМЯ':25s}: {total_time:6.2f} сек")
        logger.info("=" * 60)

        # Сохраняем информацию о времени
        time_stats = pd.DataFrame([
            {'Этап': stage, 'Время_сек': round(duration, 2), 'Доля_%': round(duration / total_time * 100, 1)}
            for stage, duration in stage_times.items()
        ])
        time_stats.to_excel(os.path.join(output_dir, "time_statistics.xlsx"), index=False)

        # Вывод сводной статистики
        if not df_stats.empty:
            logger.info("\n" + "=" * 60)
            logger.info("СВОДНАЯ СТАТИСТИКА:")
            logger.info("-" * 60)
            logger.info(f"Всего скважин: {len(df_stats)}")
            logger.info(f"Средний R²: {df_stats['R^2'].mean():.4f}")
            logger.info(f"Медианный R²: {df_stats['R^2'].median():.4f}")
            logger.info(f"Мин R²: {df_stats['R^2'].min():.4f}")
            logger.info(f"Макс R²: {df_stats['R^2'].max():.4f}")
            logger.info("-" * 60)

            # Топ методов
            method_popularity = df_stats['Метод аппроксимации'].value_counts()
            logger.info("Популярность методов аппроксимации:")
            for method, count in method_popularity.items():
                logger.info(f"  {method}: {count} скв. ({count / len(df_stats) * 100:.1f}%)")

        logger.info("=" * 60)
        logger.info(f"РЕЗУЛЬТАТЫ СОХРАНЕНЫ В: {output_dir}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Для Windows необходимо добавить этот блок
    import multiprocessing

    multiprocessing.freeze_support()
    main()