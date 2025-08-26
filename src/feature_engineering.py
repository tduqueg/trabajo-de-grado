import os
from typing import Optional

import pandas as pd
import numpy as np


def prepare_m5_features(data_dir: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Prepara las variables de entrenamiento para el dataset M5.

    Parameters
    ----------
    data_dir : str
        Directorio donde residen `sales_train_validation.csv`,
        `calendar.csv` y `sell_prices.csv`.
    nrows : Optional[int]
        Número de filas a cargar de `sales_train_validation.csv` para
        pruebas.  Si es `None`, se cargará el archivo completo (lo que
        puede requerir mucha memoria).

    Returns
    -------
    df_features : pandas.DataFrame
        DataFrame largo que contiene columnas de ventas, fechas,
        variables de calendario, precios y lags/ventanas móviles.
    """
    # Cargar archivos
    sales_path = os.path.join(data_dir, "sales_train_validation.csv")
    calendar_path = os.path.join(data_dir, "calendar.csv")
    prices_path = os.path.join(data_dir, "sell_prices.csv")

    sales = pd.read_csv(sales_path, nrows=nrows)
    calendar = pd.read_csv(calendar_path)
    prices = pd.read_csv(prices_path)

    # Convertir de wide a long
    id_cols = ['id','item_id','dept_id','cat_id','store_id','state_id']
    sales_long = sales.melt(id_vars=id_cols, var_name='d', value_name='sales')

    # Merge con calendario para obtener fecha y variables temporales
    df = sales_long.merge(calendar[['d','date','wm_yr_wk','weekday','wday','month','year',
                                    'event_name_1','event_type_1','snap_CA','snap_TX','snap_WI']],
                          on='d', how='left')

    # Merge con precios
    df = df.merge(prices, on=['store_id','item_id','wm_yr_wk'], how='left')

    # Convertir fecha a datetime y crear features temporales
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # Ingenieria de precio
    df['avg_sell_price'] = df.groupby('id')['sell_price'].transform('mean')
    df['rel_price'] = df['sell_price'] / df['avg_sell_price']

    # Lags y ventanas móviles de ventas
    lags = [7, 28, 56, 365]
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('id')['sales'].shift(lag)

    windows = [7, 28, 56]
    for window in windows:
        df[f'rolling_mean_{window}'] = df.groupby('id')['sales'].shift(1).rolling(window).mean()
        df[f'rolling_std_{window}'] = df.groupby('id')['sales'].shift(1).rolling(window).std()

    return df


def prepare_olist_features(data_dir: str) -> pd.DataFrame:
    """Genera variables para demanda e inventario a partir del dataset Olist.

    Parameters
    ----------
    data_dir : str
        Directorio que contiene los archivos de Olist.

    Returns
    -------
    df_features : pandas.DataFrame
        DataFrame con la serie temporal de unidades vendidas por producto
        y día, incluyendo lags, estadísticas por producto y variables
        derivadas.
    """
    # Cargar datasets
    orders = pd.read_csv(
        os.path.join(data_dir, "olist_orders_dataset.csv"),
        parse_dates=[
            'order_purchase_timestamp',
            'order_approved_at',
            'order_delivered_carrier_date',
            'order_delivered_customer_date',
            'order_estimated_delivery_date',
        ],
    )
    items = pd.read_csv(os.path.join(data_dir, "olist_order_items_dataset.csv"))
    payments = pd.read_csv(os.path.join(data_dir, "olist_order_payments_dataset.csv"))
    reviews = pd.read_csv(
        os.path.join(data_dir, "olist_order_reviews_dataset.csv"),
        parse_dates=['review_creation_date', 'review_answer_timestamp'],
    )
    customers = pd.read_csv(os.path.join(data_dir, "olist_customers_dataset.csv"))
    products = pd.read_csv(os.path.join(data_dir, "olist_products_dataset.csv"))
    translations = pd.read_csv(os.path.join(data_dir, "product_category_name_translation.csv"))

    # Traducir categorías
    products = products.merge(translations, on='product_category_name', how='left')
    products['category_en'] = products['product_category_name_english']

    # Unir tablas para crear dataset de ventas por ítem
    order_data = items.merge(orders[['order_id','customer_id','order_purchase_timestamp']], on='order_id')
    order_data = order_data.merge(customers[['customer_id','customer_unique_id','customer_zip_code_prefix']], on='customer_id')
    order_data = order_data.merge(
        products[['product_id','category_en','product_weight_g','product_length_cm','product_height_cm','product_width_cm']],
        on='product_id', how='left'
    )
    order_data = order_data.merge(
        payments[['order_id','payment_type','payment_installments','payment_value']],
        on='order_id', how='left'
    )
    order_data = order_data.merge(reviews[['order_id','review_score']], on='order_id', how='left')

    # Fecha de la orden (día)
    order_data['order_date'] = order_data['order_purchase_timestamp'].dt.date

    # Cantidad vendida por producto y día 
    demand_df = order_data.groupby(['product_id','order_date']).size().reset_index(name='units')

    # Calcular lags semanales y mensuales
    demand_df = demand_df.sort_values(['product_id','order_date'])
    for lag in [7, 14, 28]:
        demand_df[f'lag_{lag}'] = demand_df.groupby('product_id')['units'].shift(lag)

    # Estadísticas adicionales por producto
    product_stats = order_data.groupby('product_id').agg(
        avg_price=('payment_value', 'mean'),
        avg_review=('review_score', 'mean'),
        avg_weight=('product_weight_g','mean'),
        avg_length=('product_length_cm','mean'),
        avg_height=('product_height_cm','mean'),
        avg_width=('product_width_cm','mean'),
        n_orders=('order_id','nunique')
    ).reset_index()

    # Fusionar estadísticas con demanda
    df_features = demand_df.merge(product_stats, on='product_id', how='left')

    return df_features
