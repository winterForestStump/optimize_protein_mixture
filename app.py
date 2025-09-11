import streamlit as st
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

st.header("Исходные продукты")
col1, col2, col3, col4 = st.columns(4)
with col1:
    product_1 = st.text_input("Название продукта", "Соя", key="product_1")
    protein_conc_1 = st.number_input("Содержание протеина в продукте", key="protein_conc_1", min_value = 0, max_value=100)
    protein_price_1 = st.number_input("Стоимость продукта, руб за тонну", key="protein_price_1", min_value = 0, step = 10)

with col2:
    product_2 = st.text_input("Название продукта", "Горох", key="product_2")
    protein_conc_2 = st.number_input("Содержание протеина в продукте", key="protein_conc_2", min_value = 0, max_value=100)
    protein_price_2 = st.number_input("Стоимость продукта, руб за тонну", key="protein_price_2", min_value = 0, step = 10)

with col3:
    product_3 = st.text_input("Название продукта", key="product_3")
    protein_conc_3 = st.number_input("Содержание протеина в продукте", key="protein_conc_3", min_value = 0, max_value=100)
    protein_price_3 = st.number_input("Стоимость продукта, руб за тонну", key="protein_price_3", min_value = 0, step = 10)

with col4:
    product_4 = st.text_input("Название продукта", key="product_4")
    protein_conc_4 = st.number_input("Содержание протеина в продукте", key="protein_conc_4", min_value = 0, max_value=100)
    protein_price_4 = st.number_input("Стоимость продукта, руб за тонну", key="protein_price_4", min_value = 0, step = 10)

st.header("Концентрация протеина в конечном продукте")
min_protein = st.number_input("Введите минимальное содержание протеина в конечной смеси, %", key="min_protein", min_value = 0, max_value=100)

st.header("Цена реализации конечного продукта")
mixture_price_input = st.number_input("Введите цену реализации конечной смеси, руб за тонну", key="mixture_price_input", min_value = 0, step = 10)

products_data = []
if product_1 and protein_conc_1 > 0 and protein_price_1 > 0:
    products_data.append({
        'name': product_1,
        'protein_concentration': protein_conc_1,
        'price_per_kg': protein_price_1
    })

if product_2 and protein_conc_2 > 0 and protein_price_2 > 0:
    products_data.append({
        'name': product_2,
        'protein_concentration': protein_conc_2,
        'price_per_kg': protein_price_2
    })

if product_3 and protein_conc_3 > 0 and protein_price_3 > 0:
    products_data.append({
        'name': product_3,
        'protein_concentration': protein_conc_3,
        'price_per_kg': protein_price_3
    })

if product_4 and protein_conc_4 > 0 and protein_price_4 > 0:
    products_data.append({
        'name': product_4,
        'protein_concentration': protein_conc_4,
        'price_per_kg': protein_price_4
    })

def optimize_protein_mixture_multi(products_data, min_protein, mixture_price=None):
    """
    Optimize protein mixture with multiple products to minimize cost while meeting protein requirements.
    
    Parameters:
    products_data - list of dictionaries with 'protein_concentration' and 'price_per_kg'
    min_protein - minimum protein concentration in mixture (%)
    mixture_price - price of final mixture (per kg) - optional for profit calculation
    
    Returns:
    dict with optimization results
    """
    
    n_products = len(products_data)

    if n_products == 0:
        st.error("ОШИБКА: Не введено ни одного продукта для оптимизации!")
        return None

    if all(product['price_per_kg'] == 0 for product in products_data):
        st.error("ОШИБКА: У всех продуктов указана нулевая стоимость!")
    
    # Check if problem is feasible
    max_protein = max(product['protein_concentration'] for product in products_data)
    if max_protein < min_protein:
        st.error("НЕВЫПОЛНИМАЯ ЗАДАЧА!")
        st.error(f"Максимальная доступная концентрация протеина ({max_protein:.2f}%) меньше требуемой ({min_protein:.2f}%)")

    # Objective function coefficients (minimize cost)
    # Cost = sum(price_i * share_i) for all products
    c = [product['price_per_kg'] for product in products_data]
    
    # Inequality constraint matrix (A_ub * x <= b_ub)
    # We need: sum(protein_i * share_i) >= min_protein
    # Rewritten as: -sum(protein_i * share_i) <= -min_protein
    A_ub = [[-product['protein_concentration'] for product in products_data]]
    b_ub = [-min_protein]
    
    # Equality constraint matrix (A_eq * x = b_eq)
    # We need: sum(share_i) = 1
    A_eq = [[1] * n_products]
    b_eq = [1]
    
    # Bounds for variables (all shares >= 0, <= 1)
    bounds = [(0, 1)] * n_products
    
    # Solve the optimization problem
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                     bounds=bounds, method='highs')
    
    if result.success:
        optimal_shares = result.x
        optimal_cost = result.fun
        
        # Calculate actual protein concentration
        protein_concentration = sum(products_data[i]['protein_concentration'] * optimal_shares[i] 
                                  for i in range(n_products))
        
        # Calculate profit if mixture_price is provided
        profit = mixture_price - optimal_cost if mixture_price else None

        st.header("=== РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ ===")
        st.write(f"Использовано продуктов в расчете: {n_products}")
        with st.container(border=True):
            st.write("Оптимальная композиция для смеси и разбивка затрат:")
            cols = st.columns(n_products)        
            for i, col in enumerate(cols):
                with col:
                    st.write(f"  {products_data[i]['name']}: {optimal_shares[i]*100:.2f}%")
                    cost_contribution = products_data[i]['price_per_kg'] * optimal_shares[i]
                    st.write(f"  {products_data[i]['name']}: РУБ {cost_contribution:.2f}")
            st.write(f"Общая стоимость смеси за тонну: РУБ {optimal_cost:.2f}")
        
        with st.container(border=True):
            st.write(f"Полученная концентрация протеина: {protein_concentration:.2f}%")
            st.write(f"Минимальная концентрация протеина: {min_protein:.2f}%")
            st.write(f"Требование соблюдено: {protein_concentration >= min_protein}")
            
        if profit is not None:
            with st.container(border=True):
                st.write("Анализ прибыли:")
                st.write(f"  Цена смеси для продажи за тонну: РУБ {mixture_price:.2f}")
                st.write(f"  Затраты на смесь: РУБ {optimal_cost:.2f}")
                st.write(f"  Прибыль на тонну: РУБ {profit:.2f}")
                st.write(f"  Маржа прибыли: {(profit/mixture_price*100):.1f}%")
    
    else:
        st.write("Ошибка оптимизации!")


if st.button("Рассчитать", type="primary"):
    st.write(f"Продуктов для расчета: {len(products_data)}")
    if len(products_data) > 0:
        optimize_protein_mixture_multi(products_data, min_protein, mixture_price_input)
    else:
        st.error("Добавьте хотя бы один продукт с названием, содержанием протеина и ценой!")
if st.button("Сбросить", type="primary"):
    st.rerun()