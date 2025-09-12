import streamlit as st
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

st.header("Исходные продукты")
col1, col2, col3, col4 = st.columns(4, border=True)
with col1:
    product_1 = st.text_input("Название продукта", "Соя", key="product_1")
    protein_conc_1 = st.number_input("Содержание протеина в продукте", key="protein_conc_1", min_value = 0, max_value=100)
    protein_price_1 = st.number_input("Стоимость продукта, руб за тонну", key="protein_price_1", min_value = 0.00, step = 1.00)
with col2:
    product_2 = st.text_input("Название продукта", "Горох", key="product_2")
    protein_conc_2 = st.number_input("Содержание протеина в продукте", key="protein_conc_2", min_value = 0, max_value=100)
    protein_price_2 = st.number_input("Стоимость продукта, руб за тонну", key="protein_price_2", min_value = 0.00, step = 1.00)
with col3:
    product_3 = st.text_input("Название продукта", key="product_3")
    protein_conc_3 = st.number_input("Содержание протеина в продукте", key="protein_conc_3", min_value = 0, max_value=100)
    protein_price_3 = st.number_input("Стоимость продукта, руб за тонну", key="protein_price_3", min_value = 0.00, step = 1.00)
with col4:
    product_4 = st.text_input("Название продукта", key="product_4")
    protein_conc_4 = st.number_input("Содержание протеина в продукте", key="protein_conc_4", min_value = 0, max_value=100)
    protein_price_4 = st.number_input("Стоимость продукта, руб за тонну", key="protein_price_4", min_value = 0.00, step = 1.00)

with st.container(border=True):
    st.header("Добавление карбамида")
    use_urea = st.checkbox("Использовать карбамид", value=False)
    if use_urea:
        urea_percentage = st.slider("Добавление карбамида, % от массы смеси", 
                                min_value=0.0, max_value=3.0, value=1.0, step=0.1)
        urea_price = st.number_input("Стоимость карбамида, руб за тонну", 
                                    min_value=0.00, value=10000.00, step=1.00)
    else:
        urea_percentage = 0.0
        urea_price = 0.0

with st.container(border=True):
    st.header("Данные конечного продукта")
    min_protein = st.number_input("Введите минимальное содержание протеина в конечной смеси, %", key="min_protein", min_value = 0, max_value=100)
    mixture_price_input = st.number_input("Введите цену реализации конечной смеси, руб за тонну", key="mixture_price_input", min_value = 0.00, step = 1.00)

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

def optimize_protein_mixture_multi(products_data, min_protein, mixture_price=None, use_urea=False, urea_percentage=0.0, urea_price=0.0):
    """
    Optimize protein mixture with multiple products to minimize cost while meeting protein requirements.
    
    Parameters:
    products_data - list of dictionaries with 'protein_concentration' and 'price_per_kg'
    min_protein - minimum protein concentration in mixture (%)
    mixture_price - price of final mixture (per kg) - optional for profit calculation
    use_urea - whether to use urea
    urea_percentage - percentage of urea in final mixture
    urea_price - price of urea per kg
    """
    
    n_products = len(products_data)

    if n_products == 0:
        st.error("ОШИБКА: Не введено ни одного продукта для оптимизации!")
        return None

    if all(product['price_per_kg'] == 0 for product in products_data):
        st.error("ОШИБКА: У всех продуктов указана нулевая стоимость!")
    
    # Adjust for urea
    if use_urea and urea_percentage > 0:
        # Urea contributes 2.875% "protein" per 1% of urea (46% nitrogen * 6.25)
        urea_protein_contribution = urea_percentage * 2.875
        # Reduce required protein from agricultural products
        effective_min_protein = max(0, min_protein - urea_protein_contribution)
        # Adjust total shares for agricultural products
        agri_share = 1 - urea_percentage / 100
    else:
        urea_protein_contribution = 0.0
        effective_min_protein = min_protein
        agri_share = 1.0

    # Check if problem is feasible
    max_protein = max(product['protein_concentration'] for product in products_data)
    if max_protein * agri_share < effective_min_protein:
        st.error("НЕВЫПОЛНИМАЯ ЗАДАЧА!")
        st.error(f"Максимальная доступная концентрация протеина ({max_protein * agri_share:.2f}%) меньше требуемой ({effective_min_protein:.2f}%)")

    # Objective function coefficients (minimize cost)
    # Cost = sum(price_i * share_i) for all products
    c = [product['price_per_kg'] for product in products_data]
    
    # Inequality constraint matrix (A_ub * x <= b_ub)
    # We need: sum(protein_i * share_i) >= effective_min_protein
    # Rewritten as: -sum(protein_i * share_i) <= -effective_min_protein
    A_ub = [[-product['protein_concentration'] for product in products_data]]
    b_ub = [-effective_min_protein]
    
    # Equality constraint matrix (A_eq * x = b_eq)
    # We need: sum(share_i) = agri_share
    A_eq = [[1] * n_products]
    b_eq = [agri_share]
    
    # Bounds for variables (all shares >= 0, <= 1)
    bounds = [(0, 1)] * n_products
    
    # Solve the optimization problem
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                     bounds=bounds, method='highs')
    
    if result.success:
        optimal_shares = result.x
        agricultural_cost = result.fun

        # Calculate urea cost
        urea_cost = (urea_percentage / 100) * urea_price if use_urea else 0.0
        optimal_cost = agricultural_cost + urea_cost
        
        # Calculate protein concentrations
        agricultural_protein = sum(products_data[i]['protein_concentration'] * optimal_shares[i] 
                                  for i in range(n_products))
        
        total_protein = agricultural_protein + urea_protein_contribution
        
        st.header("=== РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ ===")
                
        with st.container(border=True):
            st.write("Оптимальная композиция для смеси и разбивка затрат:")
            # Agricultural products
            for i in range(n_products):
                if optimal_shares[i] > 0:
                    share_percent = optimal_shares[i] * 100
                    cost_contribution = products_data[i]['price_per_kg'] * optimal_shares[i]
                    st.write(f"  {products_data[i]['name']}: {share_percent:.2f}% - РУБ {cost_contribution:.2f}")

            # Urea if used
            if use_urea and urea_percentage > 0:
                st.write(f"  Карбамид: {urea_percentage:.1f}% - РУБ {urea_cost:.2f}")
                st.write(f"  Сельхозпродукция: {100 - urea_percentage:.1f}% - РУБ {agricultural_cost:.2f}")
            
            st.write(f"Общая стоимость смеси за тонну: РУБ {optimal_cost:.2f}")


        with st.container(border=True):
            st.write("Содержание протеина в смеси:")
            if use_urea and urea_percentage > 0:
                st.write(f"  От сельхозпродукции: {agricultural_protein:.2f}%")
                st.write(f"  От карбамида: {urea_protein_contribution:.2f}%")
            st.write(f"  Общий видимый белок: {total_protein:.2f}%")
            st.write(f"  Минимальная концентрация протеина: {min_protein:.2f}%")
            st.write(f"  Требование соблюдено: {total_protein >= min_protein}")

        if mixture_price is not None:
            with st.container(border=True):
                st.write("Анализ прибыли:")
                st.write(f"  Цена смеси для продажи за тонну: РУБ {mixture_price:.2f}")
                st.write(f"  Затраты на смесь: РУБ {optimal_cost:.2f}")
                profit = mixture_price - optimal_cost
                st.write(f"  Прибыль на тонну: РУБ {profit:.2f}")
                st.write(f"  Маржа прибыли: {(profit/mixture_price*100):.1f}%")

    else:
        st.write("Ошибка оптимизации!")


if st.button("Рассчитать", type="primary"):
    st.write(f"Продуктов для расчета: {len(products_data)}")
    if len(products_data) > 0:
        optimize_protein_mixture_multi(products_data, min_protein, mixture_price_input, use_urea, urea_percentage, urea_price)
    else:
        st.error("Добавьте хотя бы один продукт с названием, содержанием протеина и ценой!")
if st.button("Сбросить", type="primary"):
    st.rerun()