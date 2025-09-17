import streamlit as st
import numpy as np
from scipy.optimize import linprog
import io
from datetime import datetime
import pandas as pd
import copy

if 'calculation_done' not in st.session_state:
    st.session_state.calculation_done = False
if 'solutions' not in st.session_state:
    st.session_state.solutions = None
if 'products_data' not in st.session_state:
    st.session_state.products_data = []
if 'all_solutions' not in st.session_state:
    st.session_state.all_solutions = []

# Create initial DataFrame
def create_initial_df():
    return pd.DataFrame({
        'Название продукта': ['', '', '', ''],
        'Содержание протеина в продукте': [0, 0, 0, 0],
        'Стоимость продукта, руб за тонну': [0, 0, 0, 0],
        'Остаток продукта на складе, тонн': [0, 0, 0, 0]
    })

# Uploading and saving data
if 'products_df' not in st.session_state:
    st.session_state.products_df = create_initial_df()

with st.expander("Краткое описание", expanded=False):
    st.write('''
Это веб-приложение рассчитывает оптимальные рецептуры комбикорма для достижения заданного содержания протеина с максимальной прибылью. 
Оно находит наилучшие варианты сочетания ингредиентов с учетом остатков на складе и выполняет многоэтапную оптимизацию, пока остатки продуктов позволяют производить прибыльные партии смеси.
''')
with st.expander("Инструкция по использованию", expanded=False):
    st.write('''
**1. Заполните данные по продуктам**
- Используйте таблицу ниже для ввода данных о доступных продуктах.
- Добавляйте/удаляйте строки с помощью кнопок в таблице.
- Заполните все колонки: название, содержание протеина (%), стоимость (руб/т) и остаток на складе (т).
- Вы можете сохранить заполненную таблицу в CSV формате, чтобы в дальнейшем загружать её автоматически.

**2. Настройте добавление карбамида (опционально)**
- Поставьте галочку "Использовать карбамид", если он нужен.
- С помощью ползунка задайте, какой процент от массы смеси будет занимать карбамид (до 3%).
- Укажите цену карбамида за тонну.

**3. Задайте параметры конечной смеси**
- Введите минимальное требуемое содержание протеина (%) в вашей смеси.
- Укажите цену реализации вашей смеси за тонну, чтобы рассчитать прибыль.

**4. Запустите расчет и анализируйте результаты**
- Нажмите кнопку «Рассчитать».
- Приложение выполнит многоэтапный расчет, пока из остатков можно собрать прибыльную партию.
- Результаты появятся ниже. Вы увидите несколько "партий" - оптимальных вариантов состава смеси с максимальной прибылью для каждого этапа использования остатков.
- Для каждой партии отображается состав, финансовые показатели и остатки продуктов после производства.
             
**5. Работа с данными и результатами**
- Используйте кнопку "Загрузить данные из CSV" для импорта данных.
- После расчета используйте кнопку "Скачать полный отчет (TXT)" для сохранения детализированных результатов.
- Кнопка "Сбросить данные" очистит все введенные данные и результаты для нового расчета.
''')

with st.container(border=True):
    # Button for uploading CSV data
    uploaded_file = st.file_uploader("Загрузить данные из CSV", type=['csv'])
    if uploaded_file is not None:
        try:
            st.session_state.products_df = pd.read_csv(uploaded_file)
            st.success("Данные успешно загружены!")
        except Exception as e:
            st.error(f"Ошибка при загрузке файла: {e}")

    st.header("Исходные продукты")

    # Eddditing product table
    edited_df = st.data_editor(
        st.session_state.products_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Название продукта": st.column_config.TextColumn(
                "Название продукта",
                required=True,
            ),
            "Содержание протеина в продукте": st.column_config.NumberColumn(
                "Содержание протеина, %",
                min_value=0.0,
                max_value=100.0,
                format="%.2f",
                required=True
            ),
            "Стоимость продукта, руб за тонну": st.column_config.NumberColumn(
                "Стоимость, руб/т",
                min_value=0.00,
                format="%.2f",
                required=True
            ),
            "Остаток продукта на складе, тонн": st.column_config.NumberColumn(
                "Остаток, тонн",
                min_value=0.00,
                format="%.2f",
                required=True
            )
        },
        hide_index=True
    )

# Renew session state
st.session_state.products_df = edited_df

with st.container(border=True):
    st.header("Добавление карбамида")
    use_urea = st.checkbox("Использовать карбамид", value=False)
    if use_urea:
        urea_percentage = st.slider("Добавление карбамида, % от массы смеси", 
                                min_value=0.0, max_value=3.0, value=1.0, step=0.1)
        urea_price = st.number_input("Стоимость карбамида, руб за тонну", 
                                    min_value=0.00, value=38000.00, step=1.00)
    else:
        urea_percentage = 0.0
        urea_price = 0.0

with st.container(border=True):
    st.header("Данные конечного продукта")
    min_protein = st.number_input("Введите минимальное содержание протеина в конечной смеси, %", 
                                 key="min_protein", min_value=0, max_value=100, value=30)
    mixture_price_input = st.number_input("Введите цену реализации конечной смеси, руб за тонну", 
                                        key="mixture_price_input", min_value=0.00, value=0.00, step=1.00)

# Preparing products data
products_data = []
for _, row in edited_df.iterrows():
    if (pd.notna(row['Название продукта']) and 
        row['Название продукта'].strip() != '' and
        row['Содержание протеина в продукте'] > 0 and
        row['Стоимость продукта, руб за тонну'] > 0):
        products_data.append({
            'name': row['Название продукта'],
            'protein_concentration': row['Содержание протеина в продукте'],
            'price_per_kg': row['Стоимость продукта, руб за тонну'],
            'stock_available': row['Остаток продукта на складе, тонн']
        })


def optimize_protein_mixture_max_profit(products_data, min_protein, mixture_price=None, 
                                      use_urea=False, urea_percentage=0.0, urea_price=0.0):
    """
    Optimize protein mixture to maximize profit with given constraints.
    Returns the optimal solution for maximum profit.
    """
    
    n_products = len(products_data)

    if n_products == 0:
        return None

    if all(product['price_per_kg'] == 0 for product in products_data):
        return None
    
    # Adjust for urea
    if use_urea and urea_percentage > 0:
        urea_protein_contribution = urea_percentage * 2.875
        effective_min_protein = max(0, min_protein - urea_protein_contribution)
        agri_share = 1 - urea_percentage / 100
    else:
        urea_protein_contribution = 0.0
        effective_min_protein = min_protein
        agri_share = 1.0

    # Check if problem is feasible
    max_protein = max(product['protein_concentration'] for product in products_data)
    if max_protein * agri_share < effective_min_protein:
        return None

    solutions = []
    
    # MAX PROFIT GOAL
    try:
        # Objective function coefficients (minimize cost)
        # Cost = sum(price_i * share_i) for all products
        c_cost = [product['price_per_kg'] for product in products_data]
        
        # Inequality constraint matrix (A_ub * x <= b_ub)
        # We need: sum(protein_i * share_i) >= effective_min_protein
        # Rewritten as: -sum(protein_i * share_i) <= -effective_min_protein
        A_ub_cost = [[-product['protein_concentration'] for product in products_data]]
        b_ub_cost = [-effective_min_protein]
        
        # Equality constraint matrix (A_eq * x = b_eq)
        # We need: sum(share_i) = agri_share
        A_eq_cost = [[1] * n_products]
        b_eq_cost = [agri_share]
        
        # Bounds for variables (all shares >= 0, <= 1)
        bounds_cost = [(0, 1)] * n_products
        
        # Solve the optimization problem
        result_cost = linprog(c_cost, A_ub=A_ub_cost, b_ub=b_ub_cost, 
                             A_eq=A_eq_cost, b_eq=b_eq_cost, 
                             bounds=bounds_cost, method='highs')
        
        if result_cost.success:
            optimal_shares = result_cost.x
            
            # Max amount for the max profit
            limiting_product = None
            max_possible_volume = float('inf')
            for i in range(n_products):
                if optimal_shares[i] > 0:
                    product_max = products_data[i]['stock_available'] / optimal_shares[i]
                    if product_max < max_possible_volume:
                        max_possible_volume = product_max
                        limiting_product = products_data[i]['name']
            
            optimal_volume = max_possible_volume
            
            # Calculate finansials
            agricultural_cost = sum(products_data[i]['price_per_kg'] * optimal_shares[i] * optimal_volume 
                                  for i in range(n_products))
            
            urea_cost = (urea_percentage / 100) * urea_price * optimal_volume if use_urea else 0.0
            total_cost = agricultural_cost + urea_cost
            
            revenue = mixture_price * optimal_volume
            profit = revenue - total_cost
            
            cost_per_ton = result_cost.fun + (urea_percentage / 100 * urea_price if use_urea else 0)
            profit_per_ton = mixture_price - cost_per_ton
            
            agricultural_protein = sum(products_data[i]['protein_concentration'] * optimal_shares[i] 
                                     for i in range(n_products))
            total_protein = agricultural_protein + urea_protein_contribution
            
            # Calculate products remains
            remaining_stock = []
            new_products_data = []
            for i in range(n_products):
                used_amount = optimal_shares[i] * optimal_volume
                remaining = max(0, products_data[i]['stock_available'] - used_amount)
                remaining_stock.append({
                    'name': products_data[i]['name'],
                    'remaining': remaining
                })
                # New data for the next itteration
                if remaining > 0:
                    new_products_data.append({
                        'name': products_data[i]['name'],
                        'protein_concentration': products_data[i]['protein_concentration'],
                        'price_per_kg': products_data[i]['price_per_kg'],
                        'stock_available': remaining
                    })
            
            solutions.append({
                            'type': 'max_profit',
                            'shares': optimal_shares.copy(),
                            'mixture_volume': optimal_volume,
                            'total_cost': total_cost,
                            'revenue': revenue,
                            'profit': profit,
                            'profit_per_ton': profit_per_ton,
                            'total_protein': total_protein,
                            'description': f'Максимальная прибыль',
                            'limiting_product': limiting_product,
                            'remaining_stock': remaining_stock,
                            'new_products_data': new_products_data,
                            'used_products': copy.deepcopy(products_data)  # сохраняем список продуктов
                        })
    except Exception as e:
        st.warning(f"Ошибка в расчете варианта максимальной прибыли: {e}")
    
    return solutions

def recursive_optimization(initial_products_data, min_protein, mixture_price, 
                          use_urea, urea_percentage, urea_price, max_iterations=10):
    
    all_solutions = []
    current_products_data = copy.deepcopy(initial_products_data)
    iteration = 0
    
    while iteration < max_iterations and current_products_data:
        iteration += 1
        
        solution = optimize_protein_mixture_max_profit(
            current_products_data, min_protein, mixture_price,
            use_urea, urea_percentage, urea_price
        )
        
        if not solution:
            break
            
        current_solution = solution[0]

        #Check the positive financial result
        if current_solution['profit'] <= 0:
            st.warning(f"Оптимизация остановлена: партия #{iteration} дала бы отрицательный финансовый результат ({current_solution['profit']:,.2f} руб).")
            break
        
        all_solutions.append(current_solution)
        
        # Check if any more products for the next iteration
        remaining_products = []
        total_remaining = 0
        
        for product in current_solution['new_products_data']:
            if product['stock_available'] > 0.001:  # Ignore little remains
                remaining_products.append(product)
                total_remaining += product['stock_available']
        
        # Check if it is possible to make new mixture from the remaining products
        if not remaining_products or total_remaining < 1.0:  # Min amount
            break
            
        # Check it it's anough protein in remaining products
        max_protein_remaining = max(p['protein_concentration'] for p in remaining_products)
        if use_urea and urea_percentage > 0:
            urea_contribution = urea_percentage * 2.875
            effective_min_protein = max(0, min_protein - urea_contribution)
            agri_share = 1 - urea_percentage / 100
        else:
            effective_min_protein = min_protein
            agri_share = 1.0
            
        if max_protein_remaining * agri_share < effective_min_protein:
            break
            
        current_products_data = remaining_products
    
    return all_solutions

def display_solution(solution, use_urea, urea_percentage, urea_price, batch_number):
    """Display the optimal solution in the UI with DataFrame table"""
    
    products_data = solution['used_products']

    with st.container(border=True):
        st.subheader(f"Партия #{batch_number}")
        
        # Main indicators
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Объем смеси", f"{solution['mixture_volume']:,.2f} т")
        with col2:
            st.metric("Общая прибыль", f"{solution['profit']:,.2f} руб")
        with col3:
            st.metric("Прибыль на тонну", f"{solution['profit_per_ton']:,.2f} руб/т")
        
        st.write("**Состав смеси:**")
        
        comp_data = []
        
        for i in range(len(products_data)):
            if solution['shares'][i] > 1e-6: 
                share_percent = solution['shares'][i] * 100
                amount_required = solution['shares'][i] * solution['mixture_volume']
                cost_in_mixture = products_data[i]['price_per_kg'] * solution['shares'][i]

                comp_data.append({
                    "Продукт": products_data[i]['name'],
                    "Доля в смеси, %": f"{share_percent:.3f}",
                    "Требуется, т": f"{amount_required:,.3f}",
                    "Стоимость, руб/т": f"{products_data[i]['price_per_kg']:,.2f}",
                    "Стоимость в смеси, руб/т": f"{cost_in_mixture:.2f}",
                    "Протеин, %": f"{products_data[i]['protein_concentration']:.2f}",
                    "Остаток на складе, т": f"{products_data[i]['stock_available']:,.3f}"
                })


        if use_urea and urea_percentage > 0:
            urea_cost_per_ton = (urea_percentage / 100) * urea_price
            comp_data.append({
                "Продукт": "Карбамид",
                "Доля в смеси, %": f"{urea_percentage:.3f}",
                "Требуется, т": f"{(urea_percentage / 100 * solution['mixture_volume']):,.3f}",
                "Стоимость, руб/т": f"{urea_price:,.2f}",
                "Стоимость в смеси, руб/т": f"{urea_cost_per_ton:.2f}",
                "Протеин, %": f"{urea_percentage * 2.875:.3f}",
                "Остаток на складе, т": "—"
            })
        
        if comp_data:
            comp_df = pd.DataFrame(comp_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        st.write("**Итоговые показатели партии:**")
        
        agricultural_cost_total = sum(products_data[i]['price_per_kg'] * solution['shares'][i] * solution['mixture_volume'] 
                                    for i in range(len(products_data)) if solution['shares'][i] > 1e-6)
        
        urea_cost_total = (urea_percentage / 100) * urea_price * solution['mixture_volume'] if use_urea else 0.0
        
        summary_data = {
            "Показатель": [
                "Общий объем смеси",
                "Стоимость сельхозпродукции",
                "Стоимость карбамида",
                "Общая себестоимость",
                "Выручка от реализации",
                "Общая прибыль",
                "Прибыль на тонну",
                "Содержание протеина"
            ],
            "Значение": [
                f"{solution['mixture_volume']:,.2f} т",
                f"{agricultural_cost_total:,.2f} руб",
                f"{urea_cost_total:,.2f} руб" if use_urea else "0.00 руб",
                f"{solution['total_cost']:,.2f} руб",
                f"{solution['revenue']:,.2f} руб",
                f"{solution['profit']:,.2f} руб",
                f"{solution['profit_per_ton']:,.2f} руб/т",
                f"{solution['total_protein']:.2f}%"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
      
        st.write("**Остатки продуктов:**")
        remaining_data = []
        total_remaining = 0
        for item in solution['remaining_stock']:
            if item['remaining'] > 0.001: 
                remaining_data.append({
                    "Продукт": item['name'],
                    "Остаток, т": f"{item['remaining']:,.3f}"
                })
                total_remaining += item['remaining']
        
        if remaining_data:
            remaining_df = pd.DataFrame(remaining_data)
            st.dataframe(remaining_df, use_container_width=True, hide_index=True)
            st.write(f"Общий остаток: {total_remaining:,.3f} т")
        else:
            st.success("✅ Все продукты полностью использованы!")

if st.button("Рассчитать", type="primary"):
    if len(products_data) > 0:
        st.session_state.all_solutions = recursive_optimization(
            products_data, min_protein, mixture_price_input, 
            use_urea, urea_percentage, urea_price
        )
        st.session_state.calculation_done = True
        st.session_state.products_data = products_data.copy()
        st.rerun()
            
    else:
        st.error("Добавьте хотя бы один продукт с названием, содержанием протеина и ценой!")

if st.button("Сбросить данные", type="secondary"):
    st.session_state.calculation_done = False
    st.session_state.solutions = None
    st.session_state.all_solutions = []
    st.session_state.products_data = []
    st.session_state.products_df = create_initial_df()
    st.rerun()

# Display results if they are exist in session state
if st.session_state.calculation_done and st.session_state.all_solutions:
    st.header("РЕЗУЛЬТАТЫ РАСЧЕТОВ")
    
    total_profit = 0
    total_volume = 0
    total_batches = len(st.session_state.all_solutions)
    
    for i, solution in enumerate(st.session_state.all_solutions, 1):
        if i == 1:
            current_products = st.session_state.products_data
        else:
            current_products = st.session_state.all_solutions[i-2]['new_products_data']
        
        display_solution(solution, use_urea, urea_percentage, urea_price, i)
        total_profit += solution['profit']
        total_volume += solution['mixture_volume']
    

    with st.container(border=True):
        st.subheader("📊 Сводная информация по всем партиям")
        col1, col2, col3, col4 = st.columns([0.15, 0.25, 0.3, 0.3])
        with col1:
            st.metric("Количество партий", total_batches)
        with col2:
            st.metric("Общий объем производства", f"{total_volume:,.2f} т")
        with col3:
            st.metric("Общая прибыль", f"{total_profit:,.2f} руб")
        with col4:
            avg_profit_per_ton = total_profit / total_volume if total_volume > 0 else 0
            st.metric("Средняя прибыль на тонну", f"{avg_profit_per_ton:,.2f} руб/т")
    
    final_solution = st.session_state.all_solutions[-1]
    if final_solution['remaining_stock']:
        with st.container(border=True):
            st.subheader("Финальные остатки продукции")
            remaining_data = []
            total_final_remaining = 0
            for item in final_solution['remaining_stock']:
                if item['remaining'] > 0.001:
                    remaining_data.append({
                        "Продукт": item['name'],
                        "Остаток, т": f"{item['remaining']:,.3f}"
                    })
                    total_final_remaining += item['remaining']
            
            if remaining_data:
                remaining_df = pd.DataFrame(remaining_data)
                st.dataframe(remaining_df, use_container_width=True, hide_index=True)
                st.write(f"Общий неиспользованный остаток: {total_final_remaining:,.3f} т")
                
                max_protein_remaining = max(st.session_state.products_data, 
                                          key=lambda x: x['protein_concentration'])['protein_concentration']
                if use_urea and urea_percentage > 0:
                    effective_min = min_protein - (urea_percentage * 2.875)
                    if max_protein_remaining * (1 - urea_percentage/100) < effective_min:
                        st.error("Остатки невозможно использовать: недостаточное содержание протеина даже с карбамидом")
                elif max_protein_remaining < min_protein:
                    st.error("Остатки невозможно использовать: недостаточное содержание протеина")
            else:
                st.write("Вся продукция полностью использована")
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_text = f"Отчет по оптимизации состава смеси для максимальной прибыли\n"
    report_text += f"Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report_text += f"Всего произведено партий: {total_batches}\n"
    report_text += f"Общий объем производства: {total_volume:.2f} тонн\n"
    report_text += f"Общая прибыль: {total_profit:,.2f} руб\n"
    report_text += f"Средняя прибыль на тонну: {avg_profit_per_ton:,.2f} руб/т\n\n"
    
    for i, solution in enumerate(st.session_state.all_solutions, 1):
        report_text += f"ПАРТИЯ #{i}\n"
        report_text += f"Объем смеси: {solution['mixture_volume']:.2f} тонн\n"
        report_text += f"Прибыль: {solution['profit']:,.2f} руб\n"
        report_text += f"Прибыль на тонну: {solution['profit_per_ton']:,.2f} руб/т\n"
        report_text += "Состав:\n"
        
        if i == 1:
            current_products = st.session_state.products_data
        else:
            current_products = st.session_state.all_solutions[i-2]['new_products_data']
        
        for j in range(len(current_products)):
            if solution['shares'][j] > 0:
                share_percent = solution['shares'][j] * 100
                amount_required = solution['shares'][j] * solution['mixture_volume']
                report_text += f"  {current_products[j]['name']}: {share_percent:.1f}% ({amount_required:.2f} т)\n"
        
        if use_urea and urea_percentage > 0:
            urea_amount = (urea_percentage / 100) * solution['mixture_volume']
            report_text += f"  Карбамид: {urea_percentage:.1f}% ({urea_amount:.2f} т)\n"
        
        report_text += f"Протеин: {solution['total_protein']:.2f}%\n\n"
    
    report_text += "ФИНАЛЬНЫЕ ОСТАТКИ:\n"
    for item in final_solution['remaining_stock']:
        if item['remaining'] > 0.001:
            report_text += f"  {item['name']}: {item['remaining']:.2f} т\n"
    
    # Download button for report
    st.download_button(
        label="Скачать полный отчет (TXT)",
        data=report_text,
        file_name=f"profit_optimization_report_{timestamp}.txt",
        mime="text/plain",
        type="primary"
    )
