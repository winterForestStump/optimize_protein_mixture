import streamlit as st

def wide_space_default():
    st.set_page_config(layout='wide')

wide_space_default()

pg = st.navigation([st.Page("page_1.py", title = "Расчет Проекта", url_path='project'), 
                    st.Page("page_2.py", title = "Расчет Остатков", url_path='residuals')])
pg.run()