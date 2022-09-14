import pandas as pd
import yfinance as yf
import streamlit as st 
import numpy as np
import plotly #graph_objects as go
import plotly.express as px
import sklearn
from sklearn.ensemble import RandomForestClassifier

tips = pd.read_csv('tips.csv', index_col=[0])

st.write("""
# Анализ данных по чаевым
""")

with st.expander('Самые частые случаи'):
  #st.write("""На гистограммах""")
  st.write("""Гистограммы позволяют нам ответить на такие вопросы: 
  \n В какой день чаще заходят клиенты?
  \n Насколько часто это курящие люди? 
  \n Чаще ли в заведение заходят мужчины, чем женщины?
  \n Сколько чаевых оставляют чаще всего?
  \n Больше ли людей заглядывает на ланч, чем на обед? """)
  #st.image(image='hist.png', use_column_width=True)
  with st.form(key='hist'):
    option = st.selectbox('Выберите показатель', tips.columns) #modify_display())
    if st.form_submit_button('Построить'):
      hist = px.histogram(tips, x=option)
      st.plotly_chart(hist, use_container_width=True)
      st.info("""Как видно из графиков:
      \n Заведение чаще всего посещают в субботу.
      \n Большинство посетителей являются некурящими, значит, нужно позаботиться о том, чтобы в ресторане у них был отдельный от курящих людей зал.
      \n Абсолютное большинство клиентов берёт 2 блюда (скорее всего люди заказывают 1 блюдо + 1 напиток), следовательно, нужно замотивировать их брать больше блюд с помощью специальных акций, предложений.
      \n Основная аудитория заведения - это мужчины.
      \n На обед заходит больше клиентов, чем на ланч, стоит подумать, как привлечь больше клиентов в вечернее время.
      \n Чаще всего на чай оставляют порядка 2 \$, а суммарный чек чаще всего составляет 12-18 \$.
      """)

with st.expander('Как связаны суммарный чек, чаевые и количество блюд в заказе?'):
  st.write("""Давайте посмотрим, существует ли связь между данными параметрами:""")
  #st.image(image='../images/scatter.png')
  with st.form(key='scatter'):
    ax_x = st.selectbox('Выберите, что будет по горизонтали', tips.drop(columns=['sex','smoker','day','time']).columns.to_list())
    ax_y = st.selectbox('Выберите, что будет по вертикали', tips.drop(columns=['sex','smoker','day','time']).columns.to_list())
    scat = px.scatter(tips, x=ax_x, y=ax_y) # , trendline="ols")
    if st.form_submit_button('Построить'):
      st.plotly_chart(scat, use_container_width=True)
      st.write("""Можно сделать вывод, что в среднем люди с большей суммой заказа и большим количеством блюд щедрее оставляют чаевые.""")


with st.expander('Процентные показатели'):
  st.write("""Давайте посмотрим процентное соотношение между посетителями (мужчинами и женщинами, курящими и некурящими), а также распределение по дням и времени посещения.""")
  #st.image(image='../images/pie_chart.png')
  with st.form(key='pie'):
    col_option = st.selectbox('Выберите показатель', tips.drop(columns=['total_bill', 'tip']).columns)
    pie_df = tips[col_option].value_counts()
    pie_fig = plotly.graph_objs.Figure(data=[plotly.graph_objs.Pie(labels=pie_df.index, values=pie_df.tolist(), textinfo='label+percent',
                    insidetextorientation='radial')])
    pie_fig.update_traces(textposition='inside', textinfo='percent+label')
    if st.form_submit_button('Построить'):
      st.plotly_chart(pie_fig, use_container_width=True)
  st.write("""Как видно из графика, доля курящих людей в зведении составляет почти 40%, а значит, что возможно стоит сделать для них отдельный зал или, например, добавить в меню кальян, чтобы получить большую прибыль. \n
  Среди клиентов преобладают мужчины, значит, стоит организовать в заведении трансляции спортивных матчей, либо подумать, как привлечь женскую аудиторию и выровнять состав посетителей.
  \n Видно, что посещаемость в пятницу сильно ниже, чем в прочие дни, хотя пятница, традиционно, должна привлекать больше посетителей под конец рабочей недели. Возможно стоит сделать в этот день какую-нибудь акцию, вроде 'счастливых часов' или 'приведи друга'. """)

with st.expander('Существует ли связь между количеством и качеством?'):
  st.write("""Кто щедрее расплачивается: мужчины или женщины, курящие или нет? Как связаны день и чаевые? Размер заказа и пол? Давайте посмотрим:""")
  
  with st.form(key='box_plot'):

    ax_x = st.selectbox('Выберите количественный показатель', tips.drop(columns=['total_bill', 'tip', 'size']).columns.tolist())
    ax_y = st.selectbox('Выберите качественный показатель', tips.drop(columns=['sex', 'smoker','day','time']).columns.tolist())
    button_box = st.form_submit_button('Построить')
    if button_box:
      box = px.box(tips, x=ax_x, y=ax_y)
      st.plotly_chart(box, use_container_width=True)

  st.write("""Мужчины, в среднем, оставляют больше чаевых и тратят на заказ, а вот по количеству блюд в заказе равенство.
  \n Курящие посетители оставляют больше чаевых и чаще делают большие заказы. 
  \n Средний чек больше всего в воскресенье, чаевых в этот день также оставляют больше. Кроме того, видно, что в выходные чаще встречаются большие заказы (более 2 блюд). 
  \n Что касается времени дня, то в обед клиенты платят больший средний чек и чаевые, чем на ужин. """)