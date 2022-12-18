import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


st.title('Analise de Desepenho de Modelo IA')
st.write('Para analisar paramentros de Acuracia, Precissão e as demais metricas, precisamos analisar 4 ocorrencias')

st.subheader('Matriz de Confusão')
st.markdown(''' 
Para entender melhor cada métrica, primeiro é necessário entender alguns conceitos.
Uma matriz de confusão é uma tabela que indica os erros e acertos do seu modelo, comparando com o resultado esperado (ou etiquetas/labels). A imagem abaixo demonstra um exemplo de uma matriz de confusão.
''')
st.image('https://miro.medium.com/max/720/1*s7VB26Cfo1LdVZcLou-e0g.webp', width=500)

st.markdown('''
* Verdadeiros Positivos: classificação correta da classe Positivo;
* Falsos Negativos (Erro Tipo II): erro em que o modelo previu a classe Negativo quando o valor real era classe Positivo;
* Falsos Positivos (Erro Tipo I): erro em que o modelo previu a classe Positivo quando o valor real era classe Negativo;
* Verdadeiros Negativos: classificação correta da classe Negativo.
''')

st.subheader('Métricas de Avaliação')
st.markdown('''Ao ser feita a contagem de todos esses termos e obter a matriz de confusão, é possível calcular métricas de avaliação para a classificação. ''')
st.image('https://miro.medium.com/max/720/1*t1vf-ofJrJqtmam0KSn3EQ.webp', width=500)

av_desempenho = pd.read_excel('dados de avaliação.xlsx', sheet_name=1)
av_desempenho = av_desempenho.loc[0:2]

st.subheader('Tabela de Paramentros')
st.markdown(''' Abaixo temos a tabela de métricas com os dados para analisar o desempenho da IA. ''')
st.markdown('''Foram testado 3 dataset como entrada para a IA. ''')
st.markdown(''' 
 * Teste: Dataset original de imagens de ressonância magnética
 * Teste_Ruido: Dataset de imagens de ressonância magnética com adição de ruidos RGB
 * Teste_Inver: Dataset de imagens de ressonância magnética com fitro de inversão (Imagem negativa)
''')

st.dataframe(av_desempenho)

av_desempenho['ACURÁCIA'] = (av_desempenho['VP'] + av_desempenho['VN']) / (av_desempenho['VP'] + av_desempenho['VN'] + av_desempenho['FP'] + av_desempenho['FN'])
av_desempenho['PRECISÃO'] = (av_desempenho['VP']) / (av_desempenho['VP'] + av_desempenho['FP'])
av_desempenho['RECALL'] = (av_desempenho['VP']) / (av_desempenho['VP'] + av_desempenho['FN'])
av_desempenho['F1 SCORE'] = (2*(av_desempenho['PRECISÃO'])*av_desempenho['RECALL']) / ((av_desempenho['PRECISÃO']) + av_desempenho['RECALL'])

x = av_desempenho[['DataSet','ACURÁCIA']].set_index('DataSet')
#Mostrando Gráfico de Área de Destino
dataChart = x.plot()
st.subheader('Grafico de Barra - ACURÁCIA')
st.markdown(
    '''
    * Acurácia: indica uma performance geral do modelo. Dentre todas as classificações, quantas o modelo classificou corretamente;
    '''
)
st.bar_chart(x)

x = av_desempenho[['DataSet','PRECISÃO']].set_index('DataSet')
#Mostrando Gráfico de Área de Destino
dataChart = x.plot()

st.subheader('Grafico de Barra - PRECISÃO')
st.markdown(
    ''' 
    * Precisão: dentre todas as classificações de classe Positivo que o modelo fez, quantas estão corretas;
    ''')
st.bar_chart(x)

x = av_desempenho[['DataSet','RECALL']].set_index('DataSet')
#Mostrando Gráfico de Área de Destino
dataChart = x.plot()
st.subheader('Grafico de Barra - RECALL')
st.markdown(
    '''
    * Recall/Revocação/Sensibilidade: dentre todas as situações de classe Positivo como valor esperado, quantas estão corretas;   '''
)
st.bar_chart(x)

x = av_desempenho[['DataSet','F1 SCORE']].set_index('DataSet')
#Mostrando Gráfico de Área de Destino
dataChart = x.plot()
st.subheader('Grafico de Barra - F1 SCORE')
st.markdown(
    '''
    * F1-Score: média harmônica entre precisão e recall.'''
)
st.bar_chart(x)

st.subheader('Melhor modelo')
st.markdown(
    '''
    O modelo teste foi o melhor avaliado nas 4 metricas de Avaliação'''
)
dat_av = av_desempenho[['DataSet','PRECISÃO','ACURÁCIA','RECALL','F1 SCORE']].set_index('DataSet')
dataChart = dat_av.plot()
st.area_chart(dat_av)
