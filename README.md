# 🍦 Gelato Mágico - Previsão de Vendas com ML

Este projeto tem como objetivo prever a quantidade de sorvetes vendidos com base na temperatura do dia, ajudando sorveterias a reduzir desperdícios e maximizar seus lucros.

## 📌 Objetivos

- Treinar um modelo de regressão preditiva.
- Usar o MLflow para rastreamento de experimentos.
- Implantar o modelo para previsões em tempo real.
- Criar um pipeline de ML estruturado e reprodutível.

## 📊 Insights e Análises

Após analisar as sentenças com um modelo de IA e correlacionar com dados reais de temperatura e vendas:

- Temperaturas acima de 30°C indicam pico de vendas.
- Abaixo de 20°C, a demanda cai drasticamente.
- Frases mencionando clima extremo (muito quente ou chuvoso) correlacionam bem com dados históricos.

## 🚀 Execução

```bash
# Para instalar as dependências
pip install -r requirements.txt

# Treinar o modelo
python pipeline/train.py

# Fazer previsão
python pipeline/predict.py --temp 32

