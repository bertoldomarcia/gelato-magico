import argparse
import pandas as pd
import joblib
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Prever vendas de sorvete com base na temperatura")
    parser.add_argument('--temp', type=float, help='Temperatura do dia')

    args = parser.parse_args()

    if args.temp is None:
        print("\n⚠️  Por favor, forneça a temperatura com o argumento --temp. Exemplo: python predict.py --temp 30.0\n")
        return

    model_path = "model/ice_cream_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Modelo não encontrado. Certifique-se de treinar o modelo com train.py primeiro.")

    model = joblib.load(model_path)
    input_data = pd.DataFrame({"temperatura": [args.temp]})
    prediction = model.predict(input_data)

    print(f"\n📈 Previsão de vendas para {args.temp}°C: {int(prediction[0])} sorvetes\n")

if __name__ == "__main__":
    main()
