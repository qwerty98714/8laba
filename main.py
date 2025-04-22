import xml.etree.ElementTree as ET
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def read_xml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"Файл {file_path} не найден.")
        exit()
    except ET.ParseError:
        print(f"Ошибка в структуре XML-файла {file_path}.")
        exit()

    items = []
    for offer in root.findall(".//offer"):
        name = offer.find("name").text
        price = float(offer.find("price").text)
        mission = offer.find("./param[@name='Миссия']").text
        sales = []
        for month in offer.findall(".//month"):
            sales.append(int(month.text))
        items.append({"name": name, "price": price, "mission": mission, "sales": sales})

    return items

def predict_sales_13th_month(sales_data):
    predictions = {}
    for name, data in sales_data.items():
        last_three_months = data["sales"][-3:]
        predicted_sales = round(sum(last_three_months) / len(last_three_months))
        predictions[name] = predicted_sales
    return predictions
xml_file_path = "rovers_sales_data.xml"
items = read_xml(xml_file_path)

months = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]

predictions = predict_sales_13th_month({item["name"]: item for item in items})

json_data = {
    "rovers": [
        {
            "name": item["name"],
            "price": item["price"],
            "mission": item["mission"],
            "sales": item["sales"],
            "predicted_sales": predictions[item["name"]]
        }
        for item in items
    ]
}

json_file_path = "rovers_sales_data.json"
with open(json_file_path, "w", encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)

df = pd.DataFrame({
    "Марсоход": [item["name"] for item in items for _ in range(12)],
    "Месяц": months * len(items),
    "Продажи": [item["sales"][i] for item in items for i in range(12)]
})

sales_table = df.pivot(index="Месяц", columns="Марсоход", values="Продажи")
sales_table.loc["Прогноз"] = [predictions[rover] for rover in sales_table.columns]

print(sales_table.to_string())

fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.3
x = range(len(months))

for i, item in enumerate(items):
    ax.bar([p + bar_width * i for p in x], item["sales"], width=bar_width, label=f"{item['name']} (история)")
    ax.bar(12 + bar_width * i, predictions[item['name']], width=bar_width, label=f"{item['name']} (прогноз)", color='red')

ax.set_title("Динамика продаж марсоходов с прогнозом на 13-й месяц")
ax.set_xlabel("Месяц")
ax.set_ylabel("Количество продаж")
ax.set_xticks([p + bar_width for p in range(13)])
ax.set_xticklabels(months + ["Прогноз"])
ax.legend()
plt.tight_layout()
plt.show()
