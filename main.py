import xml.etree.ElementTree as ET
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

xml_path = "rovers.xml"
tree = ET.parse(xml_path)
root = tree.getroot()

rovers = []
for offer in root.findall(".//offer"):
    name = offer.find("name").text
    price = float(offer.find("price").text)
    rovers.append({"name": name, "price": price})

with open("rovers_sales_data.json", encoding="utf-8") as f:
    sales_json = json.load(f)

for item in sales_json["sales"]:
    for r in rovers:
        if r["name"] == item["name"]:
            r["sales"] = item["sales"]

X_train = []
y_train = []
mean_values = []

for rover in rovers:
    sales = rover["sales"]
    mean_sales = np.mean(sales)
    mean_values.append(round(mean_sales, 2))

    for i in range(3, len(sales)):
        X_train.append([sales[i-3], sales[i-2], sales[i-1]])
        y_train.append(1 if sales[i] > mean_sales else 0)  

X_train = np.array(X_train)
y_train = np.array(y_train)


sgd_model = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd_model.fit(X_train, y_train)

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)


X_test = []
for rover in rovers:
    sales = rover["sales"]
    X_test.append([sales[-3], sales[-2], sales[-1]])

X_test = np.array(X_test)

sgd_preds = sgd_model.predict(X_test)
lr_preds = lr_model.predict(X_test)


result_df = pd.DataFrame({
    "Марсоход": [r["name"] for r in rovers],
    "Среднее значение": mean_values,
    "SGD Прогноз (13 мес.)": ["высокие" if p else "низкие" for p in sgd_preds],
    "LogReg Прогноз (13 мес.)": ["высокие" if p else "низкие" for p in lr_preds]
})

print("=== Результаты прогнозов (8 лабораторная, классификация) ===")
print(result_df.to_string(index=False))

