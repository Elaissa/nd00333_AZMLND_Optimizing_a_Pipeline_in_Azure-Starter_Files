# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
**In 1-2 sentences, explain the problem statement: e.g** Pracujemy nad zbiorem danych UCI Bank Marketing z http://archive.ics.uci.edu/ml/datasets/Bank+Marketing celem jest przewidywanie jak kient banku korzysta z zareklamowanych usług banku.   "This dataset contains data. about... we seek to predict..."

**In 1-2 sentences, explain the solution: e.g.**
Najlepszy model jest oparty o pipeline automl gdzie użyto algorytmu Voting Assamble, który ma accuracy Drugi algorytm to Logistic Regression z accuracy.... "The best performing model was a ..."

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
Użyłam  algorytmu klasyfikacyjnego biblioteki Scikit-learn. Wykonałam następujące działania w pliku train.py
Użyłam klasy tabulardatasetfactory do załadowania zestawu danych UCI Bank Marketing. Użyłam funckcji cean_data w pliku train.py do czyszczenia, porządowania danych oraz ustaweinia etykiet danych (Label). 
Użyłam metody train_test_split (aby podzielić wyczyszczone dane na próbę treningową i próbe testową). Następnie użyłam dwóch parametrów do algorytmu klasyfikacyjnego Logistic Regressionw w celu znalezenia najlepszych wartości dla poniższych parametrów: parametr C oznacza (inverse of regularization strength) okreslając tym mniejsze wartości zwiększą siłę regularyzacji.
Wybrałam ciągły zakres (uniform range) pomiędzy 0.5 i 0.9. Drugi parametr to max_iter czyli maksymalna liczba iteracji potrzebna do znalezienia optymalnego rozwiązania.
Wybrałam trzy dyskretne wartości do przeszukania. Jako element zakonczenia obliczeń wybrałam BanditPolicy, który bazuje na czynnikach (factory) ilości obliczeń. 
Określiłam czynnik obliczeń na 0.1 i ustawiłam parametr evaluation_interval na wartość 1 oraz parametr delay_evaluation na wartość 5.


**What are the benefits of the parameter sampler you chose?**
Użyłam prókowania losowego (random sampler) do optymalizacji (hypertuning) próbka pozwala nam na wybór parametrów takich jak parametrów reguraryzacji i iterracji. 
Wartości regularyzacji były ciągłe (C) i dyskretne (max_iter).

**What are the benefits of the early stopping policy you chose?**
BanditPolicy jest polityką używaną do zatrzymania optymalizacji powyższych hyperparametrów. Pozwala to zatrzymać proces kiedy występuje spadek wartości accuracy.
W tym eksperymencie polityka przerwania tunningu jest zastosowana dla każdej wartości metryki, która jest raportowana gdy wychodzi spadek na poziomie 5,a to oznacza kiedy najlepsza metryka jest mniejsza niż 91% od najlepszego wyniku wcześniej raportowanego.

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**
 Dla procedury AutoML kroki pierwsze są podobne jak w popszednim pipeline: czyszczenie danych, dzielenie danych, zestaw treningowy i testowy. Następnym waznym krokiem jest ustawienie konfiguracji dla AutoML gdzie accuracy jest podstawą metryką (primary metric).Użyłam krzyżowej walidacji (cross valitation) aby uniknąć przeuczenia (everlifting).

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
W pierwszym przypadku użycie hyperdrive dotyczy jednego modelu z różnymi parametrami ale AutoML działa na różnych modelach. Hyperdrive trzeba zbudować plik treningowy train.py, ale w AutoML potrzebujemy tylko przekazania danych i zdefiniowana zadania. AutoML jest łatwiejsze w użyciu praca z nim jest szybsza. Praca z hypierdrive jest dłuższa ze względu  na napisanie pliky trian.py.
Najlepszym modelem w hyperdrive jest aalgorytm klasyfikacyjny Logistic Regressor z następującymi parametrami C= 0.57, max_iter= 150 i accuracy= 91.1%.
AutoML przy algorytmie VotingAssambe z accuracy na poziomie 91.9%.


## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**
Mamy bardzo niezrównoważone dane (inbalancy).Potrzebuje więcej czasu na inżynierię cech (feature engineering).

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**

![zamykanie computing clustering](https://raw.githubusercontent.com/Elaissa/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/master/Zrzut%20ekranu%20(94).png)
