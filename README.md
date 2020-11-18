# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
**In 1-2 sentences, explain the problem statement: e.g** Pracujemy nad zbiorem danych UCI Bank Marketing z http://archive.ics.uci.edu/ml/datasets/Bank+Marketinghttp://archive.ics.uci.edu/ml/datasets/Bank+Marketing celem jest przewidywanie jak kient banku korzysta z zareklamowanych usług banku.   "This dataset contains data. about... we seek to predict..."
We are working on the UCI Bank Marketing dataset. The dataset gives information about a marketing campaign of a bank. The aim is to predict how likely the customer are to subscribe to the product being advertised.

**In 1-2 sentences, explain the solution: e.g.**
Najlepszy model jest oparty o pipeline AutoMl gdzie użyto algorytmu Voting Assamble, który ma accuracy Drugi algorytm to Logistic Regression z accuracy.... "The best performing model was a ..."

The best performing model was based on the AutoMl pipeline where the Voting Assamble algorithm has been used, which has accuracy.... The second algorithm is Logistic Regression with accuracy....

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
Użyłam  algorytmu klasyfikacyjnego biblioteki Scikit-learn. Wykonałam następujące działania w pliku train.py
Użyłam klasy TabularDatasetFactory do załadowania zestawu danych UCI Bank Marketing. Użyłam funckcji cean_data w pliku train.py do czyszczenia, porządowania danych oraz ustaweinia etykiet danych (Label). 
Użyłam metody train_test_split (aby podzielić wyczyszczone dane na próbę treningową i próbe testową). Następnie użyłam dwóch parametrów do algorytmu klasyfikacyjnego Logistic Regression w w celu znalezenia najlepszych wartości dla poniższych parametrów: parametr C oznacza (inverse of regularization strength) okreslając tym mniejsze wartości zwiększą siłę regularyzacji.
Wybrałam ciągły zakres (uniform range) pomiędzy 0.5 i 0.9. Drugi parametr to max_iter czyli maksymalna liczba iteracji potrzebna do znalezienia optymalnego rozwiązania.
Wybrałam trzy dyskretne wartości do przeszukania. Jako element zakonczenia obliczeń wybrałam BanditPolicy, który bazuje na czynnikach (factory) ilości obliczeń. 
Określiłam czynnik obliczeń na 0.1 i ustawiłam parametr evaluation_interval na wartość 1 oraz parametr delay_evaluation na wartość 5.

I used the Scikit-learn library classification algorithm. I did the following in the train.py file.
I used the TabularDatasetFactory class to load the UCI Bank Marketing dataset. I used the cean_data function in the train.py file to clean up, order data and set up data labels (Label).
I used the train_test_split method (to divide the cleared data into a training trial and a test trial). Then I used two parameters for the Logistic Regression classification algorithm in order to find the best values for the following parameters: parameter C means (inverse of regularization strength), specifying smaller values will increase the strength of regularization.
I chose a uniform range between 0.5 and 0.9. The second parameter is max_iter, which is the maximum number of iterations needed to find the optimal solution.
I chose three discrete values to search. As an element of finishing the calculations I chose BanditPolicy, which is based on the factors (factory) of the number of calculations.
I set the calculation factor to 0.1 and set the evaluation_interval parameter to 1 and the delay_evaluation parameter to 5.



**What are the benefits of the parameter sampler you chose?**
Użyłam prókowania losowego (random sampler) do optymalizacji (hypertuning) próbka pozwala nam na wybór parametrów takich jak parametrów reguraryzacji i iterracji. 
Wartości regularyzacji były ciągłe (C) i dyskretne (max_iter).

I used Random sampler for the hypertuning. The sampler allows one to chose parameter values from a set of discrete values or a distribution over a continous range. Since the two hyperparameters that I have selected are both continous (C) and discrete (max_iter) this is a natural choice.

**What are the benefits of the early stopping policy you chose?**
BanditPolicy jest polityką używaną do zatrzymania optymalizacji powyższych hyperparametrów. Pozwala to zatrzymać proces kiedy występuje spadek wartości accuracy.
W tym eksperymencie polityka przerwania tunningu jest zastosowana dla każdej wartości metryki, która jest raportowana gdy wychodzi spadek na poziomie 5,a to oznacza kiedy najlepsza metryka jest mniejsza niż 91% od najlepszego wyniku wcześniej raportowanego.

BanditPolicy is used to stop the hyperparemeter tuning, it helps to stop the process when there is a considerable drop in Accuracy(metric being checked against). In the experiment, early termination policy is applied at every interval when metrics are reported, starting at evaluation interval 5. Any run whose best metric is less than (1/(1+0.2) or 91% of the best performing run is terminated.

**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**
 Dla procedury AutoML kroki pierwsze są podobne jak w popszednim pipeline: czyszczenie danych, dzielenie danych, zestaw treningowy i testowy. Następnym waznym krokiem jest ustawienie konfiguracji dla AutoML gdzie accuracy jest podstawą metryką (primary metric).Użyłam krzyżowej walidacji (cross valitation) aby uniknąć przeuczenia (everlifting).
 
 For AutoML procedure the first intial steps were same in data pipeline: Clear data Split data to train test dataset. Next we created a config for AutoML, with accuracy as our primary metric. We also decided for cross validation to take care of any overfitting.

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
W pierwszym przypadku użycie hyperdrive dotyczy jednego modelu z różnymi parametrami ale AutoML działa na różnych modelach. Hyperdrive trzeba zbudować plik treningowy train.py, ale w AutoML potrzebujemy tylko przekazania danych i zdefiniowana zadania. AutoML jest łatwiejsze w użyciu praca z nim jest szybsza. Praca z hypierdrive jest dłuższa ze względu  na napisanie pliky trian.py.
Najlepszym modelem w hyperdrive jest aalgorytm klasyfikacyjny Logistic Regressor z następującymi parametrami C= 0.57, max_iter= 150 i accuracy= 91.1%.
AutoML przy algorytmie VotingAssambe z accuracy na poziomie 91.9%.

First in case of Hyperdrive only one model with different hyperparameters was searched, but AutoML worked with many other models as well. In Hyperdrive we have to build a training script, but in AutoML we just need to pass the data, and define the task. AutoML is easy to use, I got my AutoML working in first run, but for Hyperdrive, two days were spent. The best model using hyperdrive is LOgistic Regressor Classifier with C= 0.57, max_iter = 150, and accuracy = 91.4%. The AutoML gives us the best result for 91.9%.


## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**
Mamy bardzo niezrównoważone dane (inbalancy).Potrzebuje więcej czasu na inżynierię cech (feature engineering).

Presently, I have not spent much time on data cleansing and feature engineering. I think the two can be explored to generate better results. Also as mentioned by Hyperdrive, the data is highly imbalanced, so strategies to deal with class imbalance should be done as well.

Proof of cluster clean up If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section. Image of cluster marked for deletion.

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**

![zamykanie computing clustering](https://raw.githubusercontent.com/Elaissa/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/master/Zrzut%20ekranu%20(94).png)
