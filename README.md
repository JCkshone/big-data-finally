
<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table de Contenido</summary>
  <ol>
    <li>
      <a href="#Jupyter Lab">Jupyter Notebook y Docker</a>
      <ul>
        <li><a href="#jupyter-lab-web">Montar Jupyter Notebook en uns instancia docker</a></li>
      </ul>
    </li>
    <li>
      <a href="#web-scrapping">Analizador se sentimientos Twitter</a>
      <ul>
        <li><a href="#dependency-installation">Instalacion de dependencias</a></li>
        <li><a href="#web-request">Configuracion stopwords y vader lexicon</a></li>
        <li><a href="#data-build">Transformacion de la informacion</a></li>
        <li><a href="#train-build">Contruccion de los modelos para su entrenamiento</a></li>
        <li><a href="#model-prediction">Ejecucion de los modelos y visualizacion de escalas de prediccion</a></li>
        <li><a href="#tweets-organization">Agrupacion de tweets bazado en alasisis de sentimientos</a></li>
      </ul>
    </li>
    <li>
      <a href="#Bonus">Bonus</a>
      <ul>
        <li><a href="#tweepy">Tweepy</a></li>
      </ul>
    </li>
  </ol>
</details>

## Presentacion
- Juan Camilo Navarro Alvira
- Yury Ximena Alvarez
- John Alejandro Avila
<!-- ABOUT THE PROJECT -->
## Introduccion
En este Readme se van a demostrar y a describir cada uno de los pasos a seguir para construir un analizador de sentimientos con un conjunto de datos obtenidos desde el api de twitter.

## Jupyter Notebook y Docker
Nos dirigimos a la documentacion oficial de [Jupyter](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/running.html) alli se describe paso a paso el proceso de montaje, nos indica que debemos dirigirnos al hub de docker y ejecutar el siguiente comando en la terminal
  ```sh
  docker pull jupyter/r-notebook
  ```
 Este comando nos permitira clonar el repo con la imagen y asi, ejecutar la instancia de Jupyter lab.
 Luego nos dirigimos a docker desktop y ejecutamos la instancia

## Analizador se sentimientos Twitter
### Instalacion de dependencias
Para instalar las dependencias es muy simple, solo nos dirigimos a unas de las seccion para compilar codigo en Jupyterlab y se ejecuta la instruccion 
  ```sh
  !conda install -c intel scikit-learn pandas nltk matplotlib langdetect textblob  -y
  ```
![image](https://user-images.githubusercontent.com/19766554/143805646-24a5fb73-bd62-4e81-a8e6-256ffc65b09e.png)

### Configuracion stopwords y vader lexicon
Importamos cada una de las clases necesarias, adicional debemos ejecutar la funcion que permite descargar elementos asyncronos de la libreria nltk 
![image](https://user-images.githubusercontent.com/19766554/143805831-3e2ad24d-5695-4bbf-b6ce-b1cfe3075d28.png)

### Transformacion de la informacion
Leemos la informacion de un archivo .csv y le agregamos el parameto que permitira identificar e la separacion de cada item; se valida la informacion por medio de funciones que nos indica si existen elementos con valores nulos

  ```sh
  stop_words = set(stopwords.words('spanish'))
  tweets_df = pd.read_csv('./medellin_tweets.csv', sep = ',')
  tweets_df['sentiment'].value_counts(dropna = False)
  tweets_df['sentiment'].value_counts(dropna = False, normalize = True)
  tweets_labeled_df = tweets_df.loc[tweets_df['sentiment'].notnull()]
  tweets_labeled_df.shape
  tweets_nolabeled_df = tweets_df.loc[tweets_df['sentiment'].isnull()]
  tweets_nolabeled_df.shape
  ```
### Contruccion de los modelos para su entrenamiento
Creamos propiedades que le permitiran al modelo interpretar y organizar la formacion para posteriormente crear conteos vectorizados 

  ```sh
  X_train, X_test, y_train, y_test = train_test_split(tweets_labeled_df['full_text'], tweets_labeled_df['sentiment'], test_size = 0.2, stratify = tweets_labeled_df['sentiment'], random_state = 1)
  X_train.shape
  pd.Series(y_train).value_counts(normalize = True)
  X_test.shape
  pd.Series(y_test).value_counts(normalize = True)
  ```
  #### Training and evaluating a model using BOW
  ```
  bow = CountVectorizer(tokenizer = tokenizer, stop_words = stop_words)
  tfidf = TfidfVectorizer(tokenizer = tokenizer, stop_words = stop_words)
  X_bow = bow.fit_transform(X_train)
  X_tfidf = tfidf.fit_transform(X_train)
  ```
 La construccion de la informacion y seleccion de campos permite crear un mapa de calor el cual nos dara ua interpretacion mas acertada de la informacion
 
 ![image](https://user-images.githubusercontent.com/19766554/143806307-76f7e937-7280-4619-b535-b193c066729d.png)

### Ejecucion de los modelos y visualizacion de escalas de prediccion

Ahora se realizara una evaluacion de los modelos por medio de una regresion

  ```sh
  logistic_model = LogisticRegression(random_state = 2)
  logistic_model.fit(X_tfidf, y_train)
  y_train_tfidf_predict = logistic_model.predict(X_tfidf)
  y_test_tfidf_predict = logistic_model.predict(bow.transform(X_test))
  ConfusionMatrixDisplay.from_predictions(y_train, y_train_tfidf_predict)
  ConfusionMatrixDisplay.from_predictions(y_test, y_test_tfidf_predict)
  ```
 ![image](https://user-images.githubusercontent.com/19766554/143806653-4f0c9ac1-3e25-45af-a15b-37a733ee544b.png)


### Agrupacion de tweets bazado en alasisis de sentimientos

Por ultimo vamos a crear una agrupacion de tweets usando el data set construido anteriormente, para esto vamos a recorrer cada uno de los tweets vamos a generar una analisis por medio de la libreria TextBlod y vamos a crear un porcentaje de polaridad basado en un analisis de sentimiento, este puntaje lo vamos a analizar  para determinar si es positivo, negativo o neutro, dando un poco mas de a los datos construidos

![image](https://user-images.githubusercontent.com/19766554/143807007-8b5de401-5c19-443e-9332-9cfdbf129db0.png)

Finalmente con estos porcentamos vamos a construir un diagrama pie que nos permitira interpretar los resultados de forma grafica y concisa 

![image](https://user-images.githubusercontent.com/19766554/143807104-534feb0a-ee5c-456e-bf81-b37a1aba54f6.png)

## Bonus

### Tweepy
Tweepy es una herramienta que permite consumir cada uno de los endpoints de twitter, lo unico que se requieren son los accesos, las ventajas de usar esta libreria es que esta optimizada para realizar solicitudes en background y realizar acciones como las que realiza pandas con los data frame, como eliminar elementos duplicados o con poca relevancia 

![image](https://user-images.githubusercontent.com/19766554/143807933-4a85daf5-9abc-4a4d-afb4-298a65a0ced5.png)

![image](https://user-images.githubusercontent.com/19766554/143807964-bae8b557-6343-4fb5-90eb-f6f652325c63.png)

![image](https://user-images.githubusercontent.com/19766554/143807990-2feb4eca-9e6a-4755-92a9-ac420bc5782a.png)



## Referencias 
- [twitter analizer.ipynb](https://github.com/JCkshone/big-data-finally/blob/main/twitter%20analizer.ipynb) con la ejecucion
- [Docker](https://hub.docker.com/r/jupyter/r-notebook)
- [Jupyter](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/running.html)
- [Tweepy](https://www.tweepy.org)
