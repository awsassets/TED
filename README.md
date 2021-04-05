Hola a todos y bienvenidos al segundo post de mi blog, hace aproximadamente unos 3 meses estuve desarrollando una aplicación que me permitia sacar todos los tweets, repositorios y enlaces que contuviesen un determinado CVE mostrando con gráficas; recuentos de tweets, retweets y likes, con esos datos y gracias a los algoritmos de Machine Learning podía identificar que tweets contenian una posible prueba de concepto/exploit, dejo el enlace de YouTube: https://www.youtube.com/watch?v=bsBroYeO0e8

Debido a que me está siendo muy dificil sacar tiempo para poder continuar con el desarrollo de la aplicación, en esta semana santa (2 días) se me ocurrió que podía simplificarlo mucho más y tener algo provisional para que me alertase de cuando se publicasen pruebas de concepto de exploits en Twitter, a continuación os detallo mi aventura para poder conseguirlo.

Si tenéis twitter y estáis en el mundo de la ciberseguridad, sabréis que es muy importante, ya que te mantiene actualizado (si seguís a la gente correcta) en todo momento de nuevas vulnerabilidades o nuevas amenazas que surgen en el mundo, yo por ejemplo para algunas personas que sé que siempre publican tweets interesantes tengo la campanita activada y de esta manera me informo de lo nuevo. Siguiendo con el tema principal mi idea fue la siguiente; crear un bot que hiciese retweets a los tweets que tuviesen una prueba de concepto o exploit para de esta manera activando la campana en ese perfil me alertase de cuando se publicase algo, asi todo el mundo podría seguir al bot y mantenerse informado también.

El bot ataca a varios endpoints de la API de twitter, uno de ellos por ejemplo es donde se estrimean los tweets y se realiza un filtro que veremos más adelante, para empezar veremos como se realizan las peticiones, y es que para hacer dichas consultas es necesario tener una cuenta de desarrollador de Twitter y para conseguirla es necesario enviar un email mencionando los intereses en utilizar la API.

<p align="center"><img src="https://raw.githubusercontent.com/alexfrancow/TED/main/images/1.PNG" height="500" width="825" /></p>
<p align="center"><img src="https://raw.githubusercontent.com/alexfrancow/TED/main/images/2.PNG" height="500" width="825" /></p>
<p align="center"><img src="https://raw.githubusercontent.com/alexfrancow/TED/main/images/3.PNG" height="500" width="825" /></p>

Una vez aprobada por el equipo de Twitter, en el panel de desarrollador: https://developer.twitter.com/en/portal/dashboard, saldrá el proyecto que nos han creado, en este caso para propósitos academicos se habilitará el acceso a la V1.1 y a la V2 de la API.

<p align="center"><img src="https://raw.githubusercontent.com/alexfrancow/TED/main/images/4.PNG" height="500" width="825" /></p>

> Desde el mismo panel es donde se generan los tokens (API Key & Secret y el Bearer Token) que utilizaremos para nuestro código en Python.

Teniendo ya acceso a la API de Twitter solo quedaría entrenar los algoritmos correspondientes para detectar que Tweet se trata de una PoC/Exploit y cual no, para ello se descargarán todos los tweets que contengan "CVE-" de los años 2019-2020 y se pasaran a .csv, existe una app llamada twint desarrollada en python que hará por nosotros este trabajo:

```bash
# Get old tweets
https://github.com/twintproject/twint

conda install -c anaconda git 
pip3 install twint
pip3 install --user --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint

# Collect Tweets that were tweeted before 2020.
twint -s "cve-" --year 2020 -o cves-2019.csv --csv

# Collect Tweets that were tweeted since 2015-12-20 00:00:00.
twint -s "CVE-" --since 2015-12-20
```

Una vez descargados, viene la parte tediosa la cual es identificar que tweet es una PoC/Exploit y cual no, un proceso manual que se basa en abrir el .csv e ir linea por linea marcando 1 o 0 en función si vemos que se trata de un exploit o no, esto es necesario para el algoritmo de clasificación que vamos a utilizar ya que al ser supervisado tenemos que clasificar nosotros el dataset poniendo etiquetas (labels), en este caso la etiqueta 'exploit' que será igual a 1 o 0.

<p align="center"><img src="https://raw.githubusercontent.com/alexfrancow/TED/main/images/5.PNG" height="500" width="825" /></p>

Para esta aplicación se clasificaron muy pocos tweets, en total fueron 629, aplicación entrenanda siguiendo el modelo MultinomialNB, un tipo de algoritmo del conjunto Naive Bayes, una clase especial de algoritmos de clasificación de Machine Learning que se basan en una técnica de clasificación estadística llamada “teorema de Bayes”, como mencioné anteriormente supervisado.

Estos modelos son llamados algoritmos “Naive” y en ellos se asume que las variables predictoras son independientes entre sí. En otras palabras, que la presencia de una cierta característica en un conjunto de datos no está en absoluto relacionada con la presencia de cualquier otra característica.

Un claro ejemplo sería:

Consideremos el caso de dos compañeros que trabajan en la misma oficina: Alicia y Bruno. Sabemos que:

    Alicia viene a la oficina 3 días a la semana.
    Bruno viene a la oficina 1 día a la semana.

Esta sería nuestra información “anterior”.

Estamos en la oficina y vemos pasar delante de nosotros a alguien muy rápido, tan rápido que no sabemos si es Alicia o Bruno.

Dada la información que tenemos hasta ahora y asumiendo que solo trabajan 4 días a la semana, las probabilidades de que la persona vista sea Alicia o Bruno, son:

    P(Alicia) = 3/4 = 0.75
    P(Bruno) = 1/4 = 0.25

Cuando vimos a la persona pasar, vimos que él o ella llevaba una chaqueta roja. También sabemos lo siguiente:

    Alicia viste de rojo 2 veces a la semana.
    Bruno viste de rojo 3 veces a la semana.

Así que, para cada semana de trabajo, que tiene cinco días, podemos inferir lo siguiente:

    La probabilidad de que Alicia vista de rojo es → P(Rojo|Alicia) = 2/5 = 0.4
    La probabilidad de que Bruno vista de rojo → P(Rojo|Bruno) = 3/5 = 0.6

Entonces, con esta información, ¿a quién vimos pasar? (en forma de probabilidad), esta nueva probabilidad será la información ‘posterior’.

En este caso usaremos la implementación Naive Bayes “multinomial”. Este clasificador particular es adecuado para la clasificación de características discretas (como en nuestro caso, contador de palabras para la clasificación de texto), y toma como entrada el contador completo de palabras.


El primer paso para comenzar a utilizar el algoritmo descrito anteriormente será importar nuestro dataset, como se ha mencionado el tedioso dataset en el que hemos etiquetado cada tweet.

```python
import pandas as pd
pd.options.display.float_format = '{:.0f}'.format
columns = "id,date,username,tweet,exploit"

df = pd.read_csv('DATASET.csv', usecols=columns.split(","))  
```

Una vez cargado seleccionaremos X e y y haremos un split de nuestro dataset para posteriormente calcular el Acc:

```python
# Train/test
X = df['stems']
y = df['exploit']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 13)
```

Y con el siguiente bloque de codigo pondremos a entrenar a nuestro modelo de ML, 
importaremos el clasificador “MultinomialNB” y ajustaremos los datos de entrenamiento en el clasificador usando fit().

```python
# Train bayes classifier MultinomialNB

pipe_mnnb = Pipeline(steps = [('tf', TfidfVectorizer()), ('mnnb', MultinomialNB())])

# Parameter grid
pgrid_mnnb = {
    'tf__max_features' : [1000, 2000, 3000],
    'tf__stop_words' : ['english', stop_words_alexfrancow],
    'tf__ngram_range' : [(1,1),(1,2)],
    'tf__use_idf' : [True, False],
    'mnnb__alpha' : [0.1, 0.5, 1]
}

gs_mnnb = GridSearchCV(pipe_mnnb, pgrid_mnnb, cv=5, n_jobs=-1)
gs_mnnb.fit(X_train, y_train)
```

Se aprecia que el entrenamiento con 629 tweets da un Accuracy de 0.77:

```python
# Confusion matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, accuracy_score

y_pred_class = gs_mnnb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_class))
plot_confusion_matrix(gs_mnnb, X_test, y_test)
plt.title('Confusion matrix of the classifier')
plt.show()
```

Una vez entrenado nuestro modelo simplemente quedaría por consumir de la API de Twitter y sacar los tweets en tiempo real, para ello haremos uso del siguiente endpoint: https://api.twitter.com/2/tweets/search/stream, al no interesarnos todos los tweets antes deberemos aplicar una regla para filtrar los tweets que contengan el string "CVE-", para ello usaremos el endpoint: https://api.twitter.com/2/tweets/search/stream/rules el cual  permitirá crear una regla que omita los retweets y los replies con el siguiente bloque de código:

```python
sample_rules = [
        {'value': '"CVE-" -is:retweet -is:reply',
         'tag': 'Exploits'},
    ]
payload = {"add": sample_rules}
response = requests.post(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        headers=headers,
        json=payload,
    )
```

Para la documentación sobre el stream y las reglas podemos consultar los siguientes enlaces:

- Docu: https://developer.twitter.com/en/docs/twitter-api/tweets/filtered-stream/quick-start
- Rules: https://developer.twitter.com/en/docs/twitter-api/tweets/filtered-stream/integrate/build-a-rule


<p align="center"><img src="https://raw.githubusercontent.com/alexfrancow/TED/main/images/6.PNG" height="500" width="825" /></p>


Llegados hasta aqui solo nos queda una pregunta por responder, ¿es funcional?

Pues si, el bot es funcional, veamos un caso que ha considerado como exploit:

<p align="center"><img src="https://raw.githubusercontent.com/alexfrancow/TED/main/images/7.PNG" height="500" width="825" /></p>

Efectivamente es un exploit para la vulnerabilidad CVE-2020-25078 que afecta a las camaras D_Link-DCS-2530L, una vulnerabilidad vieja pero ejemplifica bien el funcionamiento del bot, si nos vamos a shodan.io y buscamos por ese modelo de camara nos aparecerán 4275 resultados, si entramos en una IP del listado y añadimos a la url "/config/getuser?index=0" veremos que en muchos casos es posible explotarlos, consiguiendo de esta manera las credenciales de la cuenta administrador que nos permiten entrar en la camara y ver en tiempo real el video capturado.

<p align="center"><img src="https://raw.githubusercontent.com/alexfrancow/TED/main/images/8.PNG" height="500" width="825" /></p>
<p align="center"><img src="https://raw.githubusercontent.com/alexfrancow/TED/main/images/9.PNG" height="500" width="825" /></p>
<p align="center"><img src="https://raw.githubusercontent.com/alexfrancow/TED/main/images/10.PNG" height="500" width="825" /></p>

El bot categoriza algunos tweets que no son exploits, esto es debido a que solo ofrece un 0.77 de Acc, cosa que se podrá mejorar etiquetando más tweets del dataset y entrenando de nuevo el modelo.
