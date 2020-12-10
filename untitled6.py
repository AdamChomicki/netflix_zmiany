# WCZYTANIE BIBLIOTEK
import pandas as pd 
import numpy as np 
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer 
from surprise import Reader
from surprise import SVD 
from surprise import Dataset
from surprise.model_selection import cross_validate
import matplotlib.pyplot as plt
import seaborn as sns 

import warnings; warnings.simplefilter('ignore') 

# WCZYTANIE DANYCH
filmy_dane = pd.read_csv(r'C:/Users/Adam/Desktop/netflix_dane_edit/filmy_dane.csv')  

# NIE WIEM, ALE MUSI ZOSTAĆ
filmy_dane['genres'] = filmy_dane['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])  

# PRZYPISANIE L. GŁOSÓW, USTAWIENIE WARTOSCI ODCIECIA
liczba_glosow = filmy_dane[filmy_dane['vote_count'].notnull()]['vote_count'].astype('int') 
srednia_glosow = filmy_dane[filmy_dane['vote_average'].notnull()]['vote_average'].astype('int')
srednia_ze_sredniej_liczby_glosow = srednia_glosow.mean()
liczba_glosow_kwantyl = liczba_glosow.quantile(0.95) 

# WYCIĄGNIĘCIE ROKU DO OSOBNEJ KOLUMNY
filmy_dane['year'] = pd.to_datetime(filmy_dane['release_date'], errors='coerce')

# PRZYPISANIE FILMÓW BRANYCH POD UWAGE W REKOMENDACJACH
zakwalifikowany = filmy_dane[(filmy_dane['vote_count'] >= liczba_glosow_kwantyl) & (filmy_dane['vote_count'].notnull()) & (filmy_dane['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
zakwalifikowany['vote_count'] = zakwalifikowany['vote_count'].astype('int')  
zakwalifikowany['vote_average'] = zakwalifikowany['vote_average'].astype('int')  

# TWORZENIE WYKRESU DOTYCZĄCEGO POPULARNO
zakwalifikowany.popularity = zakwalifikowany.popularity.astype('float32')
popularnosc = zakwalifikowany.sort_values('popularity', ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))
plt.barh(popularnosc['title'].head(5),popularnosc['popularity'].head(5), align='center', color='skyblue')

# WCZYTANIE NOWYCH DANYCH
dane_id = pd.read_csv(r'C:/Users/Adam/Desktop/netflix_dane_edit/dane_id_male.csv')  
dane_id = dane_id[dane_id['tmdbId'].notnull()]['tmdbId'].astype('int')  

# USUNIĘCIE BŁĘDNYCH DANYCH
filmy_dane = filmy_dane.dropna(subset=['release_date'])
filmy_dane = filmy_dane[filmy_dane['release_date'].str.contains("^[0-9]{4}-[0-9]{2}-[0-9]{2}")]

# OCZYSZCZENIE DANYCH
filmy_dane['id'] = filmy_dane['id'].astype('int')  
filmy_dane_join_dane_id = filmy_dane[filmy_dane['id'].isin(dane_id)]  
filmy_dane_join_dane_id['tagline'] = filmy_dane_join_dane_id['tagline'].fillna('')
filmy_dane_join_dane_id['description'] = filmy_dane_join_dane_id['overview'] + filmy_dane_join_dane_id['tagline']  
filmy_dane_join_dane_id['description'] = filmy_dane_join_dane_id['description'].fillna('')  

# UŻYCIE WEKTORYZATORA
wektoryzator_count = CountVectorizer(stop_words='english')
count_macierz = wektoryzator_count.fit_transform(filmy_dane_join_dane_id['description'])
count_macierz.shape

# UŻYCIE PODOBIEŃSTWA COSINUSOWEGO
podobienstwo_cosinusowe = linear_kernel(count_macierz,count_macierz)  
podobienstwo_cosinusowe[0]

# BEZ KOLUMNY INDEX NIE DOPASOWUJE FILMÓW PODOBNYCH
filmy_dane_join_dane_id = filmy_dane_join_dane_id.reset_index()

# PRZYPISANIE TYTUŁÓW DO ZMIENNEJ 'TYTULY'
tytuly = filmy_dane_join_dane_id['title']

# PRZYPISANIE INDEKSÓW DO TYTŁÓW
indeksy = pd.Series(filmy_dane_join_dane_id.index, index=filmy_dane_join_dane_id['title'])  

# PO CZYM TE ROKOMENDACJE?
def uzyskane_rekomendacje(tytul):
    indeks = indeksy[tytul]
    wynik_symulacji = list(enumerate(podobienstwo_cosinusowe[indeks]))
    wynik_symulacji = sorted(wynik_symulacji, key=lambda x: x[1], reverse=True)
    wynik_symulacji = wynik_symulacji[1:31]
    indeksy_filmowe = [i[0] for i in wynik_symulacji]
    return tytuly.iloc[indeksy_filmowe]



uzyskane_rekomendacje('Batman Returns').head(5)
uzyskane_rekomendacje('Harry Potter and the Half-Blood Prince').head(5)




# WCZYTANIE DANYCH
obsada = pd.read_csv(r'C:/Users/Adam/Desktop/netflix_dane_edit/credits.csv') 
slowa_kluczowe = pd.read_csv(r'C:/Users/Adam/Desktop/netflix_dane_edit/keywords.csv') 

# DODANIE OBSADY I SŁÓW KLUCZOWYCH DO FILMOW
filmy_dane = filmy_dane.merge(obsada, on='id') 
filmy_dane = filmy_dane.merge(slowa_kluczowe, on='id') 

# DODANIE 'ID' DO TABELI
filmy_dane_join_dane_id = filmy_dane[filmy_dane['id'].isin(dane_id)] 

# ZMIANA NA LICZBY CAŁKOWITE?
filmy_dane_join_dane_id['cast'] = filmy_dane_join_dane_id['cast'].apply(literal_eval) 
filmy_dane_join_dane_id['crew'] = filmy_dane_join_dane_id['crew'].apply(literal_eval) 
filmy_dane_join_dane_id['keywords'] = filmy_dane_join_dane_id['keywords'].apply(literal_eval)

# DODANIE DO 'FILMY_DANE_JOIN_ ... ' KOLUMN CAST I CREW SIZE
filmy_dane_join_dane_id['cast_size'] = filmy_dane_join_dane_id['cast'] 
filmy_dane_join_dane_id['crew_size'] = filmy_dane_join_dane_id['crew']

# FUNKCJA BIORĄCA POD UWAGĘ REŻYSERA
def otrzymanie_rezysera(x): 
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

# NIE ROZUMIEM TEGO BLOKU
filmy_dane_join_dane_id['Director'] = filmy_dane_join_dane_id['crew'].apply(otrzymanie_rezysera)
filmy_dane_join_dane_id['cast'] = filmy_dane_join_dane_id['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
filmy_dane_join_dane_id['cast'] = filmy_dane_join_dane_id['cast'].apply(lambda x: x[:3] if len(x) >=3 else x) 
filmy_dane_join_dane_id['keywords'] = filmy_dane_join_dane_id['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
filmy_dane_join_dane_id['cast'] = filmy_dane_join_dane_id['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x]) 
filmy_dane_join_dane_id['Director'] = filmy_dane_join_dane_id['Director'].astype('str').apply(lambda x: str.lower(x.replace(" ", ""))) 
filmy_dane_join_dane_id['Director'] = filmy_dane_join_dane_id['Director'].apply(lambda x: [x,x, x]) 

# NIE ROZUMIEM TEGO BLOKU
kluczowe_slowa = filmy_dane_join_dane_id.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)

# ZMIANA NAZWY KOLUMNY
kluczowe_slowa.name = 'keyword'

# WYSTĘPOWANIE SŁÓW KLUCZOWYCH
kluczowe_slowa = kluczowe_slowa.value_counts()
kluczowe_slowa = kluczowe_slowa[kluczowe_slowa > 1]

# UŻYCIE STEMERA
stemmer = SnowballStemmer('english')
stemmer.stem('asked')

# FUNKCJA BIORĄCA POD UWAGĘ SŁOWA KLUCZOWE PRZY REKOMENDACJI
def filtr_slow_kluczowych(x): # 
    words = []
    for i in x:
        if i in kluczowe_slowa:
            words.append(i)
    return words

# NIE ROZUMIEM
filmy_dane_join_dane_id['keywords'] = filmy_dane_join_dane_id['keywords'].apply(filtr_slow_kluczowych)
filmy_dane_join_dane_id['keywords'] = filmy_dane_join_dane_id['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
filmy_dane_join_dane_id['keywords'] = filmy_dane_join_dane_id['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
filmy_dane_join_dane_id['soup'] = filmy_dane_join_dane_id['keywords'] + filmy_dane_join_dane_id['cast'] + filmy_dane_join_dane_id['Director'] + filmy_dane_join_dane_id['genres']
filmy_dane_join_dane_id['soup'] = filmy_dane_join_dane_id['soup'].apply(lambda x: ' '.join(x)) 

# NIE ROZUMIEM
zlicz = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english') 
zlicz_macierz = zlicz.fit_transform(filmy_dane_join_dane_id['soup'])

# NIE ROZUMIEM
podobienstwo_cosinusowe = cosine_similarity(zlicz_macierz, zlicz_macierz) 

# BEZ TEGO, NIE PODAJE WŁASCIWYCH TYTUŁÓW
filmy_dane_join_dane_id = filmy_dane_join_dane_id.reset_index()

# NIE ROZUMIEM
tytuly = filmy_dane_join_dane_id['title']

# NIE ROZUMIEM
indeksy = pd.Series(filmy_dane_join_dane_id.index, index=filmy_dane_join_dane_id['title'])



# UZYSKANA REKOMENDACJA
uzyskane_rekomendacje('Batman Returns').head(5)



# NIE WIEM, ALE WAŻNE
czytelnik = Reader()

# WCZYTANIE ID UŻYTKOWNIKA, ID FILMU I OCENY
oceny = pd.read_csv(r'C:/Users/Adam/Desktop/netflix_dane_edit/ratings_small.csv') 

# NIE WIEM, ALE WAŻNE
dane = Dataset.load_from_df(oceny[['userId', 'movieId', 'rating']], czytelnik)

# METODA SVD
svd = SVD()

# KROSWALIDACJA
cross_validate (svd, dane, measures=['RMSE', 'MAE']) 
    
# WCZYTANIE DANYCH
mapa_id = pd.read_csv(r'C:/Users/Adam/Desktop/netflix_dane_edit/dane_id_male.csv')[['movieId', 'tmdbId']] 
mapa_id.columns = ['movieId', 'id']
mapa_id = mapa_id.merge(filmy_dane_join_dane_id[['title', 'id']], on='id').set_index('title') 
mapa_indeksow = mapa_id.set_index('id') 

# REKOMENDACJA HYBRYDOWA
def hybryda(userId, title): 
    idx = indeksy[title] 
    tmdbId = mapa_id.loc[title]['id']
    print(idx)
    movie_id = mapa_id.loc[title]['movieId']
    
    wynik_cosunisowy = list(enumerate(podobienstwo_cosinusowe[int(idx)])) 
    wynik_cosunisowy = sorted(wynik_cosunisowy, key=lambda x: x[1], reverse=True)  
    wynik_cosunisowy = wynik_cosunisowy[1:10]
    indeksy_filmowe = [i[0] for i in wynik_cosunisowy] 
    
    filmy = filmy_dane_join_dane_id.iloc[indeksy_filmowe][['title', 'vote_count', 'vote_average', 'year', 'id']] 
    filmy['est'] = filmy['id'].apply(lambda x: svd.predict(userId, mapa_indeksow.loc[x]['movieId']).est) 
    filmy = filmy.sort_values('est', ascending=False) 
    return filmy.head(5)

hybryda(1, 'Avatar') 
hybryda(500, 'Avatar')

# METODA .PREDICT NIE OCENIA NASZEJ REKOMENDACJI?
oceny[oceny['userId'] == 1]
svd.predict(1, 302, 3) 
