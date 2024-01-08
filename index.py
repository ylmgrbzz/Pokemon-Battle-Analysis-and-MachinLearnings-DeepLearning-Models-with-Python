import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve,mean_squared_error, r2_score,classification_report
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import neighbors
from sklearn.svm import SVC
import xgboost
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import tensorflow as tf
from tensorflow.keras.applications import VGG16 , ResNet50 , InceptionV3
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import os


import numpy as np

# Pokémon ve savaş verilerini içeren CSV dosyalarını oku
pokemon = pd.read_csv('pokemon.csv')
combat = pd.read_csv('combats.csv')

# Pokémon isimlerini içeren bir listeyi oluştur
pokemonlist = list(pokemon["Name"].unique())

# Verilen Pokémon ismine göre ID'yi getiren bir fonksiyon tanımla
def idgetir(name):
    df = pokemon.set_index("Name")
    id = df.loc[name]["#"]
    return id

# Streamlit arayüzünde iki sütun oluştur
col1, col2 = st.columns(2)

# Birinci sütunda: İlk Pokémon'u seçme kutusu ve ilgili resmi gösterme
with col1:
    poke1 = st.selectbox("First Pokemon", pokemonlist)
    poke1 = poke1.replace("Mega ", "").replace(" ", "").replace(" X", "").replace("♂", "").replace("♀", "")
    link1 = "./images/images/" + poke1.lower() + ".png"
    st.image(link1)

# İkinci sütunda: İkinci Pokémon'u seçme kutusu ve ilgili resmi gösterme
with col2:
    poke2 = st.selectbox("Second Pokemon", pokemonlist)
    poke2 = poke2.replace("Mega ", "").replace(" ", "").replace(" X", "").replace("♂", "").replace("♀", "")
    link2 = "./images/images/" + poke2.lower() + ".png"
    st.image(link2)

# Savaş verilerini içeren bir DataFrame oluştur
cdf = combat
# Kazanan sütununu oluştur: 1, eğer birinci Pokémon kazanmışsa; 0, eğer ikinci Pokémon kazanmışsa
cdf["Winner"] = cdf["First_pokemon"] == cdf["Winner"]
cdf["Winner"] = np.where(cdf["Winner"], 0, 1)

# Bağımlı değişken (y) ve bağımsız değişkenler (x) ayrıştırma
y = cdf[["Winner"]]
x = cdf.drop("Winner", axis=1)

# Veri setini eğitim ve test setlerine bölmek
trainsec = st.sidebar.slider("Train Size", 0, 100, 80)
trainsec = trainsec / 100
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=trainsec)

# Kullanıcıdan seçilen modeli belirleme
modelsec = st.sidebar.selectbox("Model Seç", ["Decision Tree", "Random Forest", "LogisticRegression", "KNeighborsClassifier", "SVC",
                                              "MLPClassifier", "GradientBoostingClassifier", "LGBMClassifier", "CatBoostClassifier"])

# Seçilen modele göre bir model oluştur ve eğitme
if modelsec=="Decision Tree":
    tree=DecisionTreeClassifier()
    model=tree.fit(x_train,y_train)
    cart_tuned = DecisionTreeClassifier(max_depth=5, min_samples_split=30).fit(x_train, y_train)
    y_pred = cart_tuned.predict(x_test)

elif modelsec=="Random Forest":
    agacsec=st.sidebar.number_input("Ağaç Sayısı",value=100)
    forest=RandomForestClassifier(n_estimators=agacsec)
    model=forest.fit(x_train,y_train)

elif modelsec=="LogisticRegression":
    randomState=st.sidebar.number_input("Random State",value=40)
    loj_model  = LogisticRegression(solver="liblinear",random_state=randomState)
    model=loj_model .fit(x_train,y_train)
    y_pred = loj_model.predict(x_test)

elif modelsec == "KNeighborsClassifier":
    knn_model = KNeighborsClassifier()
    model = knn_model.fit(x_train, y_train)
    knn_tuned = KNeighborsClassifier(n_neighbors=11).fit(x_train, y_train)
    y_pred = knn_tuned.predict(x_test)
elif modelsec == "MLPClassifier":
    mlp_model = MLPClassifier()
    model = mlp_model .fit(x_train, y_train)
    mlp_tuned = MLPClassifier(solver="lbfgs", activation="logistic",
                              alpha=5, hidden_layer_sizes=(100, 100)).fit(x_train, y_train)
    y_pred = mlp_tuned.predict(x_test)

elif modelsec == "SVC":
    svm_model= SVC(kernel = "linear")
    model = svm_model.fit(x_train, y_train)
    svm_tuned  = SVC(kernel="linear",C=2).fit(x_train,y_train)
    y_pred = svm_tuned.predict(x_test)

elif modelsec == "LGBMClassifier":
    lgb_model  = LGBMClassifier()
    model = lgb_model.fit(x_train, y_train)
    lgb_tuned = LGBMClassifier(learning_rate=0.1,
                               max_depth=1,
                               n_estimators=40).fit(x_train, y_train)
    y_pred = lgb_tuned.predict(x_test)

elif modelsec == "CatBoostClassifier":
    catb_model   = CatBoostClassifier()
    model = catb_model.fit(x_train, y_train)
    catb_tuned = CatBoostClassifier(depth=8,
                                    iterations=200,
                                    learning_rate=0.01).fit(x_train, y_train)
    y_pred = catb_tuned.predict(x_test)


# Streamlit arayüzünde üç sütun oluştur
col1, col2, col3 = st.columns(3)

# Birinci sütunda: Boş
with col1:
    pass

# İkinci sütunda: "Savaş Başlasın" adlı bir düğme oluştur
with col2:
    saldir = st.button("Savaş Başlasın")

# Üçüncü sütunda: Boş
with col3:
    pass

# I. Epoch Değeri
epochs = st.sidebar.number_input("Epochs", value=50)

# II. Batch Size Değeri
batch_size = st.sidebar.number_input("Batch Size", value=32)

# III. Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# IV. Model Checkpoint
model_checkpoint = ModelCheckpoint(filepath='best_model.h5', save_best_only=True)

# Eğer "Savaş Başlasın" düğmesine basıldıysa
if saldir:
    # Modelin tahminini yap ve sonucu ekrana yazdır
    sonuc = model.predict([[idgetir(poke1), idgetir(poke2)]])
    st.write(int(sonuc))
    
    # Eğer modelin tahmini 0 ise (birinci Pokémon kazandı ise)
    if sonuc == 0:
        st.header("Kazanan")
        st.image(link1)
        st.write("Model Skoru", model.score(x_test, y_test))
        st.write("Accuracy Skor", accuracy_score(y_test, y_pred))
    # Eğer modelin tahmini 1 ise (ikinci Pokémon kazandı ise)
    else:
        st.header("Kazanan")
        st.image(link2)
        st.write("Model Skoru", model.score(x_test, y_test))
        st.write("Accuracy Skor", accuracy_score(y_test, y_pred))

    # Confusion Matrix ve Classification Report hesapla ve ekrana yazdır
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    st.subheader("Confusion Matrix:")
    st.write(conf_matrix)

# "Winner" sütununu oluştur ve kazananın 1, kaybedenin 0 olduğu bir kodu uygula
cdf["Winner"] = np.where(cdf["Winner"], 0, 1)

# Veriyi standartlaştır
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Kategorik değişkenleri ikili formata çevir
y_train_encoded = tf.keras.utils.to_categorical(y_train)
y_test_encoded = tf.keras.utils.to_categorical(y_test)

# "Winner" sütunundaki değerleri say ve ekrana yazdır
print(cdf["Winner"].value_counts())

# Resimlerin bulunduğu klasörü belirle
image_folder = "./images/images/"

# Klasördeki tüm resimleri al
all_images = os.listdir(image_folder)

# Resim isimlerini uzantıları olmadan bir listeye ekle
image_names = [os.path.splitext(image)[0] for image in all_images]

# Kullanıcının seçebileceği bir resim seçme kutusu oluştur
selected_image = st.sidebar.selectbox("Resim Seç", image_names)

# Seçilen resmin dosya yolunu oluştur
selected_image_path = os.path.join(image_folder, selected_image + ".png")

# Seçilen resmi ekrana yazdır
st.sidebar.image(selected_image_path)


# Resim boyutunu belirle
img_size = (224, 224)

# CNN modelini yüklemek için bir fonksiyon tanımla
def load_cnn_model(model_name):
    # Model ismi 'VGG16' ise VGG16 modelini yükle
    if model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # Model ismi 'ResNet50' ise ResNet50 modelini yükle
    elif model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # Model ismi 'InceptionV3' ise InceptionV3 modelini yükle
    elif model_name == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    else:
        raise ValueError("Geçersiz model ismi")

    # Flatten, Dense ve çıkış katmanını ekleyerek CNN modelini oluştur
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x) 
    cnn_model = Model(inputs=base_model.input, outputs=predictions)

    # Modeli derleme
    cnn_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return cnn_model

# Kullanıcının seçebileceği CNN modellerini içeren bir seçim kutusu oluştur
cnn_model_name = st.sidebar.selectbox("CNN Modelini Seç", ['VGG16', 'ResNet50', 'InceptionV3'])

# Seçilen CNN modelini yükle
cnn_model = load_cnn_model(cnn_model_name)

# Eğer "Savaş Başlasın" düğmesine basıldıysa
if saldir:
    # Seçilen resmi yükle
    img = image.load_img(selected_image_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Model ismine göre resmi ön işleme
    if cnn_model_name == 'VGG16':
        img_array = vgg_preprocess_input(img_array)
    elif cnn_model_name == 'ResNet50':
        img_array = resnet_preprocess_input(img_array)
    elif cnn_model_name == 'InceptionV3':
        img_array = inception_preprocess_input(img_array)

    # CNN modeli ile tahmin yap
    cnn_prediction = cnn_model.predict(img_array)

    # Tahmin sonucunu ekrana yazdır
    st.write(f"{cnn_model_name} Model Tahmini: {int(cnn_prediction[0, 0] > 0.5)}")


# Sklearn kütüphanesinden sınıflandırma raporu ve confusion matrix hesapla ve ekrana yazdır
from sklearn.metrics import classification_report, confusion_matrix
y_pred = model.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

if modelsec == "Random Forest":
    feature_importances = forest.feature_importances_
    print("Feature Importances:")
    print(dict(zip(x.columns, feature_importances)))

elif modelsec == "Decision Tree":
    feature_importances = cart_tuned.feature_importances_
    print("Feature Importances:")
    print(dict(zip(x.columns, feature_importances)))

elif modelsec == "LogisticRegression":
    feature_importances = loj_model.coef_
    print("Feature Importances:")
    print(dict(zip(x.columns, feature_importances)))

elif modelsec == "KNeighborsClassifier":
    feature_importances = knn_tuned.feature_importances_
    print("Feature Importances:")
    print(dict(zip(x.columns, feature_importances)))

elif modelsec == "MLPClassifier":
    feature_importances = mlp_tuned.feature_importances_
    print("Feature Importances:")
    print(dict(zip(x.columns, feature_importances)))

elif modelsec == "SVC":
    feature_importances = svm_tuned.feature_importances_
    print("Feature Importances:")
    print(dict(zip(x.columns, feature_importances)))

elif modelsec == "LGBMClassifier":

    feature_importances = lgb_tuned.feature_importances_
    print("Feature Importances:")
    print(dict(zip(x.columns, feature_importances)))

elif modelsec == "CatBoostClassifier":

    feature_importances = catb_tuned.feature_importances_
    print("Feature Importances:")
    print(dict(zip(x.columns, feature_importances)))



