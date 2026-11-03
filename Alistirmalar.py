##################################################
# Pandas Alıştırmalar
##################################################

import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#########################################
# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
#########################################
df = sns.load_dataset("titanic")
print(df.head())

#########################################
# Görev 2: Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#########################################

print(df['sex'].value_counts())

#########################################
# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
#########################################

unique_counts = df.nunique()
print(unique_counts)

#########################################
# Görev 4: pclass değişkeninin unique değerleri bulunuz.
#########################################

print(df['pclass'].unique())

#########################################
# Görev 5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
#########################################

pclass_nunique = df['pclass'].nunique()
parch_nunique = df['parch'].nunique()

print(f"pclass benzersiz değer sayısı: {pclass_nunique}")
print(f"parch benzersiz değer sayısı: {parch_nunique}")

#########################################
# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz. Tekrar tipini kontrol ediniz.
#########################################

print("İlk Veri Tipi:", df['embarked'].dtype)
df['embarked'] = df['embarked'].astype('category')
print("Yeni Veri Tipi:", df['embarked'].dtype)

#########################################
# Görev 7: embarked değeri C olanların tüm bilgilerini gösteriniz.
#########################################

print(df[df['embarked'] == 'C'])

#########################################
# Görev 8: embarked değeri S olmayanların tüm bilgilerini gösteriniz.
#########################################

print(df[df['embarked'] != 'S'])

#########################################
# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
#########################################

filtre = (df['age'] < 30) & (df['sex'] == 'female')
sonuc_df = df[filtre]
print(sonuc_df)

#########################################
# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
#########################################

koşul = (df['fare'] > 500) | (df['age'] > 70)
sonuc = df[koşul]
print(sonuc)

#########################################
# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
#########################################

bos_degerler = df.isnull().sum()
print(bos_degerler)

#########################################
# Görev 12: who değişkenini dataframe'den düşürün.
#########################################

df.drop("who", axis=1, inplace=True)

#########################################
# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
#########################################

deck_mode = df['deck'].mode()[0] # [0] indeksi, en sık tekrar eden ilk değeri seçmemizi sağlar
df['deck'] = df['deck'].astype('object') #fillna esnasında olası tip/kategori uyuşmazlığı hatalarını önlemesi için
df['deck'] = df['deck'].fillna(deck_mode)#  Boş değerlerin en çok tekrar eden değer (mode) ile doldurulması
print(f"deck değişkeninin modu: {deck_mode}")
print(f"Kalan boş değer sayısı: {df['deck'].isnull().sum()}")

#########################################
# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurun.
#########################################

df['age'] = df['age'].fillna(df['age'].median())

#########################################
# Görev 15: survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
#########################################

sonuc = df.groupby(['pclass', 'sex'])['survived'].agg(['sum', 'count', 'mean'])
print(sonuc)

#########################################
# Görev 16:  30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazınız.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)
#########################################

def age_flag_func(age):
  return 1 if age < 30 else 0

df["age_flag"] = df["age"].apply(lambda x: age_flag_func(x))

#########################################
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#########################################

df2 = sns.load_dataset("tips")
print(df2.head())

#########################################
# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################
sonuc = df2.groupby("time")["total_bill"].agg(["sum", "min", "max", "mean"])
print(sonuc)

#########################################
# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

df2.groupby(["day", "time"])["total_bill"].agg(["sum", "min", "max", "mean"])

#########################################
# Görev 20:Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
#########################################

df2.loc[(df2["sex"] == "Female") & (df2["time"] == "Lunch")].groupby(["day"]).agg({"total_bill": ["sum", "min","max", "mean"], "tip": ["sum", "min","max", "mean"]})

#########################################
# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
#########################################

total_bill_ort = df2.loc[(df2['size'] < 3) & (df2['total_bill'] > 10), 'total_bill'].mean()
print("Hesap (total_bill) Ortalaması:", total_bill_ort)

#########################################
# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
#########################################

df2["total_bill_tip_sum"] = df2["total_bill"] + df2["tip"]

#########################################
# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
#########################################

new_df = df2.sort_values(by='total_bill', ascending=False).head(30)
print(new_df)
