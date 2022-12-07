# **Laporan Proyek Machine Learning – Sri Putriana**

## **Domain Proyek**

Sejak beberapa tahun belakangan, investasi tidak lagi hanya seputar saham dan tabungan. Aset diam seperti rumah juga makin meningkat peminatnya, terutama di kota-kota besar, seperti Jakarta Selatan. Harga yang terus meningkat seiring inflasi juga menjadi alasan penting kebanyakan orang memilih rumah sebagai investasi jangka panjang. 

Namun, bagi orang yang baru ingin memulai, terkadang muncul permasalahan yaitu ketidaktahuan akan harga. Seringkali melihat rumah hanya dari luas tanah dan bangunan saja tanpa mempertimbangkan aspek lain, yang berakibat buruk ketika nantinya akan menjual kembali rumah tersebut. Oleh karena itu, diperlukan sebuah sistem untuk membantu manusia dalam hal penentuan harga jual dan harga beli sebuah rumah.

Berikut merupakan artikel dan penelitian, terkait proyek yang akan dilakukan:

* [Peluang dan Tantangan Investasi Properti di Indonesia](https://core.ac.uk/download/pdf/230768312.pdf)
* [Analisis dan prediksi harga properti menggunakan teknik machine learning di Bogor dan Depok](https://repository.mercubuana.ac.id/67364/)
* [Implementasi artificial intelligence untuk memprediksi harga penjualan rumah menggunakan random forest dan Flask](https://dspace.uii.ac.id/handle/123456789/29813)


## **Business Understanding**
### **Problem Statements**

Berdasarkan kondisi yang telah diuraikan di atas, berikut adalah beberapa permasalahan yang dapat diselesaikan pada proyek ini:
- Bagaimana cara membuat sebuah model Machine Learning untuk memprediksi harga rumah di Jakarta Selatan?
- Dari serangkaian fitur yang ada, fitur apakah yang paling berpengaruh terhadap harga sebuah rumah di Jakarta Selatan?

### **Goals**

Berikut merupakan tujuan dari dibuatnya proyek ini:
- Membuat sebuah model untuk memprediksi harga rumah di Jakarta Selatan.
- Mengetahui fitur yang paling berkorelasi dengan harga rumah di Jakarta Selatan.


### **Solution statements**
Di bawah ini merupakan solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek:
- Menggunakan 2 macam algoritma dalam pembuatan model Machine Learning, yaitu:

    * `K-Nearest Neighbor` (KNN)
    * `Random Forest`

- Melakukan teknik *Exploratory Data Analysis* (`EDA`) *Multivariate*, untuk melihat hubungan (korelasi) antar fitur.
    
    * menggunakan fungsi `pairplot` dari library `Seaborn` untuk melihat hubungan yang terbentuk antar fitur.
    * Menggunakan `correlation_metrix` (metrik korelasi) untuk melihat skor korelasi setiap fitur terhadap fitur target `HARGA`.


## **Data Understanding**

Data yang digunakan pada proyek ini adalah daftar harga rumah di area Jakarta Selatan yang terdiri dari 7 kolom berisi 5 buah data numerik dan 2 buah data kategori. Adapun datanya sendiri terdiri dari 1001 baris. Berikut dataset yang digunakan pada proyek ini:
 [Kaggle Dataset: Harga Rumah Jaksel]( https://www.kaggle.com/datasets/wisnuanggara/daftar-harga-rumah)

### Variabel-variabel pada dataset `Harga Rumah Jaksel` adalah sebagai berikut:
- `HARGA` : merupakan data harga dari rumah.
- `LT` : merupakan jumlah luas tanah.
- `LB` : merupakan jumlah luas bangunan.
- `JKT` : merupakan jumlah kamar tidur.
- `JKM` : merupakan jumlah kamar mandi.
- `GRS` : merupakan keterangan adanya garasi atau tidak.
- `KOTA` : merupakan nama kota.

### Tahapan awal visualisasi data:
1) Menghapus variabel `KOTA`, karena semua nilainya sama.
   * ![](https://github.com/anaana92/Assets/blob/main/kota.png)
2) Mengecek *mising value* dan *outliers* pada data.
   * ![*missing value*](https://github.com/anaana92/Assets/blob/main/missval.png)
   * ![*outlier*](https://github.com/anaana92/Assets/blob/main/outlier.png)

### Visualisasi Data
1) *Univariate Analysis*
   * Menampilkan jumlah sampel dan presentase dari variabel `GRS`
        ![](https://github.com/anaana92/Assets/blob/main/grs1.png)
   * Menampilkan histogram dari setiap fitur numerik
        ![](https://github.com/anaana92/Assets/blob/main/hist.png)

2) *Multivariate Analysis*
   * Melihat pengaruh fitur kategori `GRS` terhadap fitur target `HARGA`
      * ![](https://github.com/anaana92/Assets/blob/main/grs2.png)
   * Melihat korelasi antar fitur numerik
      * ![](https://github.com/anaana92/Assets/blob/main/kor1.png)
   * Melihat korelasi antar fitur numerik dengan fitur target `HARGA`
      * ![](https://github.com/anaana92/Assets/blob/main/kor2.png)

## **Data Preparation**

Berikut merupakan teknik data preparation yang dilakukan pada proyek:
-	Melakukan proses Encoding fitur kategori, yaitu variabel `GRS`
-	Melakukan proses reduksi dimensi dengan `PCA` (*Principal Component Analysis*)
-	Melakukan proses pembagian dataset menjadi data latih dan data uji, dengan perbandingan 80:20
-	Melakukan proses standarisasi data menggunakan `StandardScaler`

## **Modelling**

Pengembangan model pada proyek ini menggunakan 2 macam algoritma, yaitu:
1)	`K-Nearest Neighbor` (KNN), bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k-tetangga terdekat. Untuk melatih model menggunakan algoritma ini, dilakukan beberapa tahapan, yaitu:
    * Mengimport `KNearestNeighbor` dari *library* `scikit-learn`
  
     ``` python 
     from sklearn.neighbors import KNeighborsRegressor
     ```

    * Membuat variabel `knn`, dan lakukan pemanggilan algoritma dengan nilai parameter berikut.
  
    ``` python
    knn = KNeighborsRegressor(n_neighbors=10)
    ```
    parameter k=10 artinya, mengecek 10 tetangga terdekat.

2) `Random Forest`, pada dasarnya adalah versi bagging dari algoritma *decision tree*. Model *decision tree* masing-masing memiliki hyperparameter yang berbeda dan dilatih pada beberapa bagian (subset) data yang berbeda juga. Teknik pembagian data pada algoritma decision tree adalah memilih sejumlah fitur dan sejumlah sampel secara acak dari dataset yang terdiri dari n fitur dan m sampel. Berikut tahapan yang dilakukan untuk pengaplikasian algoritma ini:
   * Mengimport `RandomForestRegressor` dari *library* `scikit-learn`
  
    ``` python
    from sklearn.ensemble import RandomForestRegressor
    ```

   * Buat variabel `RF` dan lakukan pemanggilan algoritma dengan nilai parameter berikut.
  
    ``` python
    RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
    ```
    a) n_estimator=50 adalah jumlah trees(pohon) di forest.

    b) max_depth=16 adalah ukuran seberapa banyak pohon dapat membelah untuk membagi node ke dalam jumlah pengamatan yang diinginkan.

    c) random_state=55 digunakan untuk mengontrol random sumber generator yang digunakan.

    d)  n_jobs=1 adalah jumlah pekerjaan, artinya semua proses berjalan secara paralel



## **Evaluation**

Model pada proyek ini menggunakan metric `MSE` (*Mean Squared Error*), yang cara kerjanya ialah menghitung selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi, yang didefinisikan pada persamaan berikut.
  * ![](https://github.com/anaana92/Assets/blob/main/mse1.png)
  
Berikut ini adalah hasil evaluasi pada data latih dan data uji dengan metrik `MSE`:
  * ![](https://github.com/anaana92/Assets/blob/main/mse2.png)
  
Agar lebih mudah dipahami, dilakukan plot pada metrik dengan bar chart sehingga tampilannya menjadi seperti berikut:
  * ![](https://github.com/anaana92/Assets/blob/main/mse3.png)
  
Dari gambar tersebut, dapat dilihat bahwa model dengan algoritma `Random Forest` memberikan nilai error yang lebih kecil daripada algoritma linnya yaitu `K-Nearest Neighbor` (KNN). Dan berikut hasil uji model yang dilakukan pada kedua algoritma:
  * ![](https://github.com/anaana92/Assets/blob/main/mse4.png)
  
Terlihat jelas bahwa prediksi model dengan algoritma `Random Forest` paling mendekati nilai aslinya, sehingga model inilah yang dipilih sebagai model terbaik pada proyek prediksi harga rumah di Jakarta Selatan.
