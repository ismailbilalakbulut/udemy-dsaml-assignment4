# 🏠 Udemy Veri Bilimi ve Makine Öğrenmesi: 100 Günlük Kamp — 4. Ödev

Bu depo, Udemy’de aldığım Makine Öğrenmesi kursunun ödevleri kapsamında, Pima Indians Diabetes veri setini kullanarak **Sınıflandırma Problemi** üzerine bir çözüm sunmak üzere hazırlanmıştır.

🎯 **Proje Amacı:**
Öğrencinin talimatına uygun olarak, bu projenin temel amacı hiperpametre optimizasyonu yapılmış **AdaBoostClassifier** modelinin performansını optimize etmek ve **Logistic Regression, SVM, Naive Bayes, K-Neighbors, Decision Tree ve Random Forest** gibi diğer yaygın sınıflandırma algoritmalarıyla karşılaştırmaktır.

🌷 **Kullanılan Veri Seti**
* **Veri Seti:** `diabets.csv` (Pima Indians Diabetes Dataset).
* **Hedef Değişken:** `Outcome` (0: Diyabet Değil, 1: Diyabet).
* **Problem Tipi:** Sınıflandırma.

🛠️ **Uygulanan Aşamalar ve Metodoloji**

Proje, verideki eksik veya anlamsız **0 değerlerinin** yönetimine odaklanan iki farklı veri ön işleme (preprocessing) stratejisini karşılaştırmaktadır. Tüm modeller, **%80 Eğitim / %20 Test** ayrımından sonra eğitilmiştir.

### **Metot A: Medyan Doldurma (DiabetAssignment.ipynb)**

Bu yaklaşımda, veri setindeki anlamsız 0 değerleri (özellikle `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` ve `BMI` sütunlarında tespit edilmiştir) veri setinden çıkarılmak yerine doldurulmuştur.

* **Veri Temizleme:** Sütunlardaki 0 değerleri, **eğitim setindeki** sıfır olmayan değerlerin medyanı ile doldurularak veri sızıntısı önlenmiştir.
* **Ön İşleme:** Veri setinin tamamına `StandardScaler` ile standardizasyon uygulanmıştır.
* **Optimizasyon:** AdaBoostClassifier için `GridSearchCV` kullanılarak en iyi hiperpametreler belirlenmiştir.

### **Metot B: Satır Silme (MyDiabetAssignment.ipynb)**

Bu alternatif yaklaşımda, özellikle yüksek oranda 0 içeren `Insulin` sütunundaki 0 değerleri içeren tüm satırlar veri setinden silinmiştir.

* **Veri Temizleme:** `Insulin` değeri 0 olan satırlar atılmıştır, bu da örnek sayısını **768'den 394'e** düşürmüştür.
* **Kalan Eksikler:** Kalan az sayıdaki anlamsız 0 değeri (`Glucose`, `BMI` sütunlarında) yine medyan ile doldurulmuştur.
* **Ön İşleme:** Veri setinin tamamına `StandardScaler` ile standardizasyon uygulanmıştır.
* **Optimizasyon:** AdaBoostClassifier için `GridSearchCV` kullanılarak en iyi hiperpametreler belirlenmiştir.

***

### ✅ **Sonuçlar ve Performans Değerlendirmesi**

Model performansı, temel olarak **Doğruluk (Accuracy)**, **Hassasiyet (Precision)** ve **Geri Çağırma (Recall)** metrikleri üzerinden değerlendirilmiştir.

#### **1. AdaBoostClassifier Performans Karşılaştırması (Metot A vs. Metot B)**

Veri temizleme stratejilerinin AdaBoost performansı üzerindeki etkisi incelenmiştir:

| Metodoloji | Veri Sayısı | En İyi Hiperpametreler | Test Doğruluğu (Accuracy) |
| :--- | :--- | :--- | :--- |
| **Metot A (Medyan Doldurma)** | 768 | `{'learning_rate': 1, 'n_estimators': 150}` | **0.7597** |
| **Metot B (Satır Silme)** | 394 | `{'learning_rate': 0.1, 'n_estimators': 200}` | **0.7848** |

**Bulgu:** `Insulin` değeri 0 olan satırların silinmesi (`Metot B`), örnek sayısı azalsa bile AdaBoostClassifier için **daha yüksek bir test doğruluğu** (%78.48) sağlamıştır.

#### **2. Diğer Modellerle Kıyaslama Özeti**

| Model | Metot A: Test Doğruluğu (Medyan Doldurma) | Metot B: Test Doğruluğu (Satır Silme - AdaBoost) |
| :--- | :--- | :--- |
| **AdaBoostClassifier (Optimize)** | **0.7597** | **0.7848** |
| Logistic Regression (Optimize) | **0.7468** | 0.7215 |
| Support Vector Machine (Optimize) | **0.7468** | 0.6962 |
| K-Neighbors Classifier (Optimize) | 0.7403 | 0.6709 |
| Random Forest Classifier (Optimize) | 0.7338 | 0.7721 |
| Decision Tree (Optimize) | 0.7273 | 0.6835 |
| Naive Bayes | 0.7208 | 0.7089 |

**Genel Sonuç:**
AdaBoostClassifier, `Insulin` değeri 0 olan satırların atıldığı veri seti üzerinde en iyi performansı gösteren model olmuştur. Bu durum, `Insulin` değeri 0 olan hastaların çoğunun gerçekten diyabet hastası olmaması veya bu 0 değerlerinin model için yanıltıcı bir gürültü oluşturması nedeniyle ortaya çıkmış olabilir. Her iki metodolojide de en yüksek doğruluk değerleri (Metot A: Logistic Regression/SVM - %74.68; Metot B: AdaBoostClassifier - %78.48) elde edilmiştir.
