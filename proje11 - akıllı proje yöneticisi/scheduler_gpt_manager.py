"""
problem tanımı: Proje yöneticis
    - bu proje, bir proje dökümanını okuyarak ekip üyelerine gerçek 
    zamanlı görev hatırlatmaları yapan bir YZ sistemi olacak
    - Yapay zeka yöneticisi:
        - pdf 'te bulunan görev zamanına gmre kişilere görevlerini sorar
        - çalışanların verdiği doğal dil cevabını analiz eder
        - eğer görev tamamlanmadıysa tekrardan sorar
        - tamamlanan görevleri tekrardan sormaz
        - tüm sorular gpt tarafından geçmiş cevaplara göre özelleştirilerek sorulur
    - proje 10 sn'de bir 1 dakika ilerleyen simulasyon saati ile çalışacaktır

veri seti: bir proje planı: mobil app geliştirmek için oluşturulmul basit bir proje planı

araçlar ve teknolojiler: rich( terminalde renkli çıktı verir)

plan program
    - pdf reader: proje dokumanını oku
    - gpt agent: proje yönetimi yani taskların sorulması, taskların tamamlanması
    - scheduler_gpt_manager: simulasyonun başlatılması


"""
