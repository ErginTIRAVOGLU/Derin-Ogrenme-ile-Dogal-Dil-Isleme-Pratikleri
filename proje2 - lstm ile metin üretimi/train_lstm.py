"""
problem tanımı: LSTM ile metin üretme -> Verilen kelimelerden anlamlı türkçe cümleler oluşturması
    - ben yarın ... (lstm boşluk dolduracak)    

lstm: Long Short Term Memory

veri seti: chatgpt ile oluşturulan günlük hayat cümlesi.
    - kitap okumak beni gerçekten mutlu ediyor
    - akşam yemeğinde pizza yemeği planlıyorum
    - sinemaya gitmek herzaman keyifli bir aktivite
    - sabah koşusu bana enerji veriyor

plan/program: 

install libraries (pip), requirements.txt

import libraries

"""
# import libraries

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# eğitim verisini chatgpt ile oluştur 
#   1- İnternetten veri bulma
#   2- Gerçek hayat veri toplama
#   3- Simülasyon verisi (Bilgisayar ortamında simüle edilmesi) Chatgpt Grok Gemini gibi...


data = [
    "Bugün hava, top oynamak için çok güzel",
"Markete gidip, ekmek almam gerekiyor",
"Kahvemi, sabah içmeden güne başlayamıyorum",
"Toplantı, saat üçte başlayacak",
"Derslerimi, zamanında bitirmeliyim",
"Arkadaşımla, sinemaya gitmeyi planlıyorum",
"Telefonumu, evde unuttum",
"Yarın, erken kalkmam lazım",
"Yemek tariflerini, denemeyi seviyorum",
"Spor salonuna, haftada üç kez gidiyorum",
"Otobüs, çok kalabalıktı bugün",
"Yeni bir kitap, almam gerekiyor",
"Müzik dinlemek, ruhumu dinlendiriyor",
"Kedim, bugün çok uykucu",
"Bilgisayarımla, bir proje hazırlıyorum",
"Yağmur yağdığı için, dışarı çıkmadım",
"Arkadaşımı, uzun zamandır görmedim",
"Ders çalışırken, müzik dinlemeyi seviyorum",
"Kahvaltıda, peynir ve zeytin yedim",
"Akşam yemeğini, erken hazırlamalıyım",
"Film izlemek için, zaman ayırdım",
"Bugün kendime, biraz vakit ayıracağım",
"Havaalanına taksiyle gitmek, daha hızlı",
"Ödevimi, bitirmem gerekiyor",
"Yeni bir dil, öğrenmeye başladım",
"Sokakta, yürüyüş yapmayı seviyorum",
"Bakkaldan, süt almam lazım",
"Yağmur sonrası, hava çok temiz",
"Arkadaşlarla buluşup, sohbet ettik",
"Hafta sonu, gezmeye çıkmayı planlıyorum",
"Kahve molası vermek, iyi geliyor",
"Ders notlarımı, düzenlemeliyim",
"Yeni bir dizi, izlemeye başladım",
"Marketten, sebze ve meyve aldım",
"Evde, temizlik yapmak gerekiyor",
"Pazar günü, kahvaltıya davetliyim",
"Yeni tarifler, denemeyi seviyorum",
"Spor yaptıktan sonra duş almak, ferahlatıcı",
"Hobilerime, daha fazla vakit ayırmalıyım",
"Arkadaşım, bana kitap hediye etti",
"Yolda, trafik yoğundu",
"Sabahları koşu yapmak, bana enerji veriyor",
"Film festivali için, bilet aldım",
"Evde, sessiz bir ortamda çalışmayı seviyorum",
"Arkadaşlarla oyun oynamak, çok eğlenceli",
"Bugün, markete uğramadan geçemem",
"Yeni bir müzik listesi, oluşturdum",
"Kitap okumak, stresimi azaltıyor",
"Akşam yürüyüşü yapmak, iyi geliyor",
"Kendi kahvemi, kendim hazırlıyorum",
"Bugün, biraz dinlenmeye ihtiyacım var",
"Hafta içi, iş yoğun geçiyor",
"Arkadaşımın doğum gününü, kutladık",
"Yemek pişirmek, bana keyif veriyor",
"Sabahları çay içmek, bana huzur veriyor",
"Arkadaşımı, telefonla aradım",
"Bugün, biraz erken uyandım",
"Yolda, yağmur yağmaya başladı",
"Öğle yemeğinde, pizza yedim",
"Hafta sonu, sinemaya gitmek istiyorum",
"Yeni bir spor ayakkabı, aldım",
"Evde, biraz kitap okumak istiyorum",
"Kedim, oyuncaklarıyla oynuyor",
"Ders çalışmak, bazen sıkıcı olabiliyor",
"Kahvaltıda, simit ve çay içtim",
"Toplantıya zamanında yetişmek, zorundayım",
"Arkadaşlarla, kahve içmeye gittik",
"Bugün, çok sıcak bir gün",
"Akşam yemeğini, dışarıda yedik",
"Yürüyüş yaparken, müzik dinlemeyi seviyorum",
"Yeni bir hobi, edinmek istiyorum",
"Hafta içi, işlerim çok yoğun",
"Sabah koşusu yapmak, bana enerji veriyor",
"Film izlerken, patlamış mısır yedim",
"Pazar günü, kahvaltıya davetliyim",
"Marketten, peynir ve zeytin aldım",
"Arkadaşlarımla oyun oynamak, eğlenceli",
"Yeni bir müzik albümü, dinledim",
"Hava bugün, çok rüzgârlı",
"Evde, temizlik yapmak zorundayım",
"Akşam yürüyüşüne çıkmayı, planlıyorum",
"Yeni tarifler, denemeyi seviyorum",
"Ders notlarımı, tekrar gözden geçirdim",
"Telefonumu, şarj etmeyi unuttum",
"Arkadaşımın doğum gününü, kutladık",
"Sokakta, yürüyüş yapmayı seviyorum",
"Hafta sonu, gezmeye çıkmayı planlıyorum",
"Kahve molası vermek, iyi geliyor",
"Yeni bir dil, öğrenmeye başladım",
"Bilgisayarımla, bir proje üzerinde çalışıyorum",
"Bugün kendime, biraz vakit ayıracağım",
"Otobüs, çok kalabalıktı bugün",
"Kedim, bugün çok uykucu",
"Yağmur sonrası, hava çok temiz",
"Ders çalışırken, müzik dinlemeyi seviyorum",
"Yeni bir dizi, izlemeye başladım",
"Evde sessiz bir ortamda çalışmayı, seviyorum",
"Film festivali için, bilet aldım",
"Sabahları kitap okumak, alışkanlığım oldu",
"Akşam yemeğini, erken hazırlamalıyım",
"Arkadaşlarımla sohbet etmek, iyi geliyor",
"Yolda, trafik yoğundu",
"Hobilerime, daha fazla vakit ayırmalıyım",
"Kendi kahvemi, kendim hazırlıyorum",
"Bugün, biraz dinlenmeye ihtiyacım var", 
"Bugün, futbol oynamaya gideceğim",
"Hafta sonu, basketbol turnuvası var",
"Sabahları koşu yapmak, bana enerji veriyor",
"Yüzme dersine, kayıt oldum",
"Spor salonunda, ağırlık çalışıyorum",
"Tenis oynamayı, öğrenmek istiyorum",
"Dün, voleybol maçı izledim",
"Koşu bandında, 30 dakika koştum",
"Pilates yaparken, çok rahatlıyorum",
"Yürüyüş yapmak, hem sağlıklı hem keyifli",
"Hafta içi, spor salonuna gitmeyi planlıyorum",
"Dün akşam, yoga dersi aldım",
"Futbol maçında, gol attım",
"Basketbol antrenmanım, iptal oldu",
"Spor yapmak, stres atmamı sağlıyor",
"Bisiklete binmek için, güzel bir gün",
"Masa tenisi oynamayı, seviyorum",
"Spor yaparken, müzik dinlemeyi seviyorum",
"Futbol takımım, finale kaldı",
"Koşu sırasında, yeni parkları keşfettim",
"Dün sabah, jimnastik yaptım",
"Yüzme havuzuna, üye oldum",
"Spor malzemelerini, yenilemem gerekiyor",
"Spor yapmayı, rutin hâline getirdim",
"Basketbol maçı, çok heyecanlıydı",
"Yoga pozlarını doğru yapmak, önemli",
"Hafta sonu, yürüyüşe çıkmayı planlıyorum",
"Sporcu beslenmesine, dikkat etmeliyim",
"Dün koşuda, rekor kırdım",
"Tenis kortu, kiraladım",
"Evde, fitness programına başladım",
"Koşarken, arkadaşlarımla sohbet ettim",
"Spor müsabakalarını, televizyondan izledim",
"Basketbol oynarken, sakatlandım",
"Spor yaparken, motivasyon önemli",
"Yüzme sırasında, nefes tekniklerini geliştirdim",
"Pilates dersleri, çok faydalı",
"Haftada üç gün, koşu yapıyorum",
"Futbol turnuvasına, katıldım",
"Spor sonrası, esneme hareketleri yaptım",
"Koşu bandında, 5 km koştum",
"Dün sabah, yüzme yaptım",
"Basketbol maçı, çok çekişmeliydi",
"Voleybol oynarken, çok eğlendim",
"Evde egzersiz yapmak, kolay ve pratik",
"Spor salonunda, yeni arkadaşlar edindim",
"Hafta sonu, doğa yürüyüşü yaptım",
"Koşu yaparken, tempo tutturmak önemli",
"Tenis oynamak, kondisyonu artırıyor",
"Spor aktiviteleri, ruh halini iyileştiriyor",
"Dün futbol antrenmanı, çok yoğundu",
"Spor sonrası duş almak, ferahlatıcı",
"Seçimler, bu yıl Ekim ayında yapılacak",
"Parlamento, yeni bir yasa tasarısını görüştü",
"Siyasi parti, programını açıkladı",
"Cumhurbaşkanı, bugün önemli bir açıklama yaptı",
"Yerel seçimler için, kampanyalar başladı",
"Hükûmet, ekonomik reformları tartışıyor",
"Siyasi tartışmalar, televizyonda yayınlandı",
"Meclis oturumunda, gergin anlar yaşandı",
"Seçim sonuçları, yakından takip ediliyor",
"Siyasi liderler, basın toplantısı düzenledi",
"Yerel yönetimler, altyapı projeleri başlattı",
"Seçim propagandaları, sosyal medyada yayıldı",
"Siyasi partiler, koalisyon görüşmeleri yapıyor",
"Halk, referandum için oy kullanacak",
"Parti liderleri, televizyon programına katıldı",
"Siyasi gündem, ekonomi ve sağlık üzerine yoğunlaştı",
"Seçim anketleri, halkın nabzını ölçüyor",
"Yeni yasa tasarısı, tartışmalara neden oldu",
"Hükûmet, reform paketini Meclise sundu",
"Siyasi kampanyalar, sokaklarda aktifleşti",
"Seçim sonrası, koalisyon hükümeti kurulacak",
"Siyasi partiler, seçim manifestolarını açıkladı",
"Meclis komisyonları, yasa tasarılarını inceledi",
"Siyasi tartışmalar, sosyal medyada da sürdü",
"Cumhurbaşkanı, yurtdışına resmi ziyaret yaptı",
"Siyasi liderler, ekonomi politikalarını anlattı",
"Seçim güvenliği için, önlemler alındı",
"Halkın siyasi katılımı, yüksek oldu",
"Siyasi partiler, genç seçmenleri hedefliyor",
"Mecliste, bütçe görüşmeleri başladı",
"Seçim sonuçları, kamuoyuna duyuruldu",
"Siyasi liderler, halkla buluşma toplantısı yaptı",
"Hükûmet, reformları halkla paylaştı",
"Seçim kampanyaları, televizyon ve radyo üzerinden yürütülüyor",
"Siyasi tartışmalarda, ekonomi ve eğitim ön plana çıktı",
"Mecliste, gergin tartışmalar yaşandı",
"Seçim öncesi, siyasi liderler miting düzenledi",
"Siyasi partiler, programlarını basına açıkladı",
"Halk, referandum sonuçlarını bekliyor",
"Cumhurbaşkanı, yeni yasa tasarısını imzaladı",
"Siyasi tartışmalar, gazetelerde geniş yer buldu",
"Seçim propagandaları, sokak afişleri ile yapıldı",
"Hükûmet, yeni projeleri halka tanıttı",
"Siyasi liderler, partilerini güçlendirmeye çalışıyor",
"Mecliste, yeni komisyonlar kuruldu",
"Siyasi kampanyalar, sosyal medyada yoğunlaştı",
"Seçim sonuçlarına göre, koalisyon yapılacak",
"Halkın siyasi bilinç düzeyi, arttı",
"Siyasi tartışmalar, televizyon programlarında sürdü",
"Cumhurbaşkanı, önemli bir konuşma yaptı",
"Seçim sonuçları, ülkede gündem oluşturdu",
"Parti liderleri, ekonomik planlarını açıkladı",
"Müze gezmek, bana çok şey katıyor",
"Geleneksel yemekleri, denemeyi seviyorum",
"Kültürel festival, hafta sonu düzenlenecek",
"Sanat sergisine, gitmek istiyorum",
"Tiyatro oyunu, çok beğenildi",
"Kütüphaneden, yeni kitaplar aldım",
"Farklı kültürleri tanımak, ilginç geliyor",
"Resim sergisi, çok ilgi çekiciydi",
"Kültürel etkinlikler, şehirde yoğunlaşıyor",
"Konser öncesi, biletler tükenmiş",
"El sanatları kursuna, katıldım",
"Kültürel mirasımızı, korumalıyız",
"Müze turuna, rehber eşlik etti",
"Klasik müzik dinlemeyi, seviyorum",
"Kültürel değerlerimiz, çok önemli",
"Tarihî bir mekâna, ziyarette bulundum",
"Sanatçıyla söyleşi yapmak, keyifliydi",
"Kültür festivali boyunca, konserler vardı",
"Tiyatro salonu, çok kalabalıktı",
"Kültürel kitaplar, okumayı seviyorum",
"Farklı ülkelerin müziklerini, dinledim",
"Sanat galerisi, yeni sergi açtı",
"Kültürel mirasla ilgili, belgesel izledim",
"Müzik festivali, bu yıl çok popülerdi",
"Tarihî eserleri incelemek, ilginçti",
"Kültür turları, şehri daha iyi tanımamı sağladı",
"Sanat etkinliklerine katılmak, motivasyon veriyor",
"Tiyatro oyunu, akşam seansı için uygundu",
"Kültürel mirasımızı, genç nesillere aktarmalıyız",
"Müze gezisi, çocuklar için eğitici oldu",
"Sanat sergisi, şehrin merkezinde açıldı",
"Kültürel etkinlikler, hafta boyunca sürdü",
"Konser salonunda, farklı sanatçılar sahne aldı",
"Klasik eserleri okumak, ruhu dinlendiriyor",
"Kültürel dernek etkinliklerine, katıldım",
"Sanat ve kültür, şehir hayatını zenginleştiriyor",
"Tarihî mekânlar, fotoğraf çekmek için ideal",
"Müze koleksiyonları, çok çeşitliydi",
"Kültürel eğitim seminerine, katıldım",
"Sanat atölyelerinde, yeni teknikler öğrendim",
"Tiyatro oyunu, çok duygusaldı",
"Kültürel değerlerimizi korumak, önemli",
"Müze rehberi, bilgilerle doluydu",
"Sanat festivali, çok kalabalıktı",
"Kültürel miras için, bağışta bulundum",
"Tiyatro oyunu için biletler, hızla tükendi",
"Kültürel etkinlikler, city turizmini artırıyor",
"Sanat eserleri, duygularımı etkiledi",
"Müze turunda, çok şey öğrendim",
"Kültürel programlar, televizyon ve radyoda yayınlandı",
"Sanatçıyla, atölye çalışması yaptım",
"Kültürel mirasımızı tanıtmak için, proje yaptım",
"Osmanlı İmparatorluğu, uzun yıllar hüküm sürdü",
"Tarih dersinde, Antik Mısır’ı öğrendik",
"Tarihî kitaplar okumak, çok ilginç",
"Müze gezilerinde, tarihî eserler gördüm",
"Cumhuriyetin kuruluş yılı, önemli bir dönüm noktasıdır",
"Tarihî belgeleri incelemek, zaman alıyor",
"Tarihi savaşları, araştırmayı seviyorum",
"Tarihî mekânlar, fotoğraf için idealdir",
"Tarihî olaylar, günümüzü etkiliyor",
"Geçmiş uygarlıkları anlamak, kültürümüzü geliştirir",
"Tarihî binalar, restore edilmelidir",
"Osmanlı padişahlarının hayatını, okudum",
"Tarih dersinde, Avrupa Orta Çağı işlendi",
"Tarihî kalıntılar, turizmi artırıyor",
"Tarihî belgeler, kütüphanelerde saklanıyor",
"Tarihi anıtlar, ziyaretçileri cezbediyor",
"Tarih araştırmaları, öğrenciler için faydalı",
"Tarihî haritalar, eski dönemleri gösteriyor",
"Geçmişteki icatlar, hayatımızı değiştirdi",
"Tarihî olaylar, film ve dizilere konu oluyor",
"Tarihi savaş alanlarını, gezdim",
"Tarihî eserler, korunmalı",
"Tarihî dokümanlar, arşivlerde yer alıyor",
"Tarihî figürlerin hayat hikayelerini, okudum",
"Tarihi dönemleri anlamak, önemli",
"Tarih kitapları, çocuklar için de uygun",
"Tarihî mekanlar, şehir merkezinde bulunuyor",
"Geçmişten ders çıkarmak, gerekiyor",
"Tarihî olaylar, gazetelerde tekrar anlatılıyor",
"Tarihi belgeler, dijitalleştiriliyor",
"Tarihi film, izlemeyi seviyorum",
"Tarihî eserlerin restorasyonu, devam ediyor",
"Tarihi eserleri, koleksiyonuma ekledim",
"Tarih dersinde, savaş ve barış konuları işlendi",
"Tarihi belgeleri okumak, sabır gerektiriyor",
"Tarihî mekanları gezmek, keyifli",
"Tarihi hikâyeler, çok öğretici",
"Tarihî yapılar, fotoğraf çekmek için güzel",
"Tarihi olaylar, müze sergilerinde anlatılıyor",
"Geçmişten bugüne ulaşan eserler, çok değerli",
"Tarihî liderlerin kararlarını, inceledim",
"Tarih derslerinde, kronoloji çok önemlidir",
"Tarihi şehirleri gezmek, kültürü tanıtıyor",
"Tarihî kitaplardan, çok şey öğrendim",
"Tarihi savaşlar, strateji açısından ilginçtir",
"Tarihî belgeler, öğrencilere rehber oluyor",
"Tarih derslerinde, farklı uygarlıklar anlatılıyor",
"Tarihî yerleri ziyaret etmek, unutulmaz",
"Tarihî olayların etkileri, bugün de sürüyor",
"Tarihi eserleri korumak, toplum için gerekli",
"Tarihî mekanlarda, rehber eşliğinde gezdim",
"Tarihî belgeleri, dijital ortamda inceledim", 
"Hafta sonu, İstanbul'u gezmeye gittik",
"Yeni bir şehri keşfetmek, çok heyecan verici",
"Tatilde, deniz kenarında yürüyüş yaptım",
"Müze ve sanat galerilerini gezmeyi, seviyorum",
"Doğa yürüyüşü için, dağlara çıktık",
"Şehir turuna, rehber eşlik etti",
"Gezi sırasında, çok fotoğraf çektim",
"Tarihi mekanları gezmek, keyifliydi",
"Yolculuk için, trenle seyahat ettik",
"Hafta sonu, köy turuna katıldık",
"Gezi planlarını, önceden yaptım",
"Deniz kıyısında, kamp yaptık",
"Yeni lezzetler denemek, geziyi daha eğlenceli yaptı",
"Gezi boyunca, farklı kültürleri gözlemledim",
"Şehirde, bisiklet turu yaptık",
"Turist rehberlerinden, bilgiler aldık",
"Gezi sırasında, yeni arkadaşlar edindim",
"Tatil planımızı, erken yaptık",
"Geziye, kamera ile çıktık",
"Doğa manzaraları, çok etkileyiciydi",
"Gezi sırasında, tarihi köprüleri gördük",
"Şehir meydanında, uzun yürüyüş yaptık",
"Gezi boyunca müzik dinlemek, keyif verdi",
"Yeni gezi rotaları, planlıyorum",
"Dağ köylerini gezmek, farklı bir deneyim",
"Şehir turunda, eski sarayları gördük",
"Gezi sırasında, rehber kitaplarından faydalandık",
"Hafta sonu, sahil kasabasını ziyaret ettik",
"Gezi boyunca, yerel yemekleri tattık",
"Tatil için, hafta sonu planı yaptık",
"Gezi sırasında, tekne turuna katıldık",
"Doğa parklarında, piknik yaptık",
"Şehirde yürüyerek gezi yapmak, eğlenceliydi",
"Gezi sırasında, tarihi camileri ziyaret ettik",
"Tatilde, bisiklet turuna çıktık",
"Gezi planlarını, aileyle yaptık",
"Yeni gezi noktalarını keşfetmek, keyifli",
"Gezi sırasında, çok fotoğraf çektim",
"Şehir merkezini, yürüyerek gezdik",
"Gezi sırasında, tarihi evleri inceledik",
"Hafta sonu, orman yürüyüşüne çıktık",
"Gezi boyunca, kültürel etkinliklere katıldık",
"Doğa yürüyüşü sırasında, kuşları gözlemledik",
"Gezi sırasında, yeni restoranlar denedik",
"Şehir turunda, farklı semtleri gezdik",
"Gezi boyunca, farklı müzeleri ziyaret ettik",
"Tatilde, sahil yürüyüşleri yaptık",
"Gezi sırasında, yerel halkla sohbet ettik",
"Doğa parklarında yürüyüş yapmak, çok keyifliydi",
"Gezi planımızı, hava durumuna göre yaptık",
"Şehir turunda, tarihi meydanları gezdik",
"Gezi boyunca yeni yerler keşfetmek, heyecan verici",
"Yeni bir roman, okumaya başladım",
"Romanın konusu, çok ilginçti",
"Yazar, karakterleri çok iyi işlemiş",
"Romanın sonu, beni şaşırttı",
"Klasik romanları, okumayı seviyorum",
"Bir romanın film uyarlamasını, izledim",
"Romanın dili, akıcı ve anlaşılırdı",
"Tarihi romanlar, okumak keyifli",
"Romanda ana karakterin macerasını, takip ettim",
"Yeni çıkan romanları, kitapçıdan aldım",
"Romanın atmosferi, çok etkileyiciydi",
"Çocuk romanları da, bazen çok eğlenceli oluyor",
"Romanın olay örgüsü, çok sürükleyiciydi",
"Yazarın üslubu, romanı daha güzel kılmış",
"Romanda, aşk ve dostluk temaları işlenmiş",
"Polisiye romanları, okumayı seviyorum",
"Roman karakterleri, çok gerçekçiydi",
"Yeni roman yazarını, keşfettim",
"Romanda, beklenmedik sürprizler vardı",
"Gençlik romanları, bazen nostaljik hissettiriyor",
"Romanın giriş bölümü, çok dikkat çekiciydi",
"Yazar, roman boyunca sürprizlerle dolu bir hikâye anlatmış",
"Romanı okurken, zamanın nasıl geçtiğini anlamadım",
"Romanda, detaylı karakter tasvirleri vardı",
"Yeni roman serisine, başladım",
"Romanın konusu, güncel olaylarla ilgiliydi",
"Romanda, şehir yaşamı ve ilişkiler anlatılmış",
"Klasik romanları yeniden okumak, keyifli oluyor",
"Romanda, karakterlerin iç dünyaları iyi anlatılmış",
"Roman boyunca, merak duygusu hiç kaybolmadı",
"Romanın kapağı, çok ilgi çekiciydi",
"Romanda, macera ve gizem ön plandaydı",
"Roman karakterleriyle, bağ kurdum",
"Yazarın romanındaki dil, çok sade ve etkiliydi",
"Romanın sonunda, beklenmedik bir gelişme oldu",
"Tarihî romanlar, geçmişi anlamama yardımcı oluyor",
"Roman okurken kahve içmek, keyifli oluyor",
"Romanda, farklı bakış açıları vardı",
"Romanın konusu, aile ilişkilerini anlatıyordu",
"Yeni bir roman yazarı, keşfettim",
"Romanda, mizah unsurları vardı",
"Romanın bölümleri, birbirine çok iyi bağlanmıştı",
"Roman karakterleri, birbirinden farklı ve ilginçti",
"Okuduğum roman, çok beğenildi",
"Romanda şehir ve doğa tasvirleri, etkileyiciydi",
"Romanın dili, okuyucuyu içine çekiyordu",
"Romanda duygu ve gerilim dengesi, çok iyiydi",
"Romanı, arkadaşlarıma önerdim",
"Romanda geçmiş ve şimdiki zaman arasında, geçişler vardı",
"Roman karakterlerinin yaşadığı olaylar, düşündürücüydü",
"Romanda, anlatım tarzı çok akıcıydı",
"Roman boyunca, merak duygusunu kaybetmedim",
"Okuduğum roman, hayatımda iz bıraktı",
"Hafta sonu, sinemaya gitmeyi planlıyorum",
"Yeni çıkan filmi, izledim",
"Sinema salonunda koltuklar, çok rahattı",
"Film konusu, çok sürükleyiciydi",
"Yönetmenin üslubu, filme çok şey katmış",
"Sinema biletlerini, internetten aldım",
"Film eleştirilerini okuyarak, tercih yaptım",
"Sinema salonunda, patlamış mısır aldım",
"Filmde görsel efektler, çok başarılıydı",
"Komedi filmleri izlemek, moralimi yükseltiyor",
"Sinema festivali için, bilet aldım",
"Film boyunca, karakterlerle empati kurdum",
"Sinema filmi, çok duygusaldı",
"Film müzikleri, çok etkileyiciydi",
"Romantik filmleri, izlemeyi seviyorum",
"Sinema salonunda film öncesi, reklamlar vardı",
"Filmde aksiyon sahneleri, heyecan vericiydi",
"Sinema filmi, arkadaşlarla izlemek keyifli",
"Belgesel film izlemek, bilgi verici oldu",
"Filmde, oyunculuklar çok başarılıydı",
"Sinema salonu, kalabalıktı",
"Film eleştirmenleri, filmi çok beğendi",
"Film boyunca, gerilim hiç kaybolmadı",
"Sinema salonunda, 3D film izledim",
"Filmde diyaloglar, çok etkileyiciydi",
"Sinema keyfi için, akşam seansı tercih ettim",
"Film çekimleri, farklı mekanlarda yapılmış",
"Sinema filmi sonunda, büyük sürpriz vardı",
"Filmde kullanılan kostümler, çok güzeldi",
"Sinema filmi için, uzun kuyruk vardı",
"Film konusu, gerçek olaylara dayanıyordu",
"Sinema salonunda, büyük ekran keyfi yaşadım",
"Film boyunca, kahkaha attım",
"Sinema filmi çok eleştirildi, ama ben beğendim",
"Filmde dramatik sahneler, çok etkileyiciydi",
"Sinema deneyimi, arkadaşlarla daha eğlenceli",
"Film festivalinde ödül kazanan filmi, izledim",
"Sinema filmi izlerken, notlar aldım",
"Filmde kullanılan ışık ve renkler, çok güzeldi",
"Sinema salonunda, rahat koltukta izledim",
"Film sonunda, karakterler beklediğim gibi gelişti",
"Sinema salonunda, ses sistemi çok iyiydi",
"Film müzikleri, sahneleri güçlendirmişti",
"Sinema filmi izlerken, çok duygulandım",
"Filmdeki karakterler, çok gerçekçiydi",
"Sinema keyfi için, popüler filmi seçtim",
"Film boyunca, sürprizlerle karşılaştım",
"Sinema salonu, büyük ve ferah bir mekândı",
"Film izlerken, kendimi karakterlerin yerine koydum",
"Sinema filmi, oldukça etkileyiciydi",
"Film sonrasında, arkadaşlarla tartıştık",
"Sinema keyfi için, hafta sonunu bekledim",
"Filmdeki olay örgüsü, çok başarılıydı"
]


# --- Preprocessing ---
# kelimeleri indexlere (sayılara) çevir (Tokenizer)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
total_words = len(tokenizer.word_index) + 1 # +1: padding için ekleniyor

print(f"Toplam kelime sayısı: {total_words}")
# n-gram dizileri oluştur -> her cümleden kısa diziler oluşturur (embedding)
# 3-gram: kelimeleri indexlere (sayılar) cevir -> ["kelimeleri indexlere (sayılar)", "indexlere (sayılar) çevir"]

input_sequences = []
for line in data:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

print(f"input_sequences: \n{input_sequences}")
""" (ngram dizileri)

[13, 64], [13, 64, 378], [13, 64, 378, 88], [13, 64, 378, 88, 4], [13, 64, 378, 88, 4, 1], [13, 64, 378, 88, 4, 1, 89]

"Bugün(13) hava(64) top(378) oynamak(88) için(4) çok(1) güzel(89)"

"""

# padding: farklı uzunluktaki dizileri sabitle

max_sequence_length = max(len(x) for x in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')


print(f"after padding input_sequences: \n{input_sequences}")

"""
[13, 64], [13, 64, 378], [13, 64, 378, 88], [13, 64, 378, 88, 4], [13, 64, 378, 88, 4, 1], [13, 64, 378, 88, 4, 1, 89]

padding işlemi, dizilerin başına 0 ekliyor ve tüm dizileri eşit uzunlukta tutuyor.

[  0   0   0 ...   0  13  64]
[  0   0   0 ...  13  64 378]
[  0   0   0 ...  64 378  88]

"""


# girdi (X) ve hedef değişlenler (y) ayır
X = input_sequences[:, :-1] # n-1 kelimeyi giriş olarak seç
y = input_sequences[:, -1] # n inci kelimeyi tahmin et
"""

[  0   0   0 ...   0  13  64]
X = [  0   0   0 ...   0  13]
Y = [64] # hedef değişken (tahmin işlemi)

"""

# hedef değişkene one hot encoding
y= tf.keras.utils.to_categorical(y, num_classes=total_words)
""" One hot encoding
[1,2,3] -> 
1 -> [1,0,0]
2 -> [0,1,0]
3 -> [0,0,1]
"""

print(f"Hedef değişken: {y}")

# -- LSTM Training süreci başlıyor ---
# lstm modeli tanımla
model = Sequential()
model.add(Embedding(total_words, 50, input_length=X.shape[1]))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax')) # output
"""

X = [bugün hava çok]
y = [güzel]

"""

# compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # sparse_categorical_accuracy// adaptive momentum optimizer (parametrelerimizi güncelleyen optimizerimiz), categorical_crossentropy loss fonksiyonumuz, metric'imiz accuracy

print(model.summary())


# eğitimi başla
model.fit(X, y, epochs=100, verbose=1) # verboser=1 -> console'da göster, epochs sayısı -> tüm verilerin kaç kez train edileceği, X -> bağımsız değişkenlerimiz, y -> target/bağımlı değişkenlerimiz


# örnek eğitim testi (metin üretimi)
def generate_text(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0] # tokenization
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre') # padding
        predicted_probs=model.predict(token_list,verbose=1)
        predicted_index=np.argmax(predicted_probs,axis=-1)[0]
        predicted_word=tokenizer.index_word[predicted_index]
        seed_text += " " + predicted_word
    return seed_text

print(generate_text("Bugün",5))

"""

(1)
seed_text = bu sabah
predicted_word = okula

(2)
seed_text = bu sabah okula
predicted_word = geç

(return)
seed_text = bu sabah okula geç

"""


               