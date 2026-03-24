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

import time # geçek zamanlı beklemeler
from datetime import date, datetime,timedelta
from pdf_reader import extract_tasks_from_pdf
from rich import print
from gpt_agent import generate_followaup_question, is_task_completed

task_memory={}

# simülasyon fonksiyonu
def run_gpt_scheduler(pdf_path="FitMiniApp_Proje_Dokumani.pdf", delay_sec=10):
    tasks=extract_tasks_from_pdf(pdf_path)
    
    # simülasyon başlangıç zamanı
    sim_time=datetime(2025,8,25,11,59) 
    
    print(f"[bold green] Smülasyon başladı[/bold green] -> Başlangıç: {sim_time.strftime('%d.%m.%Y %H:%M')}")
    
    while True: # Her döngüde simülasyon 1 dk ilerletilir
        sim_time += timedelta(minutes=1)
        sim_time_str = sim_time.strftime("%H:%M")
        print(f"\n[bold white on black] Saat: {sim_time_str}[/bold white on black]")    
        for task in tasks:
            ts=task["timestamp"] # görevin zamanı
            kisi=task["person"]
            gorev=task["task"]
            key=f"{ts}_{kisi}"
            print(ts)
            print(sim_time)
            if ts <= sim_time:
                onceki_cevaplar = task_memory.get(key, [])
                if onceki_cevaplar:
                    tamam_durumu=is_task_completed(kisi, gorev, onceki_cevaplar, sim_time)
                    if tamam_durumu == "tamamlandı":
                        continue # görev tamamlanmış tekrar soru sorma
                    else:
                        print(f"[yellow]{kisi} görevini henüz tamamlamadı. Tekrar sorulacak...[/yellow]")
                
                soru = generate_followaup_question(
                    person=kisi,
                    task=gorev,
                    current_time=sim_time_str,
                    previous_responses=onceki_cevaplar
                )
                
                print(f"[bold red]{kisi}[/bold red] kişisine sorulan soru: {soru}")
                print(f"[bold blue]{soru}[/bold blue]")
                
                cevap = input("Cevap: ").strip()
                
                task_memory.setdefault(key, []).append({
                    "time": sim_time.strftime("%H:%M"), 
                    "response": cevap
                })
    time.sleep(delay_sec)
                
if __name__ == "__main__":
    run_gpt_scheduler()
                

