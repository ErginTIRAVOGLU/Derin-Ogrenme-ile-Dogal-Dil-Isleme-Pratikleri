from asyncio import tasks

from PyPDF2 import PdfReader # pdf dosyasını okumak ve içeriğini çıkarmak
import re # regular expression (düzenli ifadeler) metin içerisinde desen arama
from datetime import datetime # tarhi ve saatler için datetime modülü

def extract_tasks_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in pdf_reader.pages]) # pdf içerisindeki her sayfayı çıkartır
    
    # örnek desen "12.07.2025 14:30 - Kaan:"
    pattern =r"(\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}) - (.*?): (.*)" # tarih (gg.aa.yyyy ss:dd), kişi adı ve görev açıklaması gruplar
    matches=re.findall(pattern, text)
    
    tasks = []
    for match in matches:
        tarih_str, kisi, gorev = match # eslesme 3 parçaya ayrılıyor tarih, kişi adı görev
        tarih = datetime.strptime(tarih_str, "%d.%m.%Y %H:%M")
        tasks.append({
            "timestamp": tarih,
            "person": kisi.strip(), # kisi ado, çalışan
            "task": gorev.strip() # görev açıklaması
        })
    return tasks

if __name__ == "__main__":
    path = "FitMiniApp_Proje_Dokumani.pdf"
    