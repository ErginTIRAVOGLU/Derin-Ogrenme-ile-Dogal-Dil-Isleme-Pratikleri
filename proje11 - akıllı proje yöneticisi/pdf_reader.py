from PyPDF2 import PdfReader
import re
from datetime import datetime

def extract_tasks_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])

    # sadece ilgili bölüm
    start = text.find("4. ZAMAN BAZLI GÖREV TAKVİMİ")
    end = text.find("Tablo 3")
    if start != -1 and end != -1:
        text = text[start:end]

    # görevleri yakala
    pattern = r"\d+\s+(\d{2}:\d{2}-\d{2}:\d{2})\s+([A-Za-zğüşıöçĞÜŞİÖÇ,\s]+)\n(.+?)(?=\n\d+\s+\d{2}:\d{2}-\d{2}:\d{2}|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)

    tasks = []

    for time_range, kisi, gorev in matches:

        lines = gorev.split("\n")
        clean_lines = []

        for line in lines:
            line = line.strip()

            if not line:
                continue

            # ❌ kişi bildirimi (Elif: ...)
            if re.match(r"^[A-ZÇĞİÖŞÜ][a-zçğıöşü]+:", line):
                continue

            # ❌ gereksiz sistem mesajları
            if any(x in line for x in ["Sprint", "kontrol", "başla!", "tamamla!"]):
                continue

            clean_lines.append(line)

        clean_task = " ".join(clean_lines)
        clean_task = re.sub(r"\s+", " ", clean_task).strip()
        
        saat_baslangic = time_range.split('-')[0]  # "14:00"
        saat, dakika = map(int, saat_baslangic.split(':'))
        timestamp = datetime(2025, 8, 25, saat, dakika)  # Simülasyon tarihi ile aynı
        
        tasks.append({
            "timestamp": timestamp,
            "person": kisi.strip(),
            "task": clean_task
        })

    return tasks


if __name__ == "__main__":
    path = "FitMiniApp_Proje_Dokumani.pdf"

    tasks = extract_tasks_from_pdf(path)

    for task in tasks:
        print(f"{task['timestamp']} - {task['person']}: {task['task']}")