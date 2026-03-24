import os 
from openai import OpenAI
from dotenv import load_dotenv 

load_dotenv()

client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_followaup_question(person, task, current_time, previous_responses=None):
    """
    Gpt ye kişi, görev, zaman ve geçmiş yanıtları vererek en uygun soruyu üretmesini ister
    
    person: çalışan ismi
    task: görev
    current_time: şuanki zaman
    
    previous_responses: kişinin geçmiş yanıtları
    """
    
    history=[]
    if previous_responses:
        for response in previous_responses:
            history += f"Saat {response["time"]}: {response["response"]}\n" # formatlı şekilde history oluştur.
    
    prompt=f"""
            Şu anda saat {current_time}
            Sen bir proje yöneticisisin
            
            Görev: "{task}"
            Kişi: {person}
            Bu kişiye bu görev verildi.
            Şimdiye kadar verdiği cevaplar:
            {history if history else "Henüz cevap yok."}
            
            Lütfen {person}'a doğrudan hitab ederek görevle ilgili ne durumda olduğunu soran net ve kısa bir soru yaz.
            
            Soru şunları içermeli:
            - Kişinin ismi ile hitab et
            - Görevin ne olduğu açıkça tekrar geçsin
            - Görevin tamamlanma durumu yada üzerinde çalışılıp çalışılmadığı sorgulansın
            - Sadece doğrudan bir soru cümlesi döndür, başka açıklama yazma.
            
            """
            
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "You are a project manager."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
        
        
    return response.choices[0].message.content

def is_task_completed(person, task, responses, current_time):
    """
    Gpt 'ye görevin tamamlanıp tamamlanmadığını sorar
    Yalnızca üç cevaptan birini return ederiz, "tamamlandı", "devam ediyor", "yapılmadı" şeklinde. 
    """
    history=""
    for response in responses:
        history += f"Saat {response['time']}: {response['response']}\n"
    
    prompt=f"""
            Şu anda saat {current_time}
            Sen bir proje yöneticisisin
            
            Görev: "{task}"
            Kişi: {person}
            Bu kişiye bu görev verildi.
            Şimdiye kadar verdiği cevaplar:
            Bu görevle ilgili şimdiye kadar {person} tarafından verilen cevaplar:
            {history if history else "Henüz cevap yok."}
            
            Lütfen sadece tek bir kelime ile cevap ver:
            - tamamlandı
            - devam ediyor
            - yapılmadı
            
            Talnızca b u üç kelimeden birini döndür. Açıklama yapma.
        """
        
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role":"user", "content":prompt}],
        temperature=0
    )        

    return response.choices[0].message.content
        
if __name__ == "__main__":
    example_history =[
        {"time": "10:00", "response": "Başladım ama eksik bir şeyler var"}
    ]
 
    soru = generate_followaup_question(
        person="Elif",
        task="fast api ile hello world end point yazılması",
        current_time="11:00",
        previous_responses=example_history    
    )

    print(f"Gpt'nin oluşturduğu soru: \n {soru}") 
  
    soru = is_task_completed(
        person="Elif",
        task="fast api ile hello world end point yazılması",
        responses=[{"time": "12:00", "response": "Bitti"}],
        current_time="13:00"
    
    )
    
    print(f"Gpt'nin oluşturduğu soru: \n {soru}")