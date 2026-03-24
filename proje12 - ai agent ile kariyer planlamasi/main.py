"""
problem tanımı: Kariyer Planlayıcısı
    - insanlar kariyerleri ile ilgili nasıl bir yol izlemeleri gerektiğini sorabilecekler

araçlar ve teknolojiler:
    - langchain, openai, duckduckgo

plan/program:
    - Hedef belirleme
        - kullanıcı terminalde meslek hedefini belirler
        - gpt buna uygun adımları json formatında üretir
    - Yol haritası planlama
        - görevler haftalara bölünür
        - her hafta için net hedefler belirlenir
    - Kaynak önerisi
        - kullanıcı terminal üzerinde istediği konuyu yazar
        - sistem web üzerinden önerileri getirir (video, kurs ...)
    - Gelişim takibi
        - kullanıcıdan ilerleme bilgisi alınır ve hafızada görev başarı durumu saklanır

install libraries: freeze



"""

from agents.career_goal_agent import CareerGoalAgent
from agents.task_scheduler_aget import TaskSchedulerAgent
from tools.suggestion_tool import SuggestionTool
from memory.user_memory import UserMemory
from dotenv import load_dotenv
import json

load_dotenv()

if __name__ == "__main__":
    print("Kariyer Planlayıcısı başlatılıyor...")
    
    user_memory= UserMemory()
    
    goal = input("Hedef mesleğiniz nedir?")
    user_memory.update_goal(goal)
    
    goal_agent=CareerGoalAgent()
    roadmap=goal_agent.ask_career_goal(goal)
    
    print(f"Yol Haritanız: \n")
    print(json.dumps(roadmap, indent=2, ensure_ascii=False))
    
    num_week=4
    scheduler=TaskSchedulerAgent(weeks=num_week)
    schedule =scheduler.create_schedule(roadmap)
    scheduler.save_schedule(schedule)
    
    print(f"\n {num_week} haftalık planınız: \n")
    print(json.dumps(schedule, indent=2, ensure_ascii=False))
    
    suggestor=SuggestionTool()
    topic= input("Bir beceri başlığı giriniz:")
    results=suggestor.search_rescources(topic)
    print("Kaynak önerileriniz: \n")
    # print(results)
    for result in results:
        print(f" - {result["title"]} \n {result["link"]} \n {result["snippet"]}\n")
    
    
