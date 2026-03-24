from langchain import messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json
from dotenv import load_dotenv 

load_dotenv()
class CareerGoalAgent:
    # constructor
    def __init__(self, model_name="gpt-4.1-nano"):
        self.llm = ChatOpenAI(
            model=model_name, 
            temperature=0.5)

    # kullanıcıdan geken hedef meslek bilgisine göre kariyer yol haritası oluşturma
    def ask_career_goal(self, user_input):
        messages = [
            SystemMessage(content= (
                "Sen bir kariyer planlama asistanısın. Kullanıcı bir meslek söylediğinde, "
                "bu mesleğe ulaşmak için adım adım bir yol haritası üret. "
                "Sonuçları *sadece* aşağıdaki formatta JSON olarak döndür.\n\n"
                "{\n \"adimlar\": [\"...\", \"...\", ...]\n}"
            )),
            HumanMessage(content=f"Hedef mesleğin: {user_input}.")
        ]
        response = self.llm.invoke(messages)
        return self.parse_response(response.content)
    
    def parse_response(self, response_text):
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format"}
        except Exception as e:
            return {"error": str(e)}
        
a = CareerGoalAgent()
# print(a.ask_career_goal(user_input="python developer"))