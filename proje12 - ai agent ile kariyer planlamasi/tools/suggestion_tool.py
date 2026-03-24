from ddgs import DDGS
from httpx import get

class SuggestionTool:
    def __init__(self):
        self.ddgs = DDGS()
    
    def search_rescources(self, query,max_results=5):
        results = self.ddgs.text(query, max_results=max_results)
        
        # sonuçlardan alınan başlıkları bağlantı bilgilerini ve özet bilgileri tutmak için bir liste oluştur
        suggestions=[]
        
        for result in results:
            suggestions.append({
                "title":result.get("title"),
                "link":result.get("href"),
                "snippet":result.get("body") # kısa açıklama
            })
        
        return suggestions

a = SuggestionTool()
#response=a.search_rescources("python developer")
#print(response)