"""
usd to tr

"""

 

from langchain.tools import tool # bu dekoratör sayesinde fonksiyonumuz bir langchain aracı olarak tanımlanabilecek 
import requests # http istekleri

@tool # @tool sayesinde convert_usd_to_try fonksiyonu langchain ajanları tarafından kullanılabilinecek bir araç olduğu belirtilir
def convert_usd_to_try(amount: float) -> str:
    """
    Convert USD (US Dollar) to TRY (Turkish Lira) using current exchange rate.
    
    This tool fetches the real-time USD/TRY exchange rate from CoinGecko API
    and calculates the equivalent amount in Turkish Lira.
    
    Args:
        amount (float): The amount in USD to convert. Can be a number or string 
                        containing numbers (e.g., "100", "$100", "100.50 USD").
    
    Returns:
        str: A formatted string showing the conversion result with the exchange rate.
             Example: "100 USD is equal to 4000.00 TRY (Kur: 40.00)"
    
    Example:
        >>> convert_usd_to_try(100)
        "100 USD is equal to 4000.00 TRY (Kur: 40.00)"
    """
    try:
        if isinstance(amount, str): 
            amount = float("".join(filter(lambda c: c.isdigit() or c == ".", amount)))# rakamlar ve noktalar kalır, diğer karakterler silinir ve kalan numara float'a çevrilir
        
        # api den usd/try çevrimini yap
        url = "https://api.coingecko.com/api/v3/simple/price?ids=usd&vs_currencies=try"
        response = requests.get(url)
        
        if response.status_code != 200:
            return f"Error converting USD to TRY. Status code: {response.status_code}"
        
        data = response.json()
        
        rate = data["usd"]["try"]
        
        # kullanıcının verdiği amount ile döviz kurunu çarpalım
        result =amount*rate
        
        # "100 USD is equal to 4000 TRy (Kur:40)"
        return f"{amount} USD is equal to {result:.2f} TRY (Kur: {rate:.2f})"
    except Exception as e:
        return f"Error converting USD to TRY. Error: {e}"
    
if __name__ == "__main__":
    # test tutarı 100 dolar
    test_amount = 100
    
    print(f"{test_amount} USD -> TRY")
    print(convert_usd_to_try.run({"amount":test_amount}))
        
        
                           
            