import requests
import os
from datetime import datetime, timedelta

class telegram:

    def send_message(self, message):
        api_token = os.getenv('API_TOKEN')
        chat_id = os.getenv('CHAT_ID')
        api_url = f'https://api.telegram.org/bot{api_token}/sendMessage'

        try:
            response = requests.post(api_url, json={'chat_id': chat_id, 'text': message, 'parse_mode': 'HTML'})
            k = 10
        except Exception as e:
            print(e)

    def send_formatted_message(self, stock, current_price, threshold, sell_perc, hold_till, stop_perc):
        
        today = datetime.now()

        message = "<u><b>BUY SIGNAL</b></u>\n" \
            + f"Stock: {stock} (<a href=\"http://charts.aastocks.com/servlet/Charts?fontsize=12&15MinDelay=T&lang=1&titlestyle=1&vol=1&Indicator=9&indpara1=20&indpara2=2&indpara3=0&indpara4=0&indpara5=0&subChart1=2&ref1para1=14&ref1para2=0&ref1para3=0&subChart2=3&ref2para1=12&ref2para2=26&ref2para3=9&subChart3=12&ref3para1=0&ref3para2=0&ref3para3=0&scheme=3&com=100&chartwidth=870&chartheight=855&stockid=00{stock}&period=6&type=1&logoStyle=1\">AAStock Chart</a>) \n" \
            + f"Current Price: ${round(current_price,2)} \n" \
            + f"Take Profit at: ${round(current_price * (1 + sell_perc), 2)} (+{sell_perc * 100}%) \n" \
            + f"Stop at: ${round(current_price * (1 - stop_perc), 2)} (-{stop_perc * 100}%) \n" \
            + f"Hold till: {(today + timedelta(hold_till)).strftime('%Y-%m-%d')} ({hold_till} days)\n" \
            + f"Confidence: {threshold}"

        self.send_message(message)
