import requests
from BeautifulSoup import BeautifulSoup

url = "https://www.amazon.com/Regretting-You-Colleen-Hoover-ebook/dp/B07SH5V2NB/ref=zg_bs_g_digital-text_d_sccl_1/140-5556349-9397568?psc=1"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36"
}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

print(soup)

books = soup.select('.zg-item-immersion')
for book in books:
    title = book.select_one('.p13n-sc-truncate').get_text(strip=True) if book.select_one('.p13n-sc-truncate') else "No title"
    rank = book.select_one('.zg-badge-text').get_text(strip=True) if book.select_one('.zg-badge-text') else "No rank"
    print(rank, title)
