from konlpy.tag import Okt
import pandas as pd
import fitz
import re


pdf_document = fitz.open('./data/news.pdf')
num_pages = pdf_document.page_count
print(f'문서의 페이지 수: {num_pages} 페이지입니다.')

metadata = pdf_document.metadata
print('메타데이터:', metadata)

page_number = 0
page = pdf_document.load_page(page_number)

# 텍스트 추출
text = page.get_text()


# 데이터를 확인 해보니 띄어쓰기가 없고 이상하게 줄 바꿈도 되어 있음
print(text)

# 데이터 정제하기
clean_text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\s]', '', text)
clean_text = re.sub(r'\s+', '', clean_text)
# 띄어쓰기 싹 제거

print(clean_text)

# 띄어쓰기 복원
from pykospacing import Spacing
spacing = Spacing()
new_sent = spacing(clean_text)

print(new_sent)
# 데이터 정제 완료

okt = Okt()

# 형태소 분리
morphs = okt.morphs(new_sent)

# 품사
pos = okt.pos(new_sent)

# 명사만 뽑자
nouns = okt.nouns(new_sent)

print(morphs)
print('===============================================')
print(pos)
print('===============================================')
print(nouns)

# 일단 임의로 불용어 설정
stopwords = ['것', '수', '있', '하', '되', '이', '그', '및', '등', '를', '은', '는', '이', '가']

filtered_words = [word for word in nouns if word not in stopwords and len(word) > 1]

print(filtered_words)

# 단위빈도 출력
from collections import Counter

word_freq = Counter(filtered_words)
print(word_freq)


# torch tensor 변환
import torch

freqs = [count for word, count in word_freq.most_common(20)]
tensor = torch.tensor(freqs, dtype=torch.float32)
print(tensor)

# 워드클라우드 생성
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wc = WordCloud(
    font_path='./fonts/malgun.ttf',
    background_color='white',
    width=800,
    height=400
)

wc.generate_from_frequencies(dict(word_freq))

plt.figure(figsize=(10, 10))
plt.imshow(wc)
plt.axis('off')
plt.show()

import matplotlib.font_manager as fm

font_path = './fonts/malgunsl.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['axes.unicode_minus'] = False

# 상위 20개 시각화?
top20 = word_freq.most_common(20)
words, counts = zip(*top20)

plt.figure(figsize=(10, 10))
plt.barh(words, counts)
plt.xticks(rotation=45)
plt.title('Top 20 words')
plt.tight_layout()
plt.show()