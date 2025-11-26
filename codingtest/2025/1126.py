import numpy as np
# 텍스트 읽고 처리하기
with open('1268-0.txt', 'r', encoding='UTF8') as fp:
    text = fp.read()
start_indx = text.find('THE MYSTERIOUS ISLAND')
end_indx = text.find('End of the Project Gutenberg')
text = text[start_indx:end_indx]
char_set = set(text)
print('total length : ', len(text))
print('고유한 문자 : ', len(char_set))