# DEMOVERSION OF IMAGE TRANSLATION MADE USING STREAMLIT

'''
  To run in google colab:

  upload translate_image.py into files of colab
  
  install libraries necessary for this app 
  !pip install easyocr
  !pip install googletrans==3.1.0a0
  !pip install chinese
    
  mount google drive to google colab, because fonts of the text for text_inpainting are saved on google drive
  from google.colab import drive
  drive.mount('/content/drive/')
  (or upload them into files of colab and change font_path)

  !pip install streamlit
  !npm install localtunnel
  !streamlit run translate_image.py & npx localtunnel --p 8501

  follow the link written in output after words 'your url is: '
  Enter IP, given in output in External URL.

'''
# import os
# import pytesseract
# import cv2
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from torch import index_add
# from tqdm import tqdm
# from google.colab.patches import cv2_imshow

import streamlit as st
import io
import easyocr
from googletrans import Translator
from chinese import ChineseAnalyzer
from PIL import Image, ImageDraw, ImageFont

try: 
    from PIL import Image
except ImportError:
    import Image


def text_wrap_by_width(text, font, max_width, trans_lang, analyzer=None):
    """Wrap text basing on specified width. 
        text: text to wrap
        font: font of the text
        max_width: width of the bounding box to split the text with
        trans_lang: от языка зависит, как разбивать текст на слова
        analyzer:  для разделения китайского текста на слова
        return: list of sub-strings """

    lines = []
    
    # If the text width is smaller than the image width, then no need to split
    # just add it to the line list and return
    # font.getsize(text)[0] - width, font.getsize(text)[1] - height

    if font.getsize(text)[0] <= max_width:
        lines.append(text)
    else:
        #split the line by spaces to get words
        if trans_lang == 'zh-cn':
          words = analyzer.parse(text).tokens()
        else:
          words = text.split(' ')   # список слов

        # print('words: ', words)
        i = 0   # счетчик слов
        # append every word to a line while its width is shorter than the image width
        while i < len(words):
            # print(' len(words) ',  len(words))
            line = ''
            while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                if trans_lang == 'zh-cn':  # если язык китайский, то не нужно добавлять пробелы между словами
                  line = line + words[i]
                else:
                  line = line + words[i]+ " "
                i += 1
            if not line:
                line = words[i]
                i += 1
            lines.append(line)
    return lines


def text_wrap_by_h_w(font_path, text, max_width, max_height, trans_lang):
  ''' Define suitable size of the font, considering given font, height and width of the bounding box '''

  # т.к. в китайском нет пробелов, то разделение текста по пробелам не сработает
  if trans_lang == 'zh-cn':  # если язык, на который переведен текст, - китайский
    analyzer = ChineseAnalyzer()   # то инициализируем analyzer, который поможет разбить текст на слова
    # выполним инициализацию в этой функции, до вызова функции text_wrap_by_width, т.к. инициализация занимает много времени
    # и если делать это в text_wrap_by_width, которая вызывается много раз, то это будет занимать достаточно много времени
  else: 
    analyzer = None
    
  fontsize = 5
  font = ImageFont.truetype(font=font_path, size=fontsize)  # font of the text
  line_height = font.getsize('hg')[1]    # значение высоты строки (измеряем высоту букв h,g)
  wrapped_text = text_wrap_by_width(text, font, max_width, trans_lang, analyzer)
  
  while (line_height * len(wrapped_text)) <= max_height:
    fontsize += 1
    font = ImageFont.truetype(font=font_path, size=fontsize)  # font of the text
    wrapped_text = text_wrap_by_width(text, font, max_width, trans_lang, analyzer)
    line_height = font.getsize('hg')[1]

    if (line_height * len(wrapped_text)) > max_height:
      fontsize -= 1
      font = ImageFont.truetype(font=font_path, size=fontsize)  # font of the text
      wrapped_text = text_wrap_by_width(text, font, max_width, trans_lang, analyzer)
      line_height = font.getsize('hg')[1]
      break

  return wrapped_text, font, line_height


def get_average_color(image, elem):
  '''Get average color on the borders of bounding boxes'''

  sum_all_borders = [0,0,0]
  number_of_pixels =  2*(elem[0][2][0] - elem[0][0][0]) + 2*(elem[0][2][1] - elem[0][0][1])

  y = elem[0][0][1]-2 if elem[0][0][1]-2 > 0 else 1   # проверка, не выходит ли за границы изображения
  for x in range(elem[0][0][0], elem[0][2][0]):
    for i in range(3):
      sum_all_borders[i] += image.getpixel((x, y))[i]  # get the sum of pixels above the top border of bounding box

  y = elem[0][2][1]+2 if elem[0][2][1]+2 < image.size[1] else image.size[1]-1
  for x in range(elem[0][0][0], elem[0][2][0]):
    for i in range(3):
      # print(elem[0][2][1], image.size[1], image.size[0], y)
      sum_all_borders[i] += image.getpixel((x, y))[i]  # get the sum of pixels under the bottom border of bounding box

  x = elem[0][0][0]-2 if elem[0][0][0]-2 > 0 else 1
  for y in range(elem[0][0][1], elem[0][2][1]): 
    for i in range(3):
      sum_all_borders[i] += image.getpixel((x, y))[i]   # get the sum of pixels on the left border of bounding box

  x = elem[0][2][0]+2 if elem[0][2][0]+2 < image.size[0] else image.size[0]-1
  for y in range(elem[0][0][1], elem[0][2][1]): 
    for i in range(3):
      sum_all_borders[i] += image.getpixel((x, y))[i]  # get the sum of pixels on the right border of bounding box 

  average_color = tuple([int(elem/number_of_pixels) for elem in sum_all_borders])

  return average_color


def text_inpainting(image_bytes, font_path, text_from_image, translated_text, source_lang, trans_lang):
  
  # image_name = image_path.split('/')[-1]  # split by '/' and take the last element of the list, which is the name of the image

  # image = Image.open(fp=image_path)
  image = Image.open(io.BytesIO(image_bytes))
  img_draw = ImageDraw.Draw(im=image)

  # img_draw.rectangle([(0, 0), (image.size[0]-0.5, image.size[1]-0.5)], fill ="white", outline ="red")  # boundaries of the image
  # img_draw.rectangle([(0, 0), (image.size[0]-0.5, image.size[1]-0.5)], outline ="red")

  for i, elem in enumerate(text_from_image): 
    
    # print('\n', elem[0], translated_text[i].text, '\n')

    text = translated_text[i].text
    
    # elem[0] - [[x_left, y_up], [x_right, y_up], [x_right, y_down], [x_left, y_down]]; 
    # elem[0][2] - [x_right, y_down], elem[0][0] - [x_left, y_up];     
    bb_width = elem[0][2][0] - elem[0][0][0]    # elem[0][2][0] - elem[0][0][0] = x_right - x_left
    bb_height = elem[0][2][1] - elem[0][0][1]    # elem[0][2][1] - elem[0][0][1] = y_down - y_up

    lines, font, line_height = text_wrap_by_h_w(font_path, text, bb_width, bb_height, trans_lang)

    x = elem[0][0][0]
    y = elem[0][0][1]

    # img_draw.rectangle([tuple(elem[0][0]), tuple(elem[0][2])], fill ="white", outline ="green")   # boundaries of the text in image
    
    background_color = get_average_color(image, elem)
    img_draw.rectangle([tuple(elem[0][0]), tuple(elem[0][2])], fill = background_color)
    

    # Define color of the text in contrast with background color
    if background_color > (127, 127, 127):   # если ближе к белому
      text_color = (background_color[0]-110, background_color[1]-110, background_color[2]-110)
    else:   # если ближе к темному
      text_color = (background_color[0]+100, background_color[1]+100, background_color[2]+100)

    # (0,0,0) - black
    # (255, 255, 255) - white
    for line in lines:
        img_draw.text((x,y), line, fill=text_color, font=font)
        y = y + line_height    # update y-axis for new line 

  # image.show()
  if source_lang == 'zh-cn':  # т.к. папка называется ch_sim, если оставить source_lang = 'zh-cn', путь '/content/drive/MyDrive/Универ/ocr/images/{source_lang}' не будет существовать
   source_lang = 'ch_sim'
  # image_name[image_name.index("."):] - расширение файла после точки
  # image_name[:image_name.index(".")] - имя файла без расширения
  # new_image_name = f'{image_name[:image_name.index(".")]}_{source_lang}-{trans_lang}_translated{image_name[image_name.index("."):]}'
  # image.save(f'/content/drive/MyDrive/Универ/ocr/images/{source_lang}/{source_lang}_translated/{new_image_name}')
                                                                 # {source_lang}-{trans_lang}_translated_{image_name}')
  # image.save(new_image_name)
  return image


def ocr(image_bytes, source_lang):
  # будет распознавать английский и другой переданный язык (поддерживает распознавание нескольких языков одновременно, английский совместим со всеми)
  reader = easyocr.Reader(['en', source_lang], gpu=True)  
  text_from_image = reader.readtext(image_bytes, detail = 1, paragraph = True, 
                                    # width_ths = 1.8
                                    )
  return text_from_image


def text_translation(text_from_image, source_lang, trans_lang):

  translator = Translator()

  translated_text= []
  for elem in text_from_image:
    translated_text.append(translator.translate(elem[1], src=source_lang, dest=trans_lang))  # 'zh-cn': 'chinese (simplified)', 'de': 'german', 'en': 'english', 'ru': 'russian'

  return translated_text


def image_translation(image_bytes, source_lang, trans_lang):
  
  text_from_image = ocr(image_bytes, source_lang)
  # print(text_from_image)
  print('OCR is completed!')

  if source_lang == 'ch_sim':
    source_lang = 'zh-cn'
  
  translated_text = text_translation(text_from_image, source_lang, trans_lang)
  
  # for trans in translated_text:
    # print(trans.origin)
    # print(trans.text)

  print('Text translation is completed!')

  # Save recognized text and its translation
  # txt_name = f'{image_path[:image_path.rindex("/")]}/txt/{image_path.split("/")[-1]}.txt'
  # with open(txt_name, 'a') as f:

  #   if os.path.getsize(txt_name) == 0:  # если изображение переводится первый раз (файл пустой), то записываем все
  #     f.write('Распознанный текст: \n')

  #     # создание списка строк, т.к. f.write должен принимать на вход строку; .join объединяет в строку только список строк, а text_from_image - массив массивов
  #     strings = [str(elem) for elem in text_from_image]  
  #     f.write('\n'.join(strings))   # сохраним распознанный текст
      
  #     f.write('\n\nПеревод: \n')
  #     f.write(f'{source_lang} - {trans_lang}\n')  # с какого языка на какой выполнен перевод
      
  #     # создание списка строк, т.к. f.write должен принимать на вход строку; .join объединяет в строку только список строк
  #     # а translated_text - массив объектов <googletrans.models.Translated>
  #     trans_text = [trans.text for trans in translated_text]
  #     f.write('\n'.join(trans_text))   # сохраним перевод распознанного текста

  #   else:  # если в файл уже записан какой-то перевод, то просто добавляем новый

  #     f.write(f'\n\n{source_lang} - {trans_lang}\n')  # с какого языка на какой выполнен перевод
      
  #     # создание списка строк, т.к. f.write должен принимать на вход строку; .join объединяет в строку только список строк
  #     # а translated_text - массив объектов <googletrans.models.Translated>
  #     trans_text = [trans.text for trans in translated_text]
  #     f.write('\n'.join(trans_text))   # сохраним перевод распознанного текста


  # font_path = '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'
  font_path = '/content/drive/MyDrive/Универ/ocr/TIMES.TTF'
  if trans_lang == 'zh-cn':
    font_path = '/content/drive/MyDrive/Универ/ocr/SIMSUN.TTC'   # шрифт для китайского языка


  image = text_inpainting(image_bytes, font_path, text_from_image, translated_text, source_lang, trans_lang)
  print('Text inpainting is completed!')
  return image


#STREAMLIT

st.title('Demoversion of Image Translation')

source_lang = st.selectbox(
    'Original language',
    ('Russian', 'English', 'German', 'Chinese'))

# st.write('You selected:', source_lang)

trans_languages = st.multiselect(
    'Language to translate',
    ['Russian', 'English', 'German', 'Chinese'])

# st.write('You selected:', trans_languages)

source_langs_to_code = {'Russian': 'ru', 'English': 'en', 'German': 'de', 'Chinese': 'ch_sim'}
trans_langs_to_code = {'Russian': 'ru', 'English': 'en', 'German': 'de', 'Chinese': 'zh-cn'}

source_lang = source_langs_to_code[source_lang]
trans_languages = [trans_langs_to_code[trans_lang] for trans_lang in trans_languages]

# list out keys and values
source_key_list = list(source_langs_to_code.keys())
source_val_list = list(source_langs_to_code.values())

trans_key_list = list(trans_langs_to_code.keys())
trans_val_list = list(trans_langs_to_code.values())


# Uploading the image to the page
uploaded_file = st.file_uploader(label="Upload image", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:

    # Displaying the original image
    st.write('Original image: ')
    image = Image.open(uploaded_file)
    st.image(image)  

    image_bytes = uploaded_file.getvalue()
    
    for trans_lang in sorted(trans_languages):   
      
      translated_image = image_translation(image_bytes, source_lang, trans_lang)
      
      trans_index = trans_val_list.index(trans_lang)
      st.write('Traslated to ', trans_key_list[trans_index])
      
      # Displaying the image with translated text
      st.image(translated_image)  

      source_index = source_val_list.index(source_lang)
      
      # converting PIL image to bytes
      bytes = io.BytesIO()
      translated_image.save(bytes, format="JPEG")
      byte_im = bytes.getvalue()
      
      # with open(byte_im, "rb") as file:
      btn = st.download_button(
              label="Download image",
              data=byte_im,
              file_name='translated_image'+ source_key_list[source_index] + '-' + trans_key_list[trans_index] + '.jpg',
              mime="image/jpg"
            )
