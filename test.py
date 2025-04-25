# reencode.py
import codecs

# Читаем исходный файл в MacCyrillic
with codecs.open("scene/circular_track.xml", "r", encoding="mac_cyrillic") as src:
    text = src.read()

# Перезаписываем его в UTF-8 (без BOM)
with codecs.open("scene/circular_track.xml", "w", encoding="utf-8") as dst:
    dst.write(text)
