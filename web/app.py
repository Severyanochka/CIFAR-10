from flask import Flask
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image

# Инициализация Flask-приложения
app = Flask(__name__)

# Загрузка обученной модели
model = load_model('model.py')  # замените на путь к вашей сохранённой модели

# Классы CIFAR-10
class_names = ['Самолёт', 'Автомобиль', 'Птица', 'Кот', 'Олень', 'Собака', 'Лягушка', 'Лошадь', 'Корабль', 'Грузовик']

# Главная страница с формой для загрузки изображения
@app.route("/", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Сохранение загруженного файла
            image = Image.open(file)
            image = image.resize((32, 32))  # Изменение размера изображения под размер CIFAR-10
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = image.astype('float32') / 255.0  # Нормализация

            # Классификация изображения
            prediction = model.predict(image)
            class_idx = np.argmax(prediction)
            class_name = class_names[class_idx]
            confidence = np.max(prediction) * 100

            return render_template('result.html', class_name=class_name, confidence=confidence)

    return render_template('upload.html')

# Запуск приложения
if __name__ == "__main__":
    app.run(debug=True)
