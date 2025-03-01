import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import string


my_text = "EATEN. US NAVY CRYPTOGRAPHERS WITH COOPERATION FROM BRITISH AND DUTCH CRYPTOGRAPHERS AFTER BROKE INTO SEVERAL JAPANESE NAVY CRYPTO SYSTEMS. THE BREAK INTO ONE OF THEM, JN-, FAMOUSLY LED TO THE US VICTORY IN THE BATTLE OF MIDWAY; AND TO  THE PUBLICATION OF THAT FACT IN THE CHICAGO TRIBUNE SHORTLY AFTER THE BATTLE, THOUGH THE JAPANESE SEEM NOT TO HAVE NOTICED FOR THEY KEPT USING THE JN- SYSTEM. A USAT THE END OF THE WAR, ON APRIL, BRITAIN'S TOP MILITARY OFFICERS WERE TOLD THAT THEY COULD NEVER REVEAL THAT THE GERMAN ENIGMA CIPHER HAD BEEN BROKEN BECAUSE IT WOULD GIVE THE DEFEATED ENEMY THE CHANCE TO SAY THEY WERE NOT WELL AND FAIRLY B"


alphabet2 = ['A',
             'B',
             'C',
             'D',
             'E',
             'F',
             'G',
             'H',
             'I',
             'J',
             'K',
             'L',
             'M',
             'N',
             'O',
             'P',
             'Q',
             'R',
             'S',
             'T',
             'U',
             'V',
             'W', 'X', 'Y', 'Z', ' ', '.', ',', ';', '-', "'",]


# Функція для визначення використаного алфавіту
def get_alphabet(text):
    return sorted(set(text))

print('new_changes')

# Функція для підрахунку частоти символів
def get_char_frequencies(text):
    return Counter(text)


# Функція для підрахунку біграм
def get_bigrams(text):
    bigrams = [text[i:i + 2] for i in range(len(text) - 1)]
    return Counter(bigrams)


# Функція для підрахунку трірам
def get_trigrams(text):
    trigrams = [text[i:i + 3] for i in range(len(text) - 2)]
    return Counter(trigrams)


def get_quatro(text):
    trigrams = [text[i:i + 4] for i in range(len(text) - 2)]
    return Counter(trigrams)


# Функція для побудови гістограми
def plot_histogram(freqs, title):
    chars, counts = zip(*freqs)
    plt.bar(chars, counts)
    plt.title(title)
    plt.xlabel('Символи')
    plt.ylabel('Частота')
    plt.show()


# Основна функція для аналізу тексту
def analyze_text(text):
    # Фільтрація тексту для використання тільки допустимих символів
    allowed_chars = string.ascii_letters + string.digits + string.punctuation + " "
    filtered_text = ''.join([char for char in text if char in allowed_chars]).replace(' ','_')

    # Отримуємо алфавіт
    alphabet = get_alphabet(filtered_text)
    print(f"Використаний алфавіт: {''.join(alphabet)}")

    # Отримуємо частоту повторень символів
    frequencies = get_char_frequencies(filtered_text)

    # Виводимо частоти символів за алфавітним порядком
    print("\nЧастоти символів (за алфавітним порядком):")
    for char in sorted(frequencies):
        print(f"{char}: {frequencies[char]}")

    # Виводимо частоти символів за спаданням частоти
    sorted_by_frequency = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
    print("\nЧастоти символів (за спаданням частоти):")
    for char, freq in sorted_by_frequency:
        print(f"{char}: {freq}")

    # Побудова гістограм
    plot_histogram(sorted(frequencies.items()), "Частоти символів (за алфавітним порядком)")
    plot_histogram(sorted_by_frequency, "Частоти символів (за спаданням частоти)")

    # Аналіз біграм
    bigram_frequencies = get_bigrams(filtered_text)
    filtered_bigrams = [(k, v) for k, v in bigram_frequencies.items()][:15]
    sorted_bigrams = sorted(filtered_bigrams, key=lambda x: x[1], reverse=True)

    print("\nБіграми:")
    for bigram, freq in sorted_bigrams:
        print(f"{bigram}: {freq}")

    # Побудова гістограми для біграм
    if sorted_bigrams:
        plot_histogram(sorted_bigrams, "Частоти біграм")

    # Аналіз триграм
    trigram_frequencies = get_trigrams(filtered_text)
    filtered_trigrams = [(k, v) for k, v in trigram_frequencies.items()][:15]
    sorted_trigrams = sorted(filtered_trigrams, key=lambda x: x[1], reverse=True)

    print("\nТриграми:")
    for trigram, freq in sorted_trigrams:
        print(f"{trigram}: {freq}")

    # Побудова гістограми для трірам
    if sorted_trigrams:
        plot_histogram(sorted_trigrams, "Частоти триграм ")

        # Аналіз трірам
    quatro_frequencies = get_quatro(filtered_text)
    filtered_quatrograms = [(k, v) for k, v in quatro_frequencies.items()][:10]
    sorted_quatrograms = sorted(filtered_quatrograms, key=lambda x: x[1], reverse=True)

    print("\nЧотириграми:")
    for quatro, freq in sorted_quatrograms:
        print(f"{quatro}: {freq}")

    # Побудова гістограми для трірам
    if sorted_quatrograms:
        plot_histogram(sorted_quatrograms, "Частоти чотириграми ")

    # Пошук повторень символів для 2, 3 і 4 символів
    for n in [2, 3, 4]:
        sequences = [filtered_text[i:i + n] for i in range(len(filtered_text) - n + 1)]
        sequence_frequencies = Counter(sequences)
        sorted_sequences = sorted(sequence_frequencies.items(), key=lambda x: x[1], reverse=True)

        print(f"\nПовторення для послідовностей з {n} символів:")
        for seq, freq in sorted_sequences[:10]:  # Виводимо топ-10 повторень
            print(f"{seq}: {freq}")


#analyze_text(my_text)


# Функція для перетворення букви у число відносно alphabet2
def letter_to_number(letter):
    try:
        return alphabet2.index(letter)
    except ValueError:
        raise ValueError(f"Неприпустимий символ для шифрування: {letter}")


# Функція для перетворення числа у букву відносно alphabet2
def number_to_letter(number):
    return alphabet2[number % len(alphabet2)]


# Функція для створення матриці ключа з прізвища та імені
def create_key_matrix(surname, name):
    key_string = (surname + name)[:9].upper()
    key_numbers = [letter_to_number(c) for c in key_string]
    return np.array(key_numbers).reshape(3, 3)


def get_minor(matrix, row, col):
    """Повертає підматрицю, видаливши вказаний рядок і стовпець."""
    return np.delete(np.delete(matrix, row, axis=0), col, axis=1)


def get_cofactor_matrix(matrix):
    """Обчислює матрицю алгебраїчних доповнень (кофакторів)."""
    n = matrix.shape[0]
    cofactor_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            minor = get_minor(matrix, i, j)
            cofactor = ((-1) ** (i + j)) * np.linalg.det(minor)
            cofactor_matrix[i, j] = cofactor

    return cofactor_matrix


def mod_with_sign(value, mod):
    """Обчислює залишок по модулю, зберігаючи знак для додатніх та від’ємних чисел."""
    if value >= 0:
        return value % mod  # Для додатніх чисел
    else:
        return -(-value % mod)

def extended_gcd(a, b):
    """Розширений алгоритм Евкліда."""
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y


def mod_inverse(det, mod):
    """Знаходить обернений елемент детермінанту по модулю."""
    gcd, x, _ = extended_gcd(det, mod)
    if gcd != 1:
        raise ValueError("Обернений елемент не існує (детермінант та модуль не взаємно прості).")
    return x % mod  # Повертаємо обернений елемент


# Функція для обчислення оберненої матриці в модулі len(alphabet2)
def mod_matrix_inverse(matrix, mod):
    # Обчислюємо визначник
    det = int(np.round(np.linalg.det(matrix)))  # Обчислюємо визначник
    print(mod)
    print(det)
    print(np.gcd(det, mod))
    if det == 0 or np.gcd(det, mod) != 1:  # Перевіряємо, чи визначник взаємно простий з модулем
        raise ValueError("Матриця не має оберненої (визначник 0 або не взаємно простий з модулем).")

    # Знаходимо обернений визначник в модулі
    det_inv = mod_inverse(det, mod)

    # Обчислюємо матрицю доповнень і транспонуємо її
    cofactor_matrix = np.vectorize(lambda x: mod_with_sign(x, mod))(get_cofactor_matrix(matrix))

    test1 = cofactor_matrix * det_inv
    test2 = np.vectorize(lambda x: mod_with_sign(x, mod))(test1).T

    test2[test2 < 0] = mod + test2[test2 < 0]

    return test2


# Функція шифрування тексту методом Хілла
def encrypt_hill(plaintext, key_matrix):
    plaintext = plaintext.upper()

    # Перевіряємо наявність неприпустимих символів
    if not all(c in alphabet2 for c in plaintext):
        raise ValueError("Текст повинен містити тільки символи з визначеного алфавіту.")

    while len(plaintext) % 3 != 0:
        plaintext += ' '  # Додаємо пробіл, щоб текст був кратним 3
    blocks = [plaintext[i:i + 3] for i in range(0, len(plaintext), 3)]
    encrypted_text = ''
    print(key_matrix)
    for block in blocks:
        print(block)
        vector = np.array([letter_to_number(c) for c in block])
        encrypted_vector = np.dot(vector, key_matrix) % len(alphabet2)
        encrypted_text += ''.join(number_to_letter(n) for n in encrypted_vector)
        print(vector)
        print(encrypted_vector)
        print(encrypted_text)

    return encrypted_text


# Функція дешифрування тексту методом Хілла
def decrypt_hill(ciphertext, inverse_key_matrix):
    # Перевіряємо наявність неприпустимих символів
    if not all(c in alphabet2 for c in ciphertext):
        raise ValueError("Текст повинен містити тільки символи з визначеного алфавіту.")

    blocks = [ciphertext[i:i + 3] for i in range(0, len(ciphertext), 3)]
    decrypted_text = ''

    for block in blocks:
        vector = np.array([letter_to_number(c) for c in block])
        decrypted_vector = np.dot(vector, inverse_key_matrix) % len(alphabet2)
        decrypted_vector = np.round(decrypted_vector).astype(int)
        decrypted_text += ''.join(number_to_letter(int(n)) for n in decrypted_vector)

    return decrypted_text


# Побудова графіка частот
def plot_frequencies(plaintext_freq, ciphertext_freq):
    plt.figure(figsize=(10, 6))

    # Підготовка даних для графіка
    sorted_plaintext_freq = dict(sorted(plaintext_freq.items(), key=lambda item: item[1], reverse=True))
    sorted_ciphertext_freq = dict(sorted(ciphertext_freq.items(), key=lambda item: item[1], reverse=True))

    # Масиви частот
    symbols = sorted_plaintext_freq.keys()
    plaintext_values = sorted_plaintext_freq.values()
    ciphertext_values = [ciphertext_freq.get(symbol, 0) for symbol in symbols]

    # Побудова графіків
    plt.plot(symbols, plaintext_values, label='Відкритий текст', marker='o')
    plt.plot(symbols, ciphertext_values, label='Зашифрований текст', marker='x')

    # Налаштування графіка
    plt.title('Частоти символів у відкритому та зашифрованому текстах')
    plt.xlabel('Символ')
    plt.ylabel('Частота')
    plt.legend()

    plt.show()


def char_frequency(text):
   text = text.upper()
   counter = Counter([char for char in text if char in alphabet2])
   return counter

# Основна функція

def lab5(text):
    surname = "AM.QJOJIN"
    name = "ROMAN"

    # Створення матриці ключа
    key_matrix = create_key_matrix(surname, name)
    print(f"Матриця ключа:\n{key_matrix}")

    # Обчислення оберненої матриці
    inverse_key_matrix = mod_matrix_inverse(key_matrix, len(alphabet2))
    print(f"Обернена матриця ключа:\n{inverse_key_matrix}")

    # Шифрування
    encrypted_text = encrypt_hill(text, key_matrix)
    print(f"Шифрований текст: {encrypted_text}")

    # Дешифрування
    decrypted_text = decrypt_hill(encrypted_text, inverse_key_matrix)
    print(f"Розшифрований текст: {decrypted_text}")

    analyze_text(encrypted_text)
    # Статистичний аналіз частот символів
    plaintext_freq = char_frequency(decrypted_text)
    ciphertext_freq = char_frequency(encrypted_text)

    # Відображення графіку
    plot_frequencies(plaintext_freq, ciphertext_freq)


lab5(my_text)