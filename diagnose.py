"""Диагностика: вероятные причины отказа по схеме (netlist.json) и симптомам.

Как думает модуль:
  1. По цепям и номиналам определяет роль каждой детали: силовой ключ, конденсатор
     шины, выпрямитель, драйвер затвора, бутстреп, предзаряд, стабилизатор и т.д.
  2. Оценивает нагрузку: напряжение цепей (по меткам «+600V», «+12 В», «~230 В»)
     против допустимого напряжения детали (номинал «400V» или справочник).
  3. Для каждой детали берёт типичные виды отказа её типа, усиливает те, что
     подходят к симптомам, роли и нагрузке, и сортирует по вероятности.
  4. Отдельно ищет ошибки и слабые места схемы (они сами по себе причина отказов).
  5. Учится на ремонтах: «подтвердить» (деталь оказалась виновата) и «исправна»
     пишутся в learned/repairs.jsonl и меняют вес этой роли при этих симптомах.

    python diagnose.py папка_проекта [симптом,симптом]
    python diagnose.py confirm папка_проекта REF [симптом,…]   — деталь была неисправна
    python diagnose.py ok папка_проекта REF [симптом,…]        — деталь проверена, исправна
    python diagnose.py symptoms                                — список симптомов
"""
import json
import math
import re
import sys
import time
from pathlib import Path

import core

LEARNED = Path(__file__).resolve().parent / "learned"

SYMPTOMS = {
    "no_power": "Не включается, нет признаков жизни",
    "fuse": "Сгорает предохранитель / выбивает автомат",
    "smoke": "Дым, запах гари, следы нагара",
    "overheat": "Перегрев детали или узла",
    "no_output": "Нет выходного напряжения/сигнала",
    "one_channel": "Не работает одна фаза / один канал",
    "noise": "Гул, писк, пульсации",
    "unstable": "Сбои, нестабильная работа, перезапуски",
    "protection": "Срабатывает защита / ошибка",
    "weak": "Слабый выход, просадка под нагрузкой",
    "after_time": "Отказывает после прогрева или со временем",
    # КИП и микроконтроллеры
    "no_signal": "Датчик: нет сигнала / ток петли < 3,6 мА / «обрыв»",
    "wrong_reading": "Датчик: неверные или зашкаливающие показания",
    "noisy": "Показания скачут, шумят",
    "drift": "Показания плывут (со временем, от температуры)",
    "mcu_hang": "Микроконтроллер не стартует, зависает, сбрасывается",
    "comm": "Нет связи (UART, I²C, SPI, RS-485)",
    # привод и обратная связь
    "overspeed": "Двигатель идёт «вразнос», скорость не ограничивается",
    "speed_hunt": "Скорость плавает, раскачивается, двигатель дёргается",
    "position_error": "Ошибка положения, пропуск импульсов энкодера",
    "out_high": "Выходное напряжение завышено или плавает",
    # схемы управления
    "no_start": "Двигатель / механизм не запускается кнопкой «Пуск»",
    "no_stop": "Не отключается кнопкой «Стоп»",
    "no_hold": "Работает, только пока держат «Пуск» (нет самоподхвата)",
    "trips": "Выбивает автомат / срабатывает тепловое реле",
    "hum": "Пускатель гудит, дребезжит",
}

# Описание словами → симптомы (корни слов, чтобы ловить «не запускается», «не запустился», «не пускается»)
SYMPTOM_WORDS = [
    (r"не\s*(за)?пуска|не\s*(за)?пуст|не\s*старту|не\s*включает(ся)?\s*(двигат|мотор|пускат)|не\s*крутит", "no_start"),
    (r"не\s*(вы)?ключа.*стоп|не\s*останавл|не\s*отключа", "no_stop"),
    (r"пока\s*держ|не\s*держит|самоподхват|отпуска\w*\s*кнопк", "no_hold"),
    (r"выбива|срабатыва\w*\s*(автомат|теплов)|отключа\w*\s*автомат|теплов\w*\s*реле", "trips"),
    (r"гуд\w*\s*пускат|дребезж|пускатель\s*гуд|контактор\s*гуд", "hum"),
    (r"не\s*включа|нет\s*признак|мёртв|мертв|нет\s*питан|не\s*горит\s*индикац", "no_power"),
    (r"предохранит|сгорает\s*плавк", "fuse"),
    (r"дым|гар[ьи]|нагар|запах", "smoke"),
    (r"гре[её]тся|перегре|горяч", "overheat"),
    (r"нет\s*выход|нет\s*напряжения\s*на\s*выход|нет\s*сигнала\s*на\s*выход", "no_output"),
    (r"одн\w*\s*(фаз|канал|плеч)", "one_channel"),
    (r"гул|писк|свист|пульсац", "noise"),
    (r"сбо|глюч|перезапуска|перезагруж|нестабил", "unstable"),
    (r"защит|ошибк|авари", "protection"),
    (r"слаб|просед|не\s*тянет|мало\s*мощн", "weak"),
    (r"после\s*прогрев|через\s*(некоторое\s*)?время|со\s*временем|прогре", "after_time"),
    (r"4.?20|обрыв\s*датч|нет\s*сигнала|ток\s*петл", "no_signal"),
    (r"неверн\w*\s*показ|зашкал|врёт|врет|не\s*те\s*показ", "wrong_reading"),
    (r"скач|шумят|прыга", "noisy"),
    (r"плыв|дрейф|уход\w*\s*показ", "drift"),
    (r"контроллер\w*\s*(завис|не\s*старт)|мк\s*завис|зависа|сбрасыва", "mcu_hang"),
    (r"нет\s*связи|rs.?485|modbus|uart|i2c|spi", "comm"),
    (r"вразнос|разгоня|обороты\s*не\s*огранич|максимальн\w*\s*скорост", "overspeed"),
    (r"скорость\s*плава|раскачива|дёрга|дерга|рывк", "speed_hunt"),
    (r"энкодер|положени|пропуск\w*\s*импульс", "position_error"),
    (r"завышен|напряжение\s*выше|выходное\s*плава", "out_high"),
]


def parse_symptoms(text):
    """«Двигатель не запускается, пускатель гудит» → ['no_start', 'hum']."""
    t = str(text or "").lower().replace("ё", "е")
    out = []
    for pat, key in SYMPTOM_WORDS:
        if re.search(pat.replace("ё", "е"), t) and key not in out:
            out.append(key)
    if "no_start" in out and "no_power" in out and not re.search(r"не\s*включа\w*\s*(сам|вообще|прибор|блок|плат)", t):
        out.remove("no_power")                       # «не включается двигатель» — это no_start
    return out


# Виды отказа по типу детали: (вид, базовый вес, симптомы, как проверить)
MODES = {
    "M": [("пробой сток-исток", 1.0, {"fuse", "smoke", "no_power", "protection", "overheat"},
           "прозвонить сток-исток и затвор-исток мультиметром (выпаянный): должно быть не КЗ"),
          ("обрыв / не открывается", 0.4, {"no_output", "one_channel", "weak"},
           "подать 10–12 В на затвор через 1 кОм, проверить открывание")],
    "IGBT": [("пробой коллектор-эмиттер", 1.0, {"fuse", "smoke", "no_power", "protection", "overheat"},
              "проверить диодным тестером К-Э и встречный диод модуля"),
             ("обрыв / не открывается", 0.4, {"no_output", "one_channel", "weak"},
              "проверить открывание от источника 15 В через резистор")],
    "VS": [("пробой (КЗ анод–катод / A1–A2)", 1.0, {"fuse", "trips", "no_stop", "smoke", "overheat", "protection"},
            "омметром A–K (A1–A2) в обе стороны: должно быть > 100 кОм; КЗ — пробит"),
           ("не открывается (обрыв управляющего перехода, перегорел)", 0.7, {"no_start", "no_output", "one_channel", "weak"},
            "G–K (G–A1) 10…100 Ом; открывание от батареи 9 В с лампой: ток в G — лампа горит"),
           ("самопроизвольно открывается (dU/dt, помехи, нет снаббера)", 0.5, {"no_stop", "unstable", "hum"},
            "осмотреть RC-снаббер (обрыв R или C), резистор G–K, наводки на провод управления")],
    "Q": [("пробой переходов", 0.8, {"no_output", "unstable", "protection", "smoke"},
           "диодной прозвонкой Б-Э, Б-К; К-Э не должно звониться"),
          ("утечка / потеря усиления", 0.3, {"weak", "after_time", "unstable"},
           "сравнить с исправным, проверить h21э")],
    "C": [("высох / потерял ёмкость", 1.0, {"noise", "unstable", "weak", "after_time", "protection", "speed_hunt",
                                            "noisy", "mcu_hang", "out_high"},
           "измерить ёмкость и ESR, осмотреть на вздутие"),
          ("пробой / утечка", 0.5, {"fuse", "smoke", "no_power", "overheat"},
           "проверить на КЗ, измерить утечку при рабочем напряжении")],
    "D": [("пробой", 0.8, {"fuse", "no_power", "smoke", "one_channel"},
           "диодная прозвонка в обе стороны"),
          ("обрыв", 0.4, {"no_output", "weak", "one_channel"}, "диодная прозвонка")],
    "Z": [("пробой / сдвиг напряжения", 0.6, {"protection", "unstable", "no_output"},
           "измерить напряжение стабилизации")],
    "BR": [("пробой плеча моста", 1.0, {"fuse", "no_power", "smoke"},
            "прозвонить каждое из 4 плеч"),
           ("обрыв плеча", 0.4, {"noise", "weak"}, "прозвонить плечи, посмотреть пульсации")],
    "R": [("обрыв / увеличение сопротивления", 0.6, {"no_output", "no_power", "weak", "one_channel", "no_signal",
                                                     "wrong_reading", "overspeed", "out_high", "position_error"},
           "измерить сопротивление (одним выводом выпаять)"),
          ("перегрев, потемнение", 0.3, {"overheat", "smoke", "after_time"}, "осмотреть, измерить")],
    "L": [("межвитковое замыкание / обрыв", 0.4, {"no_output", "overheat", "noise"},
           "прозвонить, сравнить индуктивность")],
    "T": [("обрыв / межвитковое", 0.6, {"no_power", "no_output", "overheat", "noise"},
           "прозвонить обмотки, измерить напряжение холостого хода")],
    "U": [("выход из строя микросхемы", 0.6, {"no_output", "unstable", "protection", "one_channel"},
           "проверить питание на выводах, сигналы на входах и выходах"),
          ("пробой по питанию", 0.3, {"no_power", "overheat", "smoke"},
           "сопротивление между выводами питания")],
    "OC": [("деградация светодиода / выхода", 0.6, {"no_output", "one_channel", "after_time"},
            "ток через светодиод и выходной сигнал")],
    "K": [("обгорели контакты / не включается", 0.7, {"no_power", "overheat", "weak", "after_time", "smoke"},
           "напряжение на обмотке, сопротивление замкнутого контакта")],
    "RV": [("пробой / разрушение варистора", 0.7, {"fuse", "smoke", "no_power"},
            "осмотреть на трещины, проверить на КЗ")],
    "F": [("сгорел", 0.8, {"no_power", "fuse"}, "прозвонить — и искать причину, почему сгорел")],
    "SENSOR": [("обрыв линии / петли", 1.0, {"no_signal", "wrong_reading"},
                "измерить ток петли миллиамперметром в разрыв: 0 мА — обрыв, 3,6–4 мА — ноль шкалы"),
               ("влага, окисление клемм, утечка на кабеле", 0.8, {"noisy", "drift", "wrong_reading"},
                "мегомметром (при отключённой электронике) кабель на экран и землю, осмотреть клеммы"),
               ("деградация/перегрузка чувствительного элемента", 0.6, {"drift", "wrong_reading"},
                "сравнить с эталоном в 2–3 точках шкалы; проверить, не превышался ли диапазон (мембрана, гидроудар)"),
               ("неисправна электроника преобразователя", 0.5, {"no_signal", "wrong_reading", "noisy"},
                "питание на клеммах датчика ≥ мин. напряжения; ток при подаче калибровочного сигнала")],
    "RT": [("обрыв или КЗ терморезистора", 0.8, {"no_signal", "wrong_reading"},
            "сопротивление при известной температуре: Pt100 ≈ 107,8 Ом при 20 °C, NTC 10k ≈ 12,5 кОм"),
           ("влага / утечка, плохой контакт", 0.7, {"noisy", "drift", "wrong_reading"},
            "сопротивление изоляции, переходное сопротивление клемм (для 2-проводной схемы 1 Ом = +2,6 °C у Pt100)")],
    "TC": [("обрыв спая или термоэлектрода", 0.8, {"no_signal", "wrong_reading"}, "прозвонка — единицы-десятки Ом"),
           ("ошибка компенсации холодного спая / не тот удлинительный провод", 0.7, {"wrong_reading", "drift"},
            "температура клемм, тип провода (K/J/…), полярность"),
           ("деградация (окисление, дрейф ТЭДС)", 0.4, {"drift"}, "сравнить с эталоном")],
    "XTAL": [("не запускается / уход частоты", 0.8, {"mcu_hang", "comm"},
              "осциллограф с щупом ×10 на выходе генератора; проверить конденсаторы нагрузки")],
    "LED": [("обрыв / деградация", 0.5, {"no_output"}, "диодной прозвонкой — должен слабо светиться")],
    "MCU": [("выгорел порт/АЦП (перенапряжение, статика)", 0.8, {"wrong_reading", "no_signal", "comm"},
             "напряжение на выводе относительно соседних, ток утечки входа; сравнить каналы"),
            ("сбой по питанию, сбросу, тактированию", 0.7, {"mcu_hang", "unstable"},
             "VCC и пульсации на выводах, уровень RESET, наличие тактовой частоты"),
            ("повреждена прошивка / настройки", 0.3, {"mcu_hang", "comm", "wrong_reading"}, "перепрошить, проверить конфигурацию")],
    "ENC": [("загрязнение диска / дефект оптики", 0.7, {"position_error", "speed_hunt"},
             "осциллограф на A и B: скважность ≈ 50 %, сдвиг 90°, без выпадающих импульсов"),
            ("обрыв канала или кабеля", 0.8, {"position_error", "overspeed", "protection"},
             "прозвонить жилы, проверить питание энкодера на его разъёме"),
            ("помехи по кабелю (нет экрана, рядом силовой кабель)", 0.6, {"position_error", "speed_hunt"},
             "экран заземлён с одной стороны, кабель отдельно от силовых")],
    "TACHO": [("износ щёток / коллектора", 0.8, {"speed_hunt", "noisy"}, "пульсации выхода осциллографом, осмотр щёток"),
              ("обрыв цепи тахогенератора", 0.8, {"overspeed"},
               "напряжение на выходе при вращении от руки; при обрыве регулятор разгоняет двигатель до максимума"),
              ("размагничивание / неверная крутизна", 0.4, {"overspeed", "wrong_reading"}, "U при известной скорости")],
    "HALL": [("отказ датчика / неверный зазор", 0.8, {"no_signal", "position_error", "speed_hunt"},
              "уровни на выходе при поднесении магнита/метки, зазор по паспорту")],
    "SW": [("контакт не замыкается (обгорел, окислился)", 0.9, {"no_start", "no_hold", "no_output"},
            "напряжение на контакте при включённом аппарате: должно быть ≈0 В; прозвонить снятый"),
           ("контакт залип / сварился", 0.6, {"no_stop"}, "прозвонка при отпущенном якоре — не должно звониться")],
    "KK": [("срабатывает: перегрузка или заклинивание двигателя, обрыв фазы, уставка мала", 0.9, {"trips", "no_start"},
            "ток двигателя клещами по трём фазам; уставка теплового реле ≈ 1,05–1,2·Iном двигателя"),
           ("контакт 95–96 не замыкается / реле не взведено", 0.6, {"no_start"}, "кнопка Reset; прозвонить 95–96"),
           ("не срабатывает при перегрузке (залип, уставка велика)", 0.3, {"overheat", "smoke"},
            "проверить уставку и срабатывание тестовой кнопкой")],
    "BTN": [("кнопка не замыкает", 0.9, {"no_start", "no_hold"}, "прозвонить в нажатом положении"),
            ("кнопка «Стоп» залипла / замкнута накоротко", 0.5, {"no_stop"}, "прозвонить в отпущенном и нажатом положении")],
    "K": [("обрыв катушки", 0.8, {"no_start", "no_output"}, "сопротивление катушки (десятки–сотни Ом); напряжение на A1–A2"),
          ("межвитковое замыкание / перегрев катушки", 0.5, {"trips", "hum", "overheat", "smoke"},
           "сопротивление ниже нормы, запах, потемнение"),
          ("механика: якорь не притягивается, грязь на сердечнике", 0.5, {"hum", "no_hold"},
           "осмотреть, почистить сердечник, проверить пружину"),
          ("обгорели силовые контакты", 0.7, {"no_start", "weak", "overheat"}, "падение напряжения на каждом полюсе под нагрузкой"),
          ("сварились силовые контакты / заклинило якорь", 0.7, {"no_stop"},
           "при снятом питании катушки двигатель не должен работать; осмотреть контакты и подвижную систему")],
    "QF": [("расцепитель срабатывает (перегрузка, КЗ ниже по цепи)", 0.7, {"trips", "no_power"},
            "мегомметр нагрузки на землю и между фазами; ток нагрузки клещами"),
           ("износ автомата: не включается, плохой контакт", 0.4, {"no_power", "overheat"}, "падение напряжения на полюсе")],
    "LAMP": [("перегорела", 0.6, {"no_output"}, "прозвонить, проверить напряжение на патроне")],
    "MOTOR": [("межвитковое / обрыв обмотки", 0.8, {"trips", "no_start", "hum", "overheat", "smoke"},
               "сопротивление трёх обмоток должно совпадать; мегомметр на корпус ≥ 0,5 МОм"),
              ("подшипники, механическая перегрузка", 0.6, {"trips", "hum", "overheat"},
               "прокрутить вал рукой, ток холостого хода клещами")],
    "TERM": [("ослабла / окислилась клемма", 0.8, {"noisy", "no_signal", "drift", "wrong_reading",
                                                  "position_error", "speed_hunt"},
              "подтянуть, измерить переходное сопротивление и падение напряжения на клемме")],
    "L1": [], "J": [], "X": [("неисправность", 0.2, {"no_output", "unstable"}, "по документации")],
}

# Роли: множители к виду отказа при симптомах
ROLE_BOOST = {
    "power_switch": {"*": 2.2, "fuse": 1.6, "smoke": 1.5, "one_channel": 1.4},
    "bus_cap": {"*": 1.8, "noise": 2.0, "weak": 1.6, "after_time": 1.6, "protection": 1.3},
    "rectifier": {"*": 1.8, "fuse": 1.8, "no_power": 1.5},
    "gate_driver": {"*": 1.6, "one_channel": 2.0, "no_output": 1.4, "protection": 1.3},
    "bootstrap": {"*": 1.4, "one_channel": 2.0, "after_time": 1.3},
    "precharge": {"*": 1.6, "no_power": 1.8, "overheat": 1.5, "fuse": 1.3},
    "regulator": {"*": 1.5, "no_power": 1.6, "unstable": 1.4},
    "input_protection": {"*": 1.4, "fuse": 2.0, "smoke": 1.5},
    "snubber": {"*": 1.2, "overheat": 1.5, "smoke": 1.4},
    "gate_resistor": {"*": 1.1, "one_channel": 1.6},
    "shunt": {"*": 1.2, "protection": 1.8},
    "logic": {"*": 0.8},
    "signal": {"*": 0.7},
    "burden": {"*": 1.6, "no_signal": 2.0, "wrong_reading": 1.8, "drift": 1.4},
    "loop_protection": {"*": 1.3, "no_signal": 1.8, "wrong_reading": 1.5, "noisy": 1.3},
    "adc_input": {"*": 1.3, "wrong_reading": 1.8, "noisy": 1.6, "drift": 1.3},
    "sensor": {"*": 1.6, "no_signal": 1.5, "wrong_reading": 1.4, "drift": 1.5, "noisy": 1.3},
    "clock": {"*": 1.3, "mcu_hang": 2.2, "comm": 1.6},
    "mcu": {"*": 1.3, "mcu_hang": 1.8, "comm": 1.5},
    "pullup": {"*": 1.1, "comm": 2.0, "mcu_hang": 1.3},
    "decoupling": {"*": 1.0, "mcu_hang": 1.6, "noisy": 1.4, "unstable": 1.4},
    "speed_feedback": {"*": 1.5, "overspeed": 2.5, "speed_hunt": 2.0},
    "position_feedback": {"*": 1.5, "position_error": 2.5, "speed_hunt": 1.5, "overspeed": 1.3},
    "feedback": {"*": 1.5, "out_high": 2.5, "unstable": 1.5, "overspeed": 1.5},
    "start_chain": {"*": 1.3, "no_start": 2.2, "no_hold": 1.2},
    "stop_chain": {"*": 1.3, "no_stop": 2.5, "no_start": 1.5},
    "seal_in": {"*": 1.3, "no_hold": 3.0},
    "coil": {"*": 1.4, "no_start": 1.8, "hum": 2.2, "trips": 1.3, "no_stop": 1.8},
    "overload": {"*": 1.3, "trips": 2.2, "no_start": 1.5},
    "opamp": {"*": 1.1, "wrong_reading": 1.6, "noisy": 1.5, "out_high": 1.5, "no_output": 1.3, "drift": 1.3},
    "comparator": {"*": 1.1, "unstable": 1.8, "noisy": 1.6, "protection": 1.6, "no_output": 1.3},
    "schmitt": {"*": 1.1, "unstable": 1.5, "no_output": 1.4, "noisy": 1.3},
    "decoder": {"*": 1.1, "one_channel": 2.2, "no_output": 1.6},
}

ROLE_RU = {
    "power_switch": "силовой ключ", "bus_cap": "конденсатор силовой шины", "rectifier": "выпрямитель",
    "gate_driver": "драйвер затвора", "bootstrap": "бутстреп-питание драйвера", "precharge": "цепь предзаряда",
    "regulator": "стабилизатор питания", "input_protection": "защита входа",
    "snubber": "снаббер", "gate_resistor": "резистор затвора", "shunt": "токовый шунт",
    "logic": "логика/управление", "signal": "сигнальная цепь",
    "burden": "нагрузочный резистор токовой петли", "loop_protection": "защита входа 4–20 мА",
    "adc_input": "входная цепь АЦП", "sensor": "датчик КИП", "clock": "тактирование МК",
    "mcu": "микроконтроллер", "pullup": "подтяжка шины/сброса", "decoupling": "развязка питания МК",
    "speed_feedback": "обратная связь по скорости", "position_feedback": "обратная связь по положению",
    "feedback": "обратная связь по напряжению",
    "start_chain": "цепь пуска", "stop_chain": "цепь останова (размыкающие контакты)",
    "seal_in": "самоподхват", "coil": "катушка пускателя/реле", "overload": "защита (тепловое реле, автомат)",
    "opamp": "операционный усилитель", "comparator": "компаратор", "schmitt": "триггер Шмитта",
    "decoder": "дешифратор/шифратор",
}


# ---------- разбор номиналов и напряжений ----------

MULT = {"p": 1e-12, "п": 1e-12, "n": 1e-9, "н": 1e-9, "u": 1e-6, "µ": 1e-6, "мк": 1e-6, "m": 1e-3,
        "м": 1e-3, "k": 1e3, "к": 1e3, "M": 1e6, "М": 1e6, "G": 1e9}


def parse_value(s):
    """'560uF/400V' → (5.6e-4, 400); '4,7k' → (4700, None); '62' → (62, None)."""
    s = str(s or "").replace(",", ".").replace(" ", "")
    vmax = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*[VВ](?![a-zA-Zа-я])", s)
    if m and ("/" in s or s.upper().endswith(("V", "В"))):
        vmax = float(m.group(1))
    head = s.split("/")[0]
    m = re.match(r"^(\d+(?:\.\d+)?)(мк|[pпnнuµmмkкMМG])?", head)
    if not m:
        return None, vmax
    val = float(m.group(1)) * MULT.get(m.group(2) or "", 1.0)
    if head.upper().endswith(("V", "В")) and "/" not in s:
        return None, vmax                                # «620V» — только напряжение
    return val, vmax


def label_voltage(text):
    """Напряжение из метки цепи: '+600V POWER' → 600, '~230 В' → 325 (амплитуда)."""
    m = re.search(r"([~+\-−]?)\s*(\d+(?:[.,]\d+)?)\s*[VВ]", str(text or ""))
    if not m:
        return None
    v = float(m.group(2).replace(",", "."))
    return v * 1.414 if m.group(1) == "~" else v


def part_vmax(comp):
    val, vmax = parse_value(comp.get("value"))
    if vmax:
        return vmax
    import parts
    _, d = parts.lookup(comp.get("value", ""))
    if d and d.get("vmax"):
        return d["vmax"]
    v = str(comp.get("value", "")).upper().replace(" ", "")
    m = re.search(r"-(\d{3,4})\b", v)                    # BTA140-600, FS…-1200
    return float(m.group(1)) if m else None


# ---------- анализ схемы ----------

class Circuit:
    def __init__(self, nl):
        self.nl = nl
        self.comps = {c["ref"]: c for c in nl.get("components", []) if c.get("ref")}
        self.nets = core.net_index(nl)
        self.meta = nl.get("nets", {})
        self.volt = self._voltages()
        self.roles = {ref: self._roles(ref) for ref in self.comps}

    def pins(self, ref):
        return self.comps[ref].get("pins") or {}

    def neighbors(self, net):
        return [r for r, _ in self.nets.get(net, [])]

    def _voltages(self):
        """Напряжение цепей: из меток и имён, затем протягивается по путям, которые
        его почти не снижают: дроссели, предохранители, резисторы до 10 Ом, а через
        диоды и мосты — только в сторону катода (выпрямленное напряжение).
        Через ключи не протягивается: их выход «плавает», а плавающие цепи
        (бутстреп, затвор верхнего ключа) считаются относительно своего эмиттера."""
        v = {}
        for net in self.nets:
            for t in (self.meta.get(net, {}).get("label"), net):
                lv = label_voltage(t)
                if lv:
                    v[net] = max(v.get(net, 0), lv)
        if not any(k.upper() in core.GROUND_NAMES for k in v):
            for g in core.GROUND_NAMES:
                if g in self.nets:
                    v[g] = 0.0
        def setv(n, val):
            if n and n not in v and n.upper() not in core.GROUND_NAMES:
                v[n] = val

        for _ in range(4):
            for ref, c in self.comps.items():
                t = c.get("type")
                pins = self.pins(ref)
                val, _ = parse_value(c.get("value"))
                if t in ("L", "F") or (t == "R" and val is not None and val <= 10):
                    top = max((v.get(n, 0) for n in pins.values()), default=0)
                    if top > 50:
                        for n in pins.values():
                            setv(n, top)
                elif t == "D" and v.get(pins.get("A"), 0) > 50:
                    setv(pins.get("K"), v[pins["A"]])
                elif t == "BR":
                    ac = [v.get(n, 0) for k, n in pins.items() if k.upper().startswith("AC")]
                    if ac and max(ac) > 50:
                        setv(pins.get("P"), max(ac))
        # цепь без метки, но на ней электролит на ≥100 В к земле — шина питания,
        # рабочее напряжение примерно 80 % от номинала конденсатора
        for ref, c in self.comps.items():
            _, vmax = parse_value(c.get("value"))
            pins = self.pins(ref)
            if c.get("type") == "C" and vmax and vmax >= 100 and pins.get("-", "").upper() in core.GROUND_NAMES:
                setv(pins.get("+"), 0.8 * vmax)
        return v

    def net_v(self, net):
        return self.volt.get(net)

    def span_v(self, ref):
        """Наибольшая разность потенциалов на выводах детали (по известным цепям)."""
        vs = [self.volt[n] for n in self.pins(ref).values() if n in self.volt]
        if len(vs) >= 2:
            return max(vs) - min(vs)
        if len(vs) == 1 and vs[0] > 0:
            return vs[0]                                  # вторая сторона неизвестна
        return None

    def _roles(self, ref):
        c = self.comps[ref]
        t, val_s = c.get("type", "X"), str(c.get("value", "")).upper()
        val, _ = parse_value(c.get("value"))
        pins = self.pins(ref)
        nets = set(pins.values())
        hv = max((self.volt.get(n, 0) for n in nets), default=0)
        roles = []
        text = (c.get("note", "") + " " + " ".join(nets)).lower()
        if t in ("M", "IGBT", "VS") and hv >= 60:
            roles.append("power_switch")
        if t == "C" and ("+" in pins or (val and val >= 47e-6)) and hv >= 60:
            roles.append("bus_cap")
        if t == "C" and hv >= 60 and val and val < 1e-6 and any(
                self.comps.get(r, {}).get("type") == "R" for n in nets for r in self.neighbors(n) if r != ref):
            roles.append("snubber")
        if t in ("BR",) or (t == "D" and hv >= 100 and any(lbl in text for lbl in ("_in", "сеть", "ac", "~", "_f"))):
            roles.append("rectifier")
        if t in ("U", "OC") and re.search(r"IR2\d{3}|TLP\d+|HCPL|IRS2|EG\d{4}|UCC27", val_s):
            roles.append("gate_driver")
        if t == "U" and re.search(r"78\d\d|79\d\d|LM317|AMS1117|LM2596", val_s):
            roles.append("regulator")
        if t == "D" and any(re.search(r"VB|BOOT", n.upper()) for n in nets):
            roles.append("bootstrap")
        if t == "C" and any(re.search(r"VB|BOOT", n.upper()) for n in nets):
            roles.append("bootstrap")
        if t in ("R", "K") and ("предзаряд" in text or "precharge" in text or "pre" in ref.lower()):
            roles.append("precharge")
        if t in ("RV", "F") or (t == "C" and "PE" in nets):
            roles.append("input_protection")
        if t == "R" and val is not None and val <= 0.1:
            roles.append("shunt")
        if t == "R" and val is not None and val <= 100 and any(
                self.comps.get(r, {}).get("type") in ("M", "IGBT") and self.pins(r).get("G") in nets
                for n in nets for r in self.neighbors(n)):
            roles.append("gate_resistor")
        if t in ("U", "DA", "DD"):
            import parts
            kind = (parts.lookup(c.get("value", ""))[1] or {}).get("kind")
            role = {"OPAMP": "opamp", "COMP": "comparator", "SCHMITT": "schmitt", "DECODER": "decoder",
                    "ENCODER": "decoder"}.get(kind)
            if role:
                roles.append(role)
        roles += self._kip_roles(ref, t, val, nets)
        roles += self._ctrl_roles(ref, t, nets, text)
        if not roles:
            roles.append("logic" if t == "U" else "signal")
        return roles

    # ---------- схемы управления ----------

    def _ctrl_roles(self, ref, t, nets, text):
        """Роли в релейно-контакторной схеме: по типу и по тому, что стоит рядом.
        Размыкающий (NC) контакт или кнопка «Стоп» — цепь останова; замыкающий параллельно
        кнопке «Пуск» — самоподхват; катушка; тепловое реле и автомат — защита."""
        roles = []
        c = self.comps[ref]
        v = (str(c.get("value", "")) + " " + str(c.get("note", ""))).lower()
        if t == "K":
            roles.append("coil")
        if t in ("QF", "F") or "тепл" in v or "kk" in ref.lower() or "thermal" in v:
            roles.append("overload")
        if "overload" in roles and t == "SW":
            return roles                            # контакт теплового реле — защита, не цепь останова
        if t in ("SW", "BTN"):
            nc = re.search(r"\bnc\b|размык|стоп|stop|н\.?з\.?", v)
            if nc:
                roles.append("stop_chain")
            else:
                # самоподхват: замыкающий контакт параллелен кнопке (те же две цепи)
                twin = [r for n in nets for r in self.neighbors(n) if r != ref
                        and self.comps[r].get("type") == "BTN" and set(self.pins(r).values()) == set(nets)]
                roles.append("seal_in" if twin and t == "SW" else "start_chain")
        return roles

    # ---------- КИП и микроконтроллеры ----------

    def mcu_pins(self, kind):
        """Цепи выводов МК определённого назначения: adc, vcc, gnd, xtal, i2c, rst."""
        pat = {"adc": r"ADC|AIN|^A\d|AN\d", "vcc": r"^(VCC|VDD|AVCC|VDDA|3V3|5V)", "gnd": r"^(GND|VSS|AGND)",
               "xtal": r"XTAL|OSC|XIN|XOUT", "i2c": r"SDA|SCL", "rst": r"RST|RESET|NRST|MCLR"}[kind]
        out = set()
        for ref, c in self.comps.items():
            if c.get("type") == "MCU":
                for k, n in self.pins(ref).items():
                    if re.search(pat, str(k).upper()):
                        out.add(n)
        return out

    def sensor_nets(self):
        return {n for r, c in self.comps.items() if c.get("type") in ("SENSOR", "RT", "TC")
                for n in self.pins(r).values() if n.upper() not in core.GROUND_NAMES}

    def _kip_roles(self, ref, t, val, nets):
        roles = []
        sens = self.sensor_nets()
        adc = self.mcu_pins("adc")
        if t in ("SENSOR", "RT", "TC"):
            roles.append("sensor")
        for types, role in ((("TACHO",), "speed_feedback"), (("ENC", "HALL"), "position_feedback")):
            fbk = {n for r, c in self.comps.items() if c.get("type") in types
                   for n in self.pins(r).values() if n.upper() not in core.GROUND_NAMES}
            if t in types:
                roles.append(role)
            elif t in ("R", "C", "Z", "D") and nets & fbk and not any(
                    re.search(r"^\+?\d|VCC|VDD|\+U", n.upper()) for n in nets & fbk):
                roles.append(role)
        if t == "R" and any(re.search(r"TL431|LM317|LM2596|MC34063|UC384|LM2576|XL4015|MP1584",
                                      str(self.comps[r].get("value", "")).upper()) and
                            any(str(k).upper() in ("REF", "ADJ", "FB") and n in nets for k, n in self.pins(r).items())
                            for n in nets for r in self.neighbors(n)):
            roles.append("feedback")
        if t == "MCU":
            roles.append("mcu")
        if t == "XTAL" or (t == "C" and nets & self.mcu_pins("xtal")):
            roles.append("clock")
        gnd = {n for n in nets if n.upper() in core.GROUND_NAMES}
        if t == "R" and val and 50 <= val <= 600 and gnd and (nets & sens or any(
                re.search(r"4.?20|AI\d|LOOP|ПЕТЛ", n.upper() + str(self.meta.get(n, {}).get("label", "")).upper())
                for n in nets)):
            roles.append("burden")
        if t in ("Z", "D", "F") and nets & sens:
            roles.append("loop_protection")
        if t in ("R", "C") and nets & adc:
            roles.append("adc_input")
        if t == "R" and nets & (self.mcu_pins("i2c") | self.mcu_pins("rst")) and nets & self.mcu_pins("vcc"):
            roles.append("pullup")
        if t == "C" and nets & self.mcu_pins("vcc") and gnd:
            roles.append("decoupling")
        return roles

    # ---------- ошибки и слабые места схемы ----------

    def design_risks(self):
        risks = []

        def add(level, refs, text, why):
            risks.append({"level": level, "refs": refs, "text": text, "why": why})

        # 1. последовательные электролиты без выравнивания
        for net, conns in self.nets.items():
            refs = [r for r, _ in conns]
            if len(refs) >= 2 and all(self.comps[r].get("type") == "C" and "+" in self.pins(r) for r in refs):
                if all((self.net_v(n) or 0) >= 100 for r in refs for n in self.pins(r).values()
                       if n != net and n.upper() not in core.GROUND_NAMES) or True:
                    others = {n for r in refs for n in self.pins(r).values()} - {net}
                    if any((self.net_v(n) or 0) >= 100 for n in others):
                        add("высокий", refs,
                            "Последовательные электролиты без выравнивающих резисторов",
                            "из-за разных токов утечки напряжение делится неравно, один конденсатор "
                            "получает больше своего номинала и со временем пробивается")
        # 2. запас по напряжению
        for ref, c in self.comps.items():
            vmax = part_vmax(c)
            span = self.span_v(ref)
            if not vmax or not span or span < 20:
                continue
            series = self._series_partners(ref)
            eff = vmax * (1 + len(series))
            margin = eff / span
            kind = "конденсатора" if c.get("type") == "C" else "детали"
            if margin < 1.0:
                add("высокий", [ref] + series, f"Напряжение {span:.0f} В выше допустимого {kind} ({eff:.0f} В)",
                    "деталь работает за пределом паспорта — почти гарантированный отказ")
            elif margin < 1.25:
                add("средний", [ref] + series,
                    f"Малый запас по напряжению: {span:.0f} В при допустимых {eff:.0f} В (×{margin:.2f})",
                    "броски сети и выбросы при переключении легко превышают номинал; обычно берут запас ×1,5")
        # 3. затвор без резистора
        for ref, c in self.comps.items():
            if c.get("type") in ("M", "IGBT") and "power_switch" in self.roles[ref]:
                g = self.pins(ref).get("G")
                if g and any(self.comps[r].get("type") == "X" for r in self.neighbors(g)):
                    continue                              # затвор идёт в неразобранный блок
                if g and not any(self.comps[r].get("type") == "R" for r in self.neighbors(g) if r != ref):
                    add("средний", [ref], "Затвор силового ключа без резистора",
                        "звон и выбросы на затворе, перегрев драйвера, ложные открывания")
                if g and not any(self.comps[r].get("type") in ("Z", "D") for r in self.neighbors(g) if r != ref):
                    if not any(self.comps[r].get("type") == "R" and self._has_net_to(r, self.pins(ref).get("S") or self.pins(ref).get("E"))
                               for r in self.neighbors(g) if r != ref):
                        add("низкий", [ref], "Нет стабилитрона или резистора затвор-исток",
                            "при обрыве управления затвор «висит» и ключ может приоткрыться от помех")
        # 4. медленный диод в бутстрепе
        for ref, c in self.comps.items():
            if c.get("type") == "D" and "bootstrap" in self.roles[ref] and re.search(r"1N400\d", str(c.get("value"))):
                add("средний", [ref], f"{c.get('value')} в бутстрепе — медленный диод",
                    "время восстановления ~30 мкс: на высоких частотах ток течёт обратно, конденсатор "
                    "бутстрепа недозаряжается, верхний ключ открывается не полностью и греется; нужен UF4007/FR107")
        # 5. логика под потенциалом сети
        for net, m in self.meta.items():
            if net.upper() in core.GROUND_NAMES and re.search(r"сет|mains|под потенциал", str(m.get("label", "")).lower()):
                add("информация", [], f"Цепь {net} гальванически связана с сетью",
                    "при наладке нужен разделительный трансформатор, иначе осциллограф или паяльник "
                    "замыкают сеть и выжигают ключи и драйверы")
        # 6. неизвестные номиналы в силовых узлах
        for ref, c in self.comps.items():
            if "?" in str(c.get("value", "")) and any(r in self.roles[ref] for r in ("bus_cap", "power_switch", "rectifier", "precharge")):
                add("низкий", [ref], "Номинал неизвестен в силовом узле", "нельзя оценить нагрузку детали")
        seen, out = set(), []
        for r in risks:
            key = (r["text"], tuple(sorted(r["refs"])))
            if key not in seen:
                seen.add(key)
                out.append(r)
        order = {"высокий": 0, "средний": 1, "низкий": 2, "информация": 3}
        return sorted(out, key=lambda r: order[r["level"]])

    def _series_partners(self, ref):
        """Конденсаторы, стоящие последовательно с этим (средняя точка только между ними)."""
        c = self.comps[ref]
        if c.get("type") != "C":
            return []
        out = []
        for n in self.pins(ref).values():
            refs = [r for r in self.neighbors(n) if r != ref]
            if refs and all(self.comps[r].get("type") == "C" for r in refs) and \
                    n.upper() not in core.GROUND_NAMES and not self.meta.get(n, {}).get("port"):
                same = [r for r in refs if self.comps[r].get("value") == c.get("value")]
                if same:
                    out.append(same[0])
        return out[:1]

    def _has_net_to(self, ref, net):
        return net in self.pins(ref).values()


# ---------- обучение на ремонтах ----------

def load_repairs():
    path = LEARNED / "repairs.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def learned_factor(repairs, typ, role, symptoms):
    """Множитель по истории: сколько раз эта роль/тип оказывалась виновата при этих симптомах."""
    conf = bad = 0.0
    for r in repairs:
        if r["type"] != typ or role not in r.get("roles", []):
            continue
        w = 1.0 if not symptoms or set(symptoms) & set(r.get("symptoms", [])) else 0.3
        if r["result"] == "faulty":
            conf += w
        else:
            bad += w
    return (1 + conf) / (1 + 0.5 * bad)


def record(project, ref, result, symptoms):
    project = Path(project)
    nl = core.load_json(project / "netlist.json")
    circ = Circuit(nl)
    if ref not in circ.comps:
        raise SystemExit(f"нет детали {ref}")
    c = circ.comps[ref]
    LEARNED.mkdir(exist_ok=True)
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "project": project.name, "ref": ref,
           "type": c.get("type"), "value": c.get("value"), "roles": circ.roles[ref],
           "symptoms": symptoms, "result": result}
    with open(LEARNED / "repairs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


# ---------- главное ----------

def diagnose(project, symptoms=()):
    project = Path(project)
    nl = core.load_json(project / "netlist.json")
    if not nl:
        raise FileNotFoundError("нет netlist.json — сначала распознайте схему")
    circ = Circuit(nl)
    symptoms = [s for s in symptoms if s in SYMPTOMS]
    repairs = load_repairs()
    risks = circ.design_risks()
    import physics                                      # расчёт режимов: нагрузка каждой детали
    phys = physics.Physics(nl, core.load_json(project / "physics.json") or {}).run()
    for f in phys["findings"]:
        if f["level"] in ("высокий", "средний"):
            risks.append({"level": f["level"], "refs": f["refs"], "text": f["title"],
                          "why": f"{f['formula']} → {f['conclusion']}"})
    risk_refs = {}
    for r in risks:
        for ref in r["refs"]:
            risk_refs.setdefault(ref, []).append(r)
    import faults                                       # моделирование поломок этой схемы
    try:
        sim_idx = faults.symptom_index(faults.simulate(project))
        sim_types = set(faults.FAULTS)
    except Exception:
        sim_idx, sim_types = {}, set()

    cands = []
    for ref, c in circ.comps.items():
        t = c.get("type", "X")
        roles = circ.roles[ref]
        modes = MODES["KK"] if t == "SW" and "overload" in roles else MODES.get(t, MODES["X"])
        for mode, base, msym, check in modes:
            w = base
            why = []
            if symptoms:
                hit = set(symptoms) & msym
                if not hit:
                    w *= 0.15
                else:
                    w *= 1 + 0.8 * len(hit)
                    why.append("подходит к симптомам: " + ", ".join(SYMPTOMS[h].lower() for h in hit))
            for role in roles:
                b = ROLE_BOOST.get(role, {})
                f = b.get("*", 1.0) * math.prod(b.get(s, 1.0) for s in symptoms)
                w *= f
                if f > 1.05:
                    why.append(f"роль: {ROLE_RU[role]}")
            # роль детали никак не связана с указанными симптомами (а у других ролей связь есть) —
            # её перегрузка по расчёту — отдельная проблема, а не причина этого отказа
            specific = {s for r in ROLE_BOOST.values() for s in r if s != "*"}
            if symptoms and set(symptoms) & specific and not any(
                    s in ROLE_BOOST.get(role, {}) for role in roles for s in symptoms):
                w *= 0.3
            span = circ.span_v(ref)
            vmax = part_vmax(c)
            if span and vmax:
                k = vmax * (1 + len(circ._series_partners(ref))) / span
                if k < 1.25:
                    w *= 2.5 if k < 1 else 1.7
                    why.append(f"работает с малым запасом по напряжению ({span:.0f} В из {vmax:.0f} В)")
            elif span and span >= 100:
                w *= 1.2
                why.append(f"под высоким напряжением (~{span:.0f} В)")
            got = set()
            if symptoms and t in sim_types:         # что показало моделирование поломок этой схемы
                got = sim_idx.get(ref, set()) & set(symptoms)
                if got:
                    w *= 2.0
                    why.append("моделирование: поломка этой детали даёт " +
                               ", ".join(SYMPTOMS[s].lower() for s in sorted(got)))
                elif any(s in {x for v in sim_idx.values() for x in v} for s in symptoms):
                    w *= 0.6                        # симптом в схеме моделируется, но не от этой детали
            sf, n_sf = faults.stats_factor(t, roles, symptoms)
            if n_sf and abs(sf - 1) > 0.05:         # статистика моделирования по всем схемам
                w *= sf
                why.append(f"по моделированию всех схем ×{sf:.2f}")
            k = phys["stress"].get(ref)
            related = not symptoms or got or any(s in ROLE_BOOST.get(role, {}) for role in roles for s in symptoms) \
                or bool(set(symptoms) & set(phys.get("stress_sym", {}).get(ref, [])))
            if k and k > 0.6 and related:           # нагрузка по расчёту: 1 = на пределе паспорта
                w *= 1 + 1.5 * min(k, 3) ** 2
                why.append(f"по расчёту нагружена на {k * 100:.0f} % допустимого")
            for r in risk_refs.get(ref, []):
                if r["level"] in ("высокий", "средний"):
                    w *= 1.6 if r["level"] == "высокий" else 1.25
                    why.append("слабое место схемы: " + r["text"].lower())
            lf = max(learned_factor(repairs, t, role, symptoms) for role in roles)
            if abs(lf - 1) > 0.05:
                w *= lf
                why.append(f"по истории ремонтов ×{lf:.2f}")
            if c.get("confidence") == "low":
                why.append("распознана неуверенно — сверить со схемой")
            cands.append({"ref": ref, "type": t, "value": c.get("value", ""), "cell": c.get("cell", ""),
                          "mode": mode, "weight": w, "roles": [ROLE_RU[r] for r in roles],
                          "why": why, "check": check})
    total = sum(c["weight"] for c in cands) or 1.0
    for c in cands:
        c["p"] = c["weight"] / total
    cands.sort(key=lambda c: -c["weight"])
    order = {"высокий": 0, "средний": 1, "низкий": 2, "информация": 3}
    risks.sort(key=lambda r: order.get(r["level"], 9))
    out = {"title": nl.get("title", ""), "symptoms": symptoms, "candidates": cands[:40], "risks": risks,
           "physics": phys["findings"], "node_voltages": phys["node_voltages"],
           "repairs_used": len(repairs),
           "voltages": {k: round(v, 1) for k, v in sorted(circ.volt.items(), key=lambda kv: -kv[1]) if v > 0}}
    import checkplan                                    # где искать и что, чем, с каким результатом измерить
    out["plan"] = checkplan.make(circ, phys, cands[:40], symptoms)
    (project / "diagnosis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (project / "diagnosis.md").write_text(to_markdown(out), encoding="utf-8")
    return out


def to_markdown(d):
    md = [f"# Диагностика: {d['title']}", ""]
    md.append("Симптомы: " + (", ".join(SYMPTOMS[s] for s in d["symptoms"]) or "не указаны (общий риск)"))
    md += ["", "## Слабые места схемы", ""]
    for r in d["risks"] or [{"level": "—", "refs": [], "text": "не найдено", "why": ""}]:
        md.append(f"- **[{r['level']}] {r['text']}** {', '.join(r['refs'])}. {r['why']}")
    md += ["", "## Что проверить в первую очередь", "",
           "| # | Деталь | Вероятность | Отказ | Почему | Как проверить |", "|---|---|---|---|---|---|"]
    for i, c in enumerate(d["candidates"][:20], 1):
        md.append(f"| {i} | {c['ref']} {c['value']} [{c['cell']}] | {c['p'] * 100:.1f}% | {c['mode']} | "
                  f"{'; '.join(c['why']) or ', '.join(c['roles'])} | {c['check']} |")
    if d.get("plan"):
        import checkplan
        md += ["", checkplan.to_markdown(d["plan"])]
    md += ["", f"_Учтено ремонтов из истории: {d['repairs_used']}. Вероятности — относительные оценки, "
               "не замена измерениям._", ""]
    import physics
    md.append(physics.to_markdown({"findings": d.get("physics", [])}))
    return "\n".join(md)


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        sys.exit(1)
    if a[0] == "symptoms":
        for k, v in SYMPTOMS.items():
            print(f"{k:12} {v}")
    elif a[0] in ("confirm", "ok"):
        syms = a[3].split(",") if len(a) > 3 else []
        row = record(a[1], a[2], "faulty" if a[0] == "confirm" else "ok", syms)
        print("записано:", row)
    else:
        syms = a[1].split(",") if len(a) > 1 else []
        if any(s not in SYMPTOMS for s in syms):          # описание словами: «не запускается двигатель»
            syms = parse_symptoms(" ".join(a[1:]))
            print("симптомы:", ", ".join(syms) or "не распознаны", file=sys.stderr)
        d = diagnose(a[0], syms)
        print(to_markdown(d))
