months = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]
seasons = ["зима", "весна", "лето", "осень"]
month = int(input())
if month < 1 or month > 12:
    print("ошибка")
else:
    print(months[month - 1])
    print(seasons[(month % 12) // 3])
