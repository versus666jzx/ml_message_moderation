import streamlit as st
import tools


st.title('Message moderation lab')


st.write(
	"""
	Термин «модерация» происходит от латинского «moderor», что значит «умерять, сдерживать». 
	Суть задачи модерации состоит в контроле за выполнением законов, правил, требований и ограничений в 
	любых сообществах и сервисах — будь то простое общение в социальных сетях или деловые переговоры на онлайн площадке.
	
	Автоматические системы модерации внедряются в веб-сервисы и приложения, где необходимо обрабатывать большое 
	количество сообщений пользователей. Такие системы позволяют сократить издержки на ручную модерацию, ускорить её и 
	обрабатывать все сообщения пользователей в real-time. 
	
	Со временем пользователи подстраиваются и учатся обманывать такие системы, например пользователи:
	- генерируют опечатки: you are stupit asswhol, fack u
	- заменяют буквенные символы на цифры, похожие по описанию: n1gga, b0ll0cks,
	-  вставляют дополнительные пробелы: i d i o t,
	- удаляют пробелы между словами: dieyoustupid
	- указывают контактные данные: восем-906-три единицы-два раза по две единицы
	и многое другое.

	Для того, чтобы обучить классификатор устойчивый к таким подменам, нужно поступить так, как поступают пользователи:
	сгенерировать такие же изменения в сообщениях и добавить их в обучающую выборку к основным данным.
	
	В целом, эта борьба неизбежна: пользователи всегда будут пытаться находить уязвимости и хаки, 
	а модераторы реализовывать новые алгоритмы.
	
	В примере ниже можно ознакомиться с работой разных алгоритмов по выявлению наличия контактных данных в сообщениях
	пользователей. Это актуально в первую очередь для торговых площадок и других онлайн площадок по продаже и
	рекомендации товаров и услуг. Актуально это потому, что пользователи не всегда желают платить комиссию за работу
	сервиса и пытаются осуществлять сделки напрямую, минуя сервис.
	
	В данном примере сообщения пользователей подвергаются проверке тремя алгоритмами по поиску контактных данных:
	 - регулярные выражения (regex)
	 - TF-IDF, на основе частотности слов 
	 - нейросеть BERT
	
	1. Регулярные выражения
	Регулярные выражения представляют собой похожий, но гораздо более сильный инструмент для поиска строк, проверки их 
	на соответствие какому-либо шаблону и другой подобной работы. Англоязычное название этого 
	инструмента — Regular Expressions или просто RegExp.
	"""
)

with st.expander(
	label='Блок теории про регулярные выражения'
):
	st.write(
		"""
		В самом общем смысле регулярные выражения — это последовательности символов для поиска соответствий шаблону. 
		Они являются экземплярами регулярного языка и широко применяются для парсинга текста или валидации входных строк.

		Представьте лист картона, в котором вырезаны определенные фигуры. И только фигуры, точно соответствующие вырезам, 
		смогут через них пройти. В данном случае лист картона аналогичен строке регулярного выражения.
		"""
	)
	st.image(
		image='images/re.jpeg',
		caption='Суть работы регулярных выражений',
		use_column_width=True
	)

	st.write(
		"""
		Несколько случаев применения регулярных выражений:
		
		- парсинг входных данных, например текста, логов, веб-информации и т.д.;
		- валидация пользовательского ввода;
		- тестирование результатов вывода;
		- точный поиск текста;
		- реструктуризация данных.
		
		Регулярные выражения отлично подходят, когда есть четкий формат и структура данных. В нашем же случае пользователям
		легко будет обмануть систему модерации сообщений, если она будет построена только на регулярных выражениях.
		Нужно что-то посложнее.
		"""
	)

st.write(
	"""
	2. TF-IDF (TF — term frequency, IDF — inverse document frequency).
	Мера TF-IDF является произведением двух сомножителей TF и IDF.
	
	TF - частота слова - отношение числа вхождений некоторого слова к общему числу слов документа. 
	Таким образом, оценивается важность слова в пределах отдельного документа.
	
	IDF - обратная частота документа - инверсия частоты, с которой некоторое слово встречается в документах коллекции. 
	Учёт IDF уменьшает вес широкоупотребительных слов. Для каждого уникального слова в пределах конкретной коллекции 
	документов существует только одно значение IDF.
	"""
)

with st.expander(
	label='Блок теории про TF-IDF'
):

	st.image(
		image='images/tf_idf_formula.jpg',
		caption='Формула TF-IDF',
		use_column_width=True
	)

	st.write(
		"""
		TF рассчитывается по следующей формуле:
		"""
	)

	st.image(
		image='images/tf_formula.jpg'
	)

	st.write(
		"""
		где t (от англ. term) — количество употребления слова, а n — общее число слов в тексте.
		"""
	)

	st.image(
		image='images/idf_formula.jpg'
	)

	st.write(
		"""
		где D - общее число текстов в корпусе, d - количество текстов, в которых это слово встречается.
				
		IDF нужна в формуле, чтобы уменьшить вес слов, наиболее распространённых в любом другом тексте заданного корпуса.
		"""
	)

	st.write(
		"""
		TF-IDF оценивает значимость слова в документе, на основе данных о всей коллекции документов. Данная мера 
		определяет вес слова за величину пропорциональную частоте его вхождения в документ и обратно пропорциональную 
		частоте его вхождения во всех документах коллекции.
		
		Большая величина TF-IDF говорит об уникальности слова в тексте по отношению к корпусу. 
		Чем чаще оно встречается в конкретном тексте и реже в остальных, тем выше значение TF-IDF.
		"""
	)

st.write(
	"""
	3. Нейросеть BERT.
	
	BERT — это нейронная сеть от Google, показавшая с большим отрывом state-of-the-art результаты на целом ряде задач. 
	С помощью BERT можно создавать программы с ИИ для обработки естественного языка: отвечать на вопросы, заданные 
	в произвольной форме, создавать чат-ботов, автоматические переводчики, анализировать текст и так далее.
	"""
)

with st.expander(
	label='Блок теории про BERT'
):
	st.write(
		"""
		Чтобы подавать на вход нейронной сети текст, нужно его как-то представить в виде чисел. Проще всего это делать 
		побуквенно, подавая на каждый вход нейросети по одной букве. Тогда каждая буква будет кодироваться числом 
		от 0 до 32 (плюс какой-то запас на знаки препинания). Это так называемый character-level.
		
		Но гораздо лучше результаты получаются, если мы предложения будем представлять не по одной букве, а подавая на 
		каждый вход нейросети сразу по целому слову (или хотя бы слогами). Это уже будет word-level. Самый простой 
		вариант — составить словарь со всеми существующими словами, и скармливать сети номер слова в этом словаре. 
		Например, если слово "собака" стоит в этом словаре на 1678 месте, то на вход нейросети для этого слова 
		подаем число 1678.
		
		Вот только в естественном языке при слове "собака" у человека всплывает сразу множество 
		ассоциаций: "пушистая", "злая", "друг человека". Нельзя ли как-то закодировать эту особенность нашего мышления 
		в представлении для нейросети? Оказывается, можно. Для этого достаточно так пересортировать номера слов, чтобы 
		близкие по смыслу слова стояли рядом. Пусть будет, например, для "собака" число 1678, а для слова "пушистая" 
		число 1680. А для слова "чайник" число 9000. Как видите, цифры 1678 и 1680 находятся намного ближе друг к другу,
		чем цифра 9000.
		
		На практике, каждому слову назначают не одно число, а несколько — вектор, скажем, из 32 чисел. И расстояния 
		измеряют как расстояния между точками, на которые указывают эти вектора в пространстве соответствущей 
		размерности (для вектора длиной в 32 числа, это пространство с 32 размерностями, или с 32 осями). 
		Это позволяет сопоставлять одному слову сразу несколько близких по смыслу слов (смотря по какой оси считать). 
		Более того, с векторами можно производить арифметические операции. Классический пример: если из вектора, 
		обозначающего слово "король", вычесть вектор "мужчина" и прибавить вектор для слова "женщина", то получится 
		некий вектор-результат. И он чудесным образом будет соответствовать слову "королева". И действительно, 
		"король — мужчина + женщина = королева". Магия! И это не абстрактный пример, а 
		[реально так происходит](https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/). Учитывая,
		что нейронные сети хорошо приспособлены для математических преобразований над своими входами, видимо это и 
		обеспечивает такую высокую эффективность этого метода.
		
		Идея в основе BERT лежит очень простая: давайте на вход нейросети будем подавать фразы, в которых 15% слов 
		заменим на [MASK], и обучим нейронную сеть предсказывать эти закрытые маской слова.
		
		Например, если подаем на вход нейросети фразу "Я пришел в [MASK] и купил [MASK]", она должна на выходе показать
		слова "магазин" и "молоко". Это упрощенный пример с официальной страницы BERT, на более длинных предложениях 
		разброс возможных вариантов становится меньше, а ответ нейросети однозначнее.
		
		А для того, чтобы нейросеть научилась понимать соотношения между разными предложениями, дополнительно обучим 
		ее предсказывать, является ли вторая фраза логичным продолжением первой. Или это какая-то случайная фраза, не 
		имеющая никакого отношения к первой.
		
		Так, для двух предложений: "Я пошел в магазин." и "И купил там молоко.", нейросеть должна ответить, 
		что это логично. А если вторая фраза будет "Карась небо Плутон", то должна ответить, что это предложение никак 
		не связано с первым. Ниже мы поиграемся с обоими этими режимами работы BERT.
		
		Обучив таким образом нейронную сеть на корпусе текстов из Wikipedia и сборнике книг BookCorpus 
		в течении 4 дней на 16 TPU, получили BERT.
		"""
	)

if st.checkbox('Сгенерировать рандомное сообщение'):
	user_text = st.text_area(
		label='Введите сообщение',
		height=200,
		value=tools.get_random_message(),
		help='Попробуйте указать ссылки на vk, twich, twitter и др. каналы связи а также почту')
else:
	user_text = st.text_area(
		label='Введите сообщение',
		height=200,
		help='Попробуйте указать ссылки на vk, twich, twitter и др. каналы связи а также почту'
	)

with st.expander(
	label='Показать примеры сообщений со скрытыми контактными данными'
):
	st.write(
		"""
		Ма8ш9и9н9а6 в 0хо0ро4ш4е2м9 состоянии
		
		Новый велосипед Работает всё Звонить на 8 девятьсот восемь 1976829
		
		Беспроводная точка доступа маршрутизатор Моя Почта xopkin317 mailru
		
		My Отличный телефон TW практически новый ich хороший экран, без трещин lork не падал ing92
		"""
	)

re_res = tools.get_re_pred(user_text)

if 'Есть контактная информация' in re_res:
	st.success(f'Regex: {re_res}')
else:
	st.error(f'Regex : {re_res}')

tf_idf_res = tools.get_tf_idf_pred(user_text)

if 'Есть контактная информация' in tf_idf_res:
	st.success(f'TF_IDF: {tf_idf_res}')
else:
	st.error(f'TF_IDF: {tf_idf_res}')

bert_res = tools.get_bert_prediction(user_text)

if 'Есть контактная информация' in bert_res:
	st.success(f'BERT: {bert_res}')
else:
	st.error(f'BERT: {bert_res}')

with st.form(key='quiz'):
	right_answers_count = 0

	st.write('QUIZ')

	answer = st.radio(
		label='Что такое регулярные выражения?',
		options=[
			'Модель машинного обучения',
			'Аналог TF-IDF',
			'Инструмент проверки строк на соответствие какому-либо шаблону',
			'Инструмент для классификации сообщений пользователя',
			'Выражения, которые регулярно используются разработчиками',
			'WEB фреймворк',
		]
	)

	if answer == 'Инструмент проверки строк на соответствие какому-либо шаблону':
		right_answers_count += 1

	answer = st.radio(
		label='Как пользователи обходят правила модерации сервиса?',
		options=[
			'Пишут в поддержку',
			'Изменяют сообщения, маскируя запрещенный контент',
			'Записывают голосовые сообщения',
			'Пользуются другими сервисами, без модерации'
		]
	)

	if answer == 'Изменяют сообщения, маскируя запрещенный контент':
		right_answers_count += 1

	answer = st.radio(
		label='Что такое TF-IDF?',
		options=[
			'Вид регулярных выражения',
			'Система модерации текстовых сообщений',
			'Запчасть автомобиля',
			'Мера оценки значимости слова в документе',
			'Модель машинного обучения',
			'Корпус текстов',
		]
	)

	if answer == 'Мера оценки значимости слова в документе':
		right_answers_count += 1

	answer = st.radio(
		label='Что оценивает TF-IDF?',
		options=[
			'Нужно ли отправлять сообщение на модерацию или нет',
			'Значимость слова в документе',
			'Частоту слова',
			'Обратную частоту слова в документе'
		]
	)

	if answer == 'Значимость слова в документе':
		right_answers_count += 1

	answer = st.radio(
		label='Что такое BERT?',
		options=[
			'Персонаж из мультика "Улица Сезам"',
			'Нейронная сеть от Google',
			'Система модерации сообщений',
			'Система оценки соответствия сообщений правилам организации и законам',
			'Вид регулярных выражений'
		]
	)

	if answer == 'Нейронная сеть от Google':
		right_answers_count += 1

	answer = st.radio(
		label='Как обучается BERT?',
		options=[
			'На GPU',
			'Никак, Google уже обучила ее, нам остается только пользоваться готовой',
			'Маскируя 15% слов символом [MASK] и пытаясь предсказать спрятанные слова'
		]
	)

	if answer == 'Маскируя 15% слов символом [MASK] и пытаясь предсказать спрятанные слова':
		right_answers_count += 1

	answer = st.radio(
		label='В каком виде подается информация на вход нейросети BERT?',
		options=[
			'Как есть без изменений',
			'В виде векторов с числами, обозначающими целевое слово и близких к нему по смыслу из словаря',
			'В виде сконкатенированных строк всего обучающего датасета',
			'В виде списка текстов'
		]
	)

	if answer == 'В виде векторов с числами, обозначающими целевое слово и близких к нему по смыслу из словаря':
		right_answers_count += 1

	answer = st.radio(
		label='BERT учитывает контекст в предложениях?',
		options=[
			'Нет',
			'Да'
		]
	)

	if answer == 'Да':
		right_answers_count += 1

	res = st.form_submit_button()

if res:
	st.info(f'Количество правильных ответов {right_answers_count} из 8.')
	if right_answers_count <= 6:
		st.warning('Для прохождения блока необходимо правильно ответить хотя бы на 7 вопросов.')
	else:
		st.success('Отлично! Блок пройден.')
