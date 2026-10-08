"""Ruta de aprendizaje: unidades ordenadas que el profesor imparte una tras otra.

Cada unidad enseña UNA cosa concreta (p. ej. presente simple con he/she/it) con su propio vocabulario, una
explicación corta, ejercicios guiados y una tarea de conversación. No se pasa a la siguiente unidad hasta que
la actual está dominada (ver `MASTERY` y `unit_status`); las unidades ya dominadas se repasan de vez en cuando.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Unit:
    key: str
    skill: str  # clave de GRAMMAR (dominio gramatical y evaluador)
    title: str
    title_es: str
    goal: str  # lo que el alumno sabrá hacer ("can-do")
    explain_es: str  # explicación que el profesor dice en español (corta: se pronuncia)
    form: str  # esquema de la estructura
    examples: tuple[str, ...]
    watch: tuple[tuple[str, str], ...]  # errores típicos de hispanohablantes (mal, bien)
    vocab: tuple[tuple[str, str], ...]  # 8 palabras / expresiones de la unidad (término, significado)
    drills: tuple[str, ...]  # ejercicios guiados: lo que el profesor pide, uno por turno
    production: str  # tarea de conversación libre para usar lo aprendido
    scenarios: tuple[str, ...]  # contextos (curriculum.SCENARIOS) que se van alternando entre clases
    detect: str  # regex: el alumno ha intentado la estructura (cuenta como uso si no hay error)


ROUTE: tuple[Unit, ...] = (
    Unit(
        "present_simple", "present_simple", "Present simple: routines (I / he / she)", "Presente simple: rutinas",
        "Talk about your routine and other people's routines, with he/she + -s.",
        "Para rutinas y cosas que pasan siempre usamos el presente simple. Con he, she o it el verbo lleva -s: "
        "she works. Para negar y preguntar: doesn't y does.",
        "I/you/we/they work · he/she/it works · don't/doesn't + verb · Do you…? / Does she…?",
        ("I get up at seven on weekdays.", "My brother works in a hospital.", "She doesn't drive; she takes the bus."),
        (("he work", "he works"), ("she don't", "she doesn't"), ("does she works", "does she work"), ("I am work", "I work")),
        (("to get up", "levantarse"), ("to commute", "ir y volver del trabajo"), ("on weekdays", "entre semana"),
         ("to be in charge of", "estar a cargo de"), ("to deal with", "ocuparse de"), ("schedule", "horario"),
         ("a day off", "un día libre"), ("to run late", "ir con retraso")),
        ("What time do you usually get up on weekdays?",
         "How do you commute to work? Tell me in one full sentence.",
         "Tell me about one person in your family: what does he or she do every day?",
         "What does your manager or a colleague deal with at work?",
         "Change this sentence to talk about your best friend: 'I live near the city centre.' Start with 'My friend…'.",
         "What do you do on your day off?",
         "Make this sentence negative: 'My boss works on Saturdays.'",
         "Who is in charge of the shopping at home? What does that person do?",
         "Ask me a question about my schedule. Start with 'Do you…' or 'What time do you…'.",
         "What does a doctor do? Tell me two things."),
        "Real conversation about the learner's daily routine and the routine of someone close to them (partner, "
        "family member or colleague): times, places, who does what. Make them use he/she often.",
        ("intro_yourself", "job_description", "family"),
        r"\b(?:he|she|it|my \w+)\s+(?:doesn't|does|(?!was\b|is\b|has\b|this\b|its\b)\w+s)\b"
        r"|\b(?:i|we|they|you)\s+(?:usually |always |often |never |sometimes )?(?:get|go|work|start|finish|take|have|live|commute|don't)\b",
    ),
    Unit(
        "frequency", "frequency_adverbs", "How often? Adverbs of frequency", "¿Con qué frecuencia? Adverbios",
        "Say how often you do things, with the adverb in the right place.",
        "Always, usually, often, sometimes y never van antes del verbo: I usually walk. Pero después de be: "
        "I'm always tired. Para preguntar: How often do you…?",
        "I always/usually/often/sometimes/never + verb · be + always… · once a week, twice a month",
        ("I usually cook at home.", "She's never late for work.", "We go to the gym twice a week."),
        (("I go always", "I always go"), ("I never am late", "I'm never late"),
         ("one time a week", "once a week"), ("two times a week", "twice a week")),
        (("once a week", "una vez a la semana"), ("twice a month", "dos veces al mes"), ("hardly ever", "casi nunca"),
         ("to work out", "hacer ejercicio"), ("to eat out", "comer fuera"), ("to catch up with", "ponerse al día con"),
         ("habit", "hábito"), ("every other day", "un día sí y otro no")),
        ("How often do you work out? Use an adverb like usually or sometimes.",
         "Tell me something you hardly ever do.",
         "Put the word in the right place: 'I am late for work.' Add 'never'.",
         "How often do you eat out? Answer with 'once a…' or 'twice a…'.",
         "Tell me one good habit you have and how often you do it.",
         "What do you always do on Sunday mornings?",
         "How often do you catch up with your old friends?",
         "Put the word in the right place: 'My sister calls me.' Add 'often'.",
         "Ask me how often I do something.",
         "Tell me something you do every other day."),
        "Conversation about habits, health and free time: what they do, how often and why. Ask 'How often…?' a lot.",
        ("health_routine", "hobbies", "weather_smalltalk"),
        r"\b(?:always|usually|often|sometimes|never|hardly ever|once a|twice a|every other|\w+ times a)\b",
    ),
    Unit(
        "present_continuous", "present_continuous", "Present continuous: now and these days", "Presente continuo",
        "Talk about what is happening now and temporary situations, and contrast it with routines.",
        "Para algo que pasa ahora o estos días usamos be más el verbo con -ing: I'm working on a new project. "
        "Para rutinas seguimos con el presente simple.",
        "am/is/are + verb-ing · I'm not… · Are you…? · now, at the moment, these days, this week",
        ("I'm working on a big project this month.", "My sister is living in London at the moment.",
         "Usually I drive, but this week I'm taking the train."),
        (("I working", "I'm working"), ("I am work now", "I'm working now"), ("Are you work?", "Are you working?"),
         ("I'm knowing", "I know")),
        (("at the moment", "en este momento"), ("these days", "estos días"), ("to work on", "trabajar en (algo)"),
         ("to look for", "buscar"), ("to get ready", "prepararse"), ("to try to", "intentar"),
         ("to get used to", "acostumbrarse a"), ("busy", "ocupado")),
        ("What are you doing at the moment, apart from talking to me?",
         "What project are you working on these days?",
         "Tell me something different you're doing this week. Start with 'Usually I…, but this week I'm…'.",
         "What is someone in your family doing right now, do you think?",
         "Correct this sentence: 'I working from home today.'",
         "Are you looking for anything new these days? A flat, a job, a hobby?",
         "Tell me something you're trying to learn or get used to.",
         "Ask me what I'm doing these days.",
         "Why are you so busy this month? Use 'I'm …ing'.",
         "Describe what people are probably doing in your office right now."),
        "Conversation about what's going on in the learner's life and work these days, contrasting with their usual routine.",
        ("work_day", "pro_smalltalk", "weather_smalltalk"),
        r"\b(?:am|is|are|'m|'s|'re)\s+(?:not\s+)?(?:\w+ing)\b",
    ),
    Unit(
        "past_simple", "past_simple", "Past simple: what happened", "Pasado simple",
        "Tell what you did, with regular (-ed) and the most common irregular verbs.",
        "Para algo terminado en el pasado usamos el pasado simple. Muchos verbos añaden -ed, worked, pero otros "
        "son irregulares: go, went; have, had; make, made.",
        "worked, played · went, had, made, saw, got, took, met, came · yesterday, last week, ago",
        ("Yesterday I worked until six.", "Last weekend we went to the beach.", "I met my friends two days ago."),
        (("yesterday I go", "yesterday I went"), ("I have gone there last year", "I went there last year"),
         ("I goed", "I went"), ("I buyed", "I bought")),
        (("to hang out", "pasar el rato / quedar"), ("to stay in", "quedarse en casa"), ("to get back", "volver"),
         ("to have a great time", "pasarlo muy bien"), ("on the way", "de camino"), ("to spend time", "pasar tiempo"),
         ("ago", "hace (tiempo)"), ("in the end", "al final")),
        ("What did you do yesterday evening?",
         "Tell me three things you did last weekend.",
         "Change this to the past: 'I go to work by bus.' Start with 'Yesterday…'.",
         "When did you last hang out with friends? What did you do?",
         "What time did you get back home yesterday?",
         "Tell me about something you bought recently.",
         "Where did you go on your last holiday? Did you have a great time?",
         "Tell me something that happened on the way to work this week.",
         "Did you stay in or go out last Friday? What did you do?",
         "What did you have for dinner two days ago?"),
        "The learner tells the story of last weekend or a recent day: what happened, who they were with, how it ended.",
        ("weekend", "work_day", "travel_story"),
        r"\b(?:went|had|made|saw|got|took|met|came|bought|did|was|were|said|told|left|felt|spent|\w+ed)\b",
    ),
    Unit(
        "past_questions", "past_simple", "Past simple: questions and negatives (did / didn't)", "Pasado: preguntas y negativas",
        "Ask and answer questions about the past and say what you didn't do.",
        "En pasado, para preguntar y negar usamos did y didn't, y el verbo vuelve a su forma normal: "
        "Did you go? I didn't go. Nunca: did you went.",
        "Did you + verb? · What did you + verb? · I didn't + verb",
        ("Did you sleep well?", "Where did you go last summer?", "I didn't have time for lunch."),
        (("did you went", "did you go"), ("I didn't went", "I didn't go"), ("where you went?", "where did you go?"),
         ("I no go", "I didn't go")),
        (("to miss", "perder (un tren) / echar de menos"), ("to forget", "olvidar"), ("to manage to", "conseguir"),
         ("to turn up", "aparecer, presentarse"), ("on time", "a tiempo"), ("to oversleep", "quedarse dormido"),
         ("to cancel", "cancelar"), ("to make it", "llegar a tiempo / conseguirlo")),
        ("Ask me what I did last weekend.",
         "Tell me one thing you didn't do yesterday, and why.",
         "Correct this question: 'Where did you went on holiday?'",
         "Did you manage to finish everything at work last week?",
         "Ask me two questions about my last holiday.",
         "Make it negative: 'I forgot my keys.'",
         "Have you ever missed a flight or a train? Tell me what happened: what didn't go well?",
         "Did anyone turn up late to a meeting recently? What happened?",
         "Ask me if I overslept today.",
         "Tell me about a plan you cancelled. Why didn't you go?"),
        "Interview game: the learner interviews you about a trip or a past event, then you ask them. Push for "
        "did-questions and didn't-negatives.",
        ("travel_story", "weekend", "explain_delay"),
        r"\b(?:did|didn't|did not)\b",
    ),
    Unit(
        "past_continuous", "past_continuous", "Past continuous: telling stories", "Pasado continuo: contar historias",
        "Tell a story with background (was doing) and events (did).",
        "Para el fondo de una historia usamos was o were más -ing: I was walking home. Para la acción que lo "
        "interrumpe, el pasado simple: when it started to rain.",
        "was/were + verb-ing · …when + past simple · while + was/were -ing",
        ("I was walking home when it started to rain.", "While we were having dinner, the lights went out.",
         "What were you doing at nine last night?"),
        (("I was walk", "I was walking"), ("when I was arrive", "when I arrived"), ("they was", "they were"),
         ("while I walked home, I was seeing", "while I was walking home, I saw")),
        (("suddenly", "de repente"), ("while", "mientras"), ("to happen", "pasar, ocurrir"), ("to notice", "darse cuenta"),
         ("to realise", "darse cuenta (de algo)"), ("to break down", "averiarse"), ("luckily", "por suerte"),
         ("to get lost", "perderse")),
        ("What were you doing at nine o'clock last night?",
         "Finish the sentence: 'I was walking to work when…'.",
         "Tell me about a time something broke down. What were you doing when it happened?",
         "Join the two ideas: 'I was cooking.' and 'My phone rang.'",
         "What was happening around you when you got lost once?",
         "Tell me a funny or strange thing that happened while you were travelling.",
         "Correct this: 'They was watching TV when I arrived.'",
         "What were you doing when you realised it was very late?",
         "Ask me what I was doing yesterday at this time.",
         "Tell me about a lucky moment. Use 'luckily'."),
        "The learner tells you a full story (a trip, a problem, a surprise) with background and events; you ask "
        "what was happening at each moment.",
        ("travel_story", "daily_problem", "weekend"),
        r"\b(?:was|were|wasn't|weren't)\s+\w+ing\b",
    ),
    Unit(
        "going_to", "future_forms", "Future plans: going to & present continuous", "Planes: going to",
        "Talk about plans and arrangements for the coming days and months.",
        "Para planes que ya decidiste usamos going to: I'm going to visit my parents. Si ya está organizado, "
        "con día y hora, también el presente continuo: I'm meeting Ana on Friday.",
        "am/is/are going to + verb · I'm meeting… on Friday · next week, this weekend, tomorrow",
        ("I'm going to start a course next month.", "We're having dinner with friends on Saturday.",
         "Are you going to take any time off?"),
        (("I going to", "I'm going to"), ("I will to go", "I'm going to go"), ("I'm going to going", "I'm going to go"),
         ("tomorrow I go to meet", "tomorrow I'm meeting")),
        (("to look forward to", "tener ganas de"), ("to take time off", "cogerse días libres"), ("to plan", "planear"),
         ("to book", "reservar"), ("to sign up for", "apuntarse a"), ("to save up", "ahorrar"),
         ("next month", "el mes que viene"), ("to be about to", "estar a punto de")),
        ("What are you going to do this weekend?",
         "Tell me about an appointment or meeting you have this week. Use 'I'm …ing on…'.",
         "What are you looking forward to this month?",
         "Are you going to take time off soon? What are you going to do?",
         "Correct this: 'Next year I going to learn to drive.'",
         "Is there anything you want to sign up for? What are you going to do?",
         "Ask me about my plans for tomorrow.",
         "What are you saving up for?",
         "What is someone in your family going to do soon?",
         "Tell me two things you're going to do after this lesson."),
        "Conversation about the learner's plans: this weekend, next holidays, goals for the year. Ask about "
        "arrangements (day, time, who with).",
        ("future_plans", "weekend", "meeting"),
        r"\b(?:going to|gonna)\b|\b(?:am|is|are|'m|'s|'re)\s+\w+ing\s+(?:on|tomorrow|next|this|tonight|at)\b",
    ),
    Unit(
        "will", "future_forms", "Will: predictions, offers and quick decisions", "Will: predicciones y decisiones",
        "Make predictions, offers and promises, and decide things on the spot.",
        "Will sirve para predecir, prometer, ofrecer ayuda o decidir en el momento: I'll call you later. "
        "I think it will rain. En negativo: won't.",
        "will / 'll + verb · won't + verb · I think… will… · I'll + verb (offer, promise)",
        ("I think it will be a busy week.", "Don't worry, I'll help you.", "I won't forget, I promise."),
        (("I will to call", "I'll call"), ("I will going", "I'll go"), ("I think will rain", "I think it will rain"),
         ("I don't will", "I won't")),
        (("probably", "probablemente"), ("to promise", "prometer"), ("I'll get back to you", "te respondo más tarde"),
         ("to take care of", "encargarse de"), ("to sort out", "solucionar"), ("in the future", "en el futuro"),
         ("for sure", "seguro"), ("to make a decision", "tomar una decisión")),
        ("What do you think the weather will be like tomorrow?",
         "I'm very thirsty. Offer to do something for me. Start with 'I'll…'.",
         "Make a prediction about your job in five years.",
         "Your colleague has a problem with the computer. What do you say? Use 'I'll sort it out'.",
         "Promise me something for our next lesson. Use 'I won't…' or 'I'll…'.",
         "Correct this: 'I will to call you tomorrow.'",
         "Do you think people will work from home more in the future? Why?",
         "A client asks a difficult question. What do you say? Use 'I'll get back to you'.",
         "Tell me one thing you'll probably do tonight.",
         "Who will take care of your plants or pets on holiday?"),
        "Light conversation mixing predictions (the future of work, cities, technology) and small role-play "
        "moments where the learner must offer help or promise something.",
        ("client_call", "future_plans", "ask_help"),
        r"\b(?:will|won't|'ll)\b",
    ),
    Unit(
        "comparatives", "comparatives", "Comparing: -er / more / the most", "Comparar: comparativos y superlativos",
        "Compare people, places and things, and say which is the best.",
        "Para comparar: adjetivos cortos con -er y than, cheaper than; largos con more, more expensive than. "
        "Para el máximo: the cheapest, the most expensive. Good, better, the best.",
        "-er than · more … than · the -est · the most … · as … as · better / worse",
        ("The train is faster than the bus.", "My new job is more interesting than my old one.",
         "It's the best restaurant in town."),
        (("more better", "better"), ("more cheap", "cheaper"), ("bigger that", "bigger than"),
         ("the more big", "the biggest")),
        (("crowded", "abarrotado"), ("affordable", "asequible"), ("convenient", "práctico, cómodo"),
         ("cost of living", "coste de vida"), ("by far", "con diferencia"), ("on the other hand", "por otro lado"),
         ("quiet", "tranquilo"), ("worth it", "merecer la pena")),
        ("Compare your city with a village. Which is quieter?",
         "Correct this: 'My phone is more better than yours.'",
         "What is the best place you have ever visited? Why?",
         "Compare the bus and the car for going to work. Which is more convenient?",
         "Is the cost of living higher in your city than in other places?",
         "Tell me about the most crowded place you know.",
         "Compare two of your friends or family members. Use 'as … as' or '-er than'.",
         "Which is better, working from home or at the office? Use 'on the other hand'.",
         "What is by far the most expensive thing you have bought?",
         "Is a long holiday worth it, or are short trips better?"),
        "Discussion comparing options (city vs countryside, two jobs, two holidays) and choosing the best; push for reasons.",
        ("opinions_city", "shopping_return", "travel_story"),
        r"\b(?:\w+er than|more \w+ than|less \w+ than|the \w+est|the most|better|worse|the best|the worst|as \w+ as)\b",
    ),
    Unit(
        "pp_experience", "present_perfect", "Present perfect: life experiences (ever / never)", "Pretérito perfecto: experiencias",
        "Talk about experiences in your life, and give details in the past simple.",
        "Para experiencias de la vida, sin decir cuándo, usamos have más el participio: I've been to Paris. "
        "Have you ever…? Si dices cuándo, pasas al pasado: I went there in 2019.",
        "have/has + participle (been, seen, done, eaten, tried) · Have you ever…? · I've never…",
        ("I've been to Italy twice.", "Have you ever tried sushi?", "I've never seen snow. — Really? I saw it last winter."),
        (("I have went", "I have gone / I went"), ("Did you ever", "Have you ever"), ("I have been in Paris last year", "I went to Paris last year"),
         ("I never have been", "I've never been")),
        (("to try", "probar"), ("abroad", "en el extranjero"), ("once in my life", "una vez en la vida"),
         ("so far", "hasta ahora"), ("to fall in love with", "enamorarse de"), ("unforgettable", "inolvidable"),
         ("to have the chance to", "tener la oportunidad de"), ("experience", "experiencia")),
        ("Have you ever been abroad? Where?",
         "Tell me something you've never done but want to do.",
         "Correct this: 'I have went to London.'",
         "Have you ever tried a very strange food? When did you try it?",
         "Ask me two questions starting with 'Have you ever…'.",
         "How many countries have you visited so far?",
         "Tell me about an unforgettable experience. First with 'I've…', then say when it happened.",
         "Have you ever had the chance to work with people from other countries?",
         "Have you ever fallen in love with a city? Which one?",
         "What is something your best friend has done that you haven't?"),
        "Conversation about life experiences (travel, food, jobs, adventures): 'Have you ever…?' and then details "
        "in the past simple ('When was that? What happened?').",
        ("travel_story", "hobbies", "job_interview"),
        r"\b(?:have|has|'ve|'s|haven't|hasn't)\s+(?:ever |never |just |already )?(?:been|seen|done|gone|had|tried|eaten|visited|made|met|worked|lived|travelled|traveled|fallen|written|taken|heard)\b",
    ),
    Unit(
        "pp_recent", "present_perfect", "Present perfect: just / already / yet (work updates)", "Pretérito perfecto: just, already, yet",
        "Give updates on what you have and haven't done, like in a work meeting.",
        "Para dar novedades usamos have más participio con just, ya mismo; already, ya; y yet, en negativas y "
        "preguntas: I haven't finished yet.",
        "I've just + participle · I've already + participle · I haven't + participle + yet · Have you … yet?",
        ("I've just sent the email.", "We've already finished the first part.", "I haven't called the client yet."),
        (("I already finished", "I've already finished"), ("I have finish", "I've finished"),
         ("I didn't finish yet", "I haven't finished yet"), ("yet I haven't", "I haven't … yet")),
        (("deadline", "fecha límite"), ("on track", "según lo previsto"), ("to finish off", "terminar"),
         ("update", "novedad, actualización"), ("to be behind schedule", "ir con retraso"), ("task", "tarea"),
         ("progress", "progreso"), ("to follow up", "hacer seguimiento")),
        ("Tell me something you've just done today.",
         "What have you already done this week at work?",
         "Tell me something you haven't done yet this week.",
         "Correct this: 'I didn't finish the report yet.'",
         "Give me a quick update on a task. Use 'already' and 'yet'.",
         "Is your project on track, or are you behind schedule? What have you done so far?",
         "Ask me if I have finished something yet.",
         "Have you followed up with anyone this week?",
         "What progress have you made in English this month?",
         "What has your team just finished?"),
        "Role-play: you are the learner's manager in a quick 1:1 update meeting; ask what they've done, what they "
        "haven't done yet and what's next.",
        ("project_update", "explain_delay", "meeting"),
        r"\b(?:have|has|'ve|'s|haven't|hasn't)\s+(?:just |already |not |never )?\w+(?:ed|en|ne|un|wn|ght|t)\b|\byet\b",
    ),
    Unit(
        "for_since", "present_perfect", "How long? Present perfect with for and since", "¿Desde cuándo? for y since",
        "Say how long something has been true: 'I've lived here for five years / since 2019'.",
        "Para algo que empezó en el pasado y sigue ahora usamos have más participio: I've lived here for five years. "
        "For con una cantidad de tiempo; since con el momento en que empezó. Nunca: I live here since 2019.",
        "have/has + participle + for (two years) / since (2019, last May) · How long have you…?",
        ("I've worked here for three years.", "She's lived in Madrid since 2018.", "How long have you known your best friend?"),
        (("I live here since 2019", "I've lived here since 2019"), ("since three years", "for three years"),
         ("I am working here for two years", "I've worked here for two years"), ("How long are you living", "How long have you lived")),
        (("how long", "cuánto tiempo"), ("since", "desde"), ("for ages", "desde hace mucho"), ("to move to", "mudarse a"),
         ("to know someone", "conocer a alguien"), ("lately", "últimamente"), ("to belong to", "pertenecer a"),
         ("colleague", "compañero de trabajo")),
        ("How long have you lived in your home?",
         "How long have you worked in your current job? Use 'for' or 'since'.",
         "Correct this: 'I live in this city since 2015.'",
         "How long have you known your best friend?",
         "Say it with 'for': 'I've had this phone since …'. How long is that?",
         "Ask me how long I have been a teacher.",
         "What have you done lately in your free time?",
         "Tell me something you haven't done for ages.",
         "How long have you been interested in English?",
         "Do you belong to any group or club? How long have you been a member?"),
        "Getting-to-know-you conversation focused on duration: home, job, friends, hobbies ('How long…?', 'Since when…?').",
        ("intro_yourself", "job_interview", "hobbies"),
        r"\b(?:for \w+ (?:years?|months?|weeks?|days?|ages)|since \w+|how long have)\b",
    ),
    Unit(
        "modals_advice", "modal_verbs", "Advice and obligation: should / have to / don't have to", "Consejos y obligaciones",
        "Give advice and talk about rules and obligations at work and in life.",
        "Para dar consejos: should, I think you should rest. Para obligaciones: have to, I have to wear a uniform. "
        "Don't have to significa que no es necesario, no que esté prohibido.",
        "should / shouldn't + verb · have to / has to + verb · don't have to (not necessary) · must (strong)",
        ("You should take a break.", "I have to start at eight.", "We don't have to wear a suit on Fridays."),
        (("you should to go", "you should go"), ("I must to", "I must / I have to"), ("she have to", "she has to"),
         ("you don't must", "you don't have to / you mustn't")),
        (("to take a break", "tomarse un descanso"), ("to get some rest", "descansar"), ("rule", "norma"),
         ("to be allowed to", "tener permiso para"), ("dress code", "código de vestimenta"), ("to avoid", "evitar"),
         ("advice", "consejo"), ("stressed out", "muy estresado")),
        ("I'm always stressed out at work. What should I do?",
         "What do you have to do every day at work?",
         "Tell me something you don't have to do at your job.",
         "Correct this: 'You should to drink more water.'",
         "Is there a dress code at your work? What do people have to wear?",
         "Give advice to someone who wants to learn English fast.",
         "What should people avoid before going to sleep?",
         "Are you allowed to work from home? What rules do you have to follow?",
         "What does your manager have to do that you don't?",
         "Ask me for advice about something."),
        "The learner gives advice for typical problems (stress, sleep, a new job) and explains the rules of their "
        "workplace; swap roles once.",
        ("daily_problem", "health_routine", "meeting"),
        r"\b(?:should|shouldn't|have to|has to|had to|don't have to|doesn't have to|must|mustn't|allowed to)\b",
    ),
    Unit(
        "polite_requests", "polite_requests", "Polite requests: could you / would you mind / can I", "Peticiones educadas",
        "Ask for things and favours politely at work, in shops and restaurants.",
        "Para pedir con educación: Could you…?, Can I…?, Could I have…? Y would you mind con el verbo en -ing: "
        "Would you mind closing the door? Sin please al final queda brusco.",
        "Could you + verb…? · Can/Could I + verb…? · Would you mind + verb-ing…? · I'd like…",
        ("Could you send me the file, please?", "Would you mind helping me with this?", "Could I have the bill, please?"),
        (("would you mind to help", "would you mind helping"), ("give me a coffee", "could I have a coffee, please?"),
         ("can you to", "can you"), ("I want the bill", "could I have the bill, please?")),
        (("would you mind", "¿te importaría…?"), ("to give someone a hand", "echar una mano"), ("I'd like", "me gustaría"),
         ("to borrow", "pedir prestado"), ("to lend", "prestar"), ("the bill", "la cuenta"), ("refund", "reembolso"),
         ("as soon as possible", "lo antes posible")),
        ("Ask me to send you a document. Be polite.",
         "You need help moving a table. Ask me using 'Would you mind…'.",
         "Correct this: 'Would you mind to open the window?'",
         "You are in a restaurant. Ask for the bill.",
         "Ask a colleague to lend you a charger.",
         "Ask me to give you a hand with a report, as soon as possible.",
         "You bought broken headphones. Ask the shop for a refund politely.",
         "Ask if you can borrow my pen.",
         "Order a coffee and something to eat. Start with 'I'd like…' or 'Could I have…'.",
         "Ask me to repeat something more slowly, politely."),
        "Role-play: you are a waiter, a shop assistant or a busy colleague; the learner has to ask for several "
        "things and solve a small problem politely.",
        ("restaurant", "ask_help", "shopping_return"),
        r"\b(?:could you|can you|could i|can i|would you mind|would you|i'd like|may i)\b",
    ),
    Unit(
        "used_to", "used_to", "Used to: how things were before", "Used to: costumbres del pasado",
        "Talk about past habits and how your life has changed.",
        "Used to más verbo cuenta costumbres del pasado que ya no haces: I used to play football. En negativa y "
        "pregunta: didn't use to, did you use to.",
        "used to + verb · didn't use to + verb · Did you use to…? · not any more / anymore",
        ("I used to play football every weekend.", "We didn't use to have smartphones.", "Did you use to live in a village?"),
        (("I use to play", "I used to play"), ("I used to playing", "I used to play"), ("I am used to play (past habit)", "I used to play"),
         ("I didn't used to", "I didn't use to")),
        (("childhood", "infancia"), ("to grow up", "crecer, criarse"), ("anymore", "ya no"), ("nowadays", "hoy en día"),
         ("to change", "cambiar"), ("to remember", "recordar"), ("back then", "en aquella época"),
         ("to give up", "dejar (un hábito)")),
        ("What did you use to do after school when you were a child?",
         "Tell me something you used to do but don't do anymore.",
         "Correct this: 'When I was young I use to watch cartoons.'",
         "Where did you grow up? What was it like back then?",
         "Tell me something you didn't use to like but like nowadays.",
         "Ask me what I used to do when I was a child.",
         "How has your city changed? Use 'There used to be…'.",
         "Is there a habit you gave up? What did you use to do?",
         "What did people use to do before smartphones?",
         "What's a good memory from your childhood?"),
        "Conversation about the learner's childhood and how life has changed (city, technology, habits): then vs now.",
        ("health_routine", "family", "opinions_city"),
        r"\b(?:used to|use to|didn't use|did you use)\b",
    ),
    Unit(
        "first_conditional", "conditionals", "First conditional: if + present, will", "Primer condicional",
        "Talk about real possibilities and their results.",
        "Para posibilidades reales: if más presente y will en la otra parte. If it rains, I'll stay at home. "
        "Ojo: después de if no va will.",
        "If + present simple, … will + verb · … will …, if + present · unless = if not",
        ("If it rains tomorrow, I'll stay at home.", "If I finish early, I'll call you.", "We'll miss the train if we don't hurry."),
        (("if it will rain", "if it rains"), ("if I will have time", "if I have time"), ("if I finish, I call you", "if I finish, I'll call you"),
         ("if you don't hurry, you miss", "if you don't hurry, you'll miss")),
        (("unless", "a menos que"), ("as long as", "siempre que"), ("to hurry", "darse prisa"), ("in case", "por si acaso"),
         ("to get a promotion", "conseguir un ascenso"), ("to be late", "llegar tarde"), ("to pass", "aprobar"),
         ("to make plans", "hacer planes")),
        ("What will you do if it rains this weekend?",
         "Finish the sentence: 'If I finish work early today…'.",
         "Correct this: 'If it will be sunny, we go to the beach.'",
         "What will happen if you don't hurry in the morning?",
         "If you get a promotion this year, what will you do?",
         "Tell me one thing you'll do unless something goes wrong.",
         "Ask me what I'll do if I have a free afternoon.",
         "What will you do if you pass your next English exam?",
         "Finish this: 'I'll go to the party as long as…'.",
         "If your train is cancelled tomorrow, how will you get to work?"),
        "Planning conversation full of possibilities: weekend plans, work situations, 'What will you do if…?'.",
        ("future_plans", "meeting", "client_call"),
        r"\bif\b.+\b(?:will|'ll|won't)\b|\b(?:will|'ll|won't)\b.+\bif\b|\bunless\b",
    ),
    Unit(
        "second_conditional", "conditionals", "Second conditional: if I had…, I would…", "Segundo condicional",
        "Imagine unreal or unlikely situations and say what you would do.",
        "Para situaciones imaginarias: if más pasado y would en la otra parte. If I had more time, I would travel. "
        "Con be se dice If I were you.",
        "If + past simple, … would + verb · If I were you, I'd… · I'd / I wouldn't…",
        ("If I won the lottery, I'd buy a house by the sea.", "If I were you, I'd ask for help.",
         "What would you do if you lived abroad?"),
        (("if I would have", "if I had"), ("if I have money, I would", "if I had money, I would"),
         ("I would to travel", "I would travel"), ("if I was you", "if I were you")),
        (("to win the lottery", "ganar la lotería"), ("to quit", "dejar (un trabajo)"), ("I'd rather", "preferiría"),
         ("to afford", "permitirse (pagar)"), ("dream", "sueño"), ("to make the most of", "aprovechar al máximo"),
         ("to live abroad", "vivir en el extranjero"), ("it depends on", "depende de")),
        ("What would you do if you won the lottery?",
         "If you could live in any country, where would you live? Why?",
         "Correct this: 'If I would have more time, I would read more.'",
         "Give me advice. Start with 'If I were you, I'd…': I'm always tired.",
         "If you quit your job tomorrow, what would you do?",
         "What would you buy if you could afford anything?",
         "Ask me what I would do if I didn't work.",
         "Would you rather live in the city or by the sea? Why?",
         "If you had one free month, how would you make the most of it?",
         "What would you change about your city if you were the mayor?"),
        "Fun hypothetical questions; push for reasons and for 'would' in every answer.",
        ("hypothetical", "future_plans", "negotiation"),
        r"\b(?:would|'d|wouldn't)\b",
    ),
    Unit(
        "opinions", "giving_opinions", "Giving opinions: agree, disagree and explain why", "Dar tu opinión",
        "Give your opinion, agree or disagree politely and justify it with reasons and examples.",
        "Para opinar: I think, in my opinion. Para estar de acuerdo se dice I agree, nunca I am agree. "
        "Y siempre explica por qué: because, for example.",
        "I think / In my opinion… because… · I agree / I don't agree · On the one hand… on the other hand… · For example…",
        ("In my opinion, working from home is better because I save time.", "I agree with you, but it depends on the job.",
         "I'm not sure about that. For example…"),
        (("I am agree", "I agree"), ("in my opinion I think", "in my opinion / I think"), ("depends of", "depends on"),
         ("for me is better", "for me, it's better")),
        (("in my opinion", "en mi opinión"), ("to agree with", "estar de acuerdo con"), ("to be against", "estar en contra de"),
         ("on the other hand", "por otro lado"), ("advantage", "ventaja"), ("drawback", "desventaja"),
         ("for example", "por ejemplo"), ("it depends on", "depende de")),
        ("What do you think about working from home? Give one reason.",
         "Correct this: 'I am agree with you.'",
         "Tell me one advantage and one drawback of living in a big city.",
         "I think English is easy. Do you agree? Why or why not?",
         "Are you against or in favour of using phones at school? Why?",
         "Give your opinion about social media, with an example.",
         "Ask me what I think about something.",
         "Is it better to rent or to buy a house? It depends on…",
         "Disagree with me politely: 'Summer is the best season.'",
         "What do you think about four-day working weeks?"),
        "Friendly debate on 2-3 topics (remote work, cities, technology); ask for reasons, examples and the other side.",
        ("opinions_city", "meeting", "presentation"),
        r"\b(?:i think|in my opinion|i agree|i don't agree|i disagree|on the other hand|because|for example|it depends)\b",
    ),
    Unit(
        "verb_patterns", "verb_patterns", "Verb patterns: enjoy doing / want to do", "Verbo + -ing o to + verbo",
        "Use the right form after common verbs: enjoy cooking, decide to go, look forward to seeing.",
        "Después de algunos verbos va -ing: enjoy, finish, avoid, mind: I enjoy cooking. Después de otros va to "
        "más verbo: want, decide, need, hope: I want to travel.",
        "enjoy / finish / avoid / don't mind + -ing · want / decide / need / hope / plan + to + verb · look forward to + -ing",
        ("I enjoy cooking for my friends.", "We decided to stay at home.", "I'm looking forward to seeing you."),
        (("I enjoy to cook", "I enjoy cooking"), ("I want go", "I want to go"), ("I look forward to see you", "I look forward to seeing you"),
         ("I avoid to drive", "I avoid driving")),
        (("to enjoy", "disfrutar"), ("to avoid", "evitar"), ("to decide", "decidir"), ("I don't mind", "no me importa"),
         ("to keep", "seguir (haciendo)"), ("to be good at", "ser bueno en"), ("to be tired of", "estar cansado de"),
         ("to hope", "esperar (con deseo)")),
        ("What do you enjoy doing at the weekend?",
         "Correct this: 'I enjoy to watch films.'",
         "Tell me something you decided to do this year.",
         "What do you try to avoid doing at work?",
         "What are you good at? Use 'I'm good at …ing'.",
         "What are you tired of doing every day?",
         "Tell me something you hope to do in the future.",
         "What don't you mind doing at home? And what do you hate doing?",
         "What are you looking forward to doing next month?",
         "Ask me what I enjoy doing."),
        "Conversation about likes, dislikes, decisions and plans, pushing for the verbs of the unit.",
        ("hobbies", "future_plans", "job_interview"),
        r"\b(?:enjoy|avoid|finish|mind|keep|good at|tired of|forward to)\s+\w+ing\b|\b(?:want|decide|decided|need|hope|plan|try|would like)\s+to\s+\w+",
    ),
    Unit(
        "questions", "question_formation", "Asking good questions (direct and polite)", "Hacer buenas preguntas",
        "Ask clear questions with the right word order, including polite indirect questions.",
        "En las preguntas el auxiliar va antes del sujeto: Where do you work? Pero en preguntas educadas como "
        "Could you tell me…, el orden es normal: Could you tell me where the station is?",
        "Wh- + do/does/did + subject + verb? · Who works…? (subject question) · Could you tell me / Do you know + where … is?",
        ("Where do you work?", "Who called you this morning?", "Could you tell me what time it starts?"),
        (("where you work?", "where do you work?"), ("what means this?", "what does this mean?"),
         ("could you tell me where is the station?", "could you tell me where the station is?"),
         ("who did call you?", "who called you?")),
        (("could you tell me", "¿podría decirme…?"), ("do you know", "¿sabes…?"), ("I was wondering", "me preguntaba"),
         ("to find out", "averiguar"), ("to wonder", "preguntarse"), ("directions", "indicaciones"),
         ("to get to", "llegar a"), ("to mean", "significar")),
        ("Ask me three questions about my job.",
         "Correct this question: 'What means this word?'",
         "You are lost in a city. Ask me politely where the station is, starting with 'Could you tell me…'.",
         "Ask me who helped me today.",
         "Ask me a polite question starting with 'Do you know…'.",
         "Ask me how I get to work.",
         "You want to find out the price of a course. Ask politely.",
         "Correct this: 'Where you live?'",
         "Ask me about my last holiday: three different question words.",
         "Start with 'I was wondering if…' and ask me for a favour."),
        "Role reversal: the learner interviews you (about your job, your city, a trip) and then asks for "
        "information politely, as if calling a company.",
        ("pro_smalltalk", "client_call", "intro_yourself"),
        r"\?|\b(?:could you tell me|do you know|i was wondering)\b|^(?:what|where|when|why|who|how|do|does|did|are|is|can|could|would|have)\b",
    ),
)

ROUTE_BY_KEY = {u.key: u for u in ROUTE}

# Preguntas abiertas para la conversación real (paso 4): obligan a usar la estructura sin parecer un ejercicio.
TALK: dict[str, tuple[str, ...]] = {
    "present_simple": (
        "Tell me about a normal weekday for you, from morning to evening.",
        "What does your partner or a friend do for work? What does a normal day look like for them?",
        "Who in your family gets up the earliest? What does that person do in the morning?",
        "What does your team do at work? Who deals with the most difficult problems?",
        "How is your schedule different on Fridays?",
        "What does your best friend do at the weekend?"),
    "frequency": (
        "How often do you see your family? And your friends?",
        "What do you usually do to relax after work?",
        "Is there anything you never do, but other people always do?",
        "How often does your city have good weather? What do people usually do then?",
        "What healthy habits do you have? How often do you do them?",
        "What does your best friend often do that you hardly ever do?"),
    "present_continuous": (
        "What's going on in your life these days? Anything new?",
        "What are people in your team working on at the moment?",
        "Are you reading or watching anything good these days?",
        "What is changing in your city at the moment?",
        "What are you trying to improve this month?",
        "How is your week going so far? What are you doing differently?"),
    "past_simple": (
        "Tell me about your last weekend, from Friday evening to Sunday night.",
        "What did you do on your last birthday?",
        "Tell me about a good day you had recently. What happened?",
        "What was the last film or series you watched? What happened in it?",
        "Tell me about your first job. What did you do there?",
        "What did you have for lunch yesterday, and where did you have it?"),
    "past_questions": (
        "Ask me five questions about my last holiday, one by one.",
        "Tell me about a day when things didn't go well. What didn't work?",
        "Did you study English at school? What did you do in class?",
        "Ask me about my weekend and react to my answers.",
        "What did you want to be when you were a child? Did you do it?",
        "Tell me about something you forgot to do recently. What happened?"),
    "past_continuous": (
        "Tell me about a surprise in your life. What were you doing when it happened?",
        "Tell me about a time something went wrong while you were travelling.",
        "Where were you and what were you doing when you heard some big news?",
        "Tell me a funny story from work. What was everyone doing?",
        "What were you doing this time last year?",
        "Tell me about a time you met someone by chance. What were you doing?"),
    "going_to": (
        "What are your plans for the next holidays?",
        "What are you going to do differently next year?",
        "What's your plan for this weekend, hour by hour?",
        "Is anyone in your family going to do something special soon?",
        "What are you going to cook this week?",
        "What are your plans at work for the next few weeks?"),
    "will": (
        "How do you think your city will change in ten years?",
        "What will work be like in the future? Will people still go to offices?",
        "I'm going on a trip and I'm nervous. What will you tell me?",
        "What do you think will happen in your favourite sport this season?",
        "What technology do you think people will use every day in the future?",
        "I have a problem with my laptop before a meeting. What will you do to help me?"),
    "comparatives": (
        "Compare the place where you grew up with the place where you live now.",
        "Which is better for a holiday: the beach or the mountains? Why?",
        "Compare your current job with a previous job or with your studies.",
        "Who is the funniest person in your family? Who is the most serious?",
        "Compare cycling, driving and public transport in your city.",
        "What is the best restaurant you know? Why is it better than others?"),
    "pp_experience": (
        "What's the most interesting place you've ever been to? When did you go?",
        "Have you ever had a really strange job or task? Tell me about it.",
        "What's something new you've tried this year?",
        "Have you ever met someone famous? What happened?",
        "Which countries have you visited? Which one did you like most?",
        "Have you ever done something really brave? Tell me about it."),
    "pp_recent": (
        "Give me an update on your week so far: what have you done and what haven't you done yet?",
        "What has your team achieved this month?",
        "Have you done anything new at home lately?",
        "What have you already planned for the next holidays? What haven't you decided yet?",
        "Imagine I'm your manager: tell me how your project is going.",
        "What have you learned in our classes so far?"),
    "for_since": (
        "Tell me about your home: how long have you lived there? What has changed?",
        "How long have you had your favourite hobby? How did you start?",
        "Who is your oldest friend? How long have you known each other?",
        "How long have you worked with your current team?",
        "Is there anything you've wanted to do for a long time?",
        "How long have you had your phone, your car or your bike? Do you want a new one?"),
    "modals_advice": (
        "A friend wants to visit your city. What should they see and what should they avoid?",
        "What rules do you have to follow at work? Which ones don't you like?",
        "What should someone do on their first day at a new job?",
        "What do you have to do at home every week? What don't you have to do?",
        "I sleep badly. What should I do?",
        "What should companies do to keep their workers happy?"),
    "polite_requests": (
        "Let's role-play: you're in a café and your order is wrong. I'm the waiter.",
        "Let's role-play: you need a colleague to help you with a report today. I'm the colleague.",
        "Let's role-play: you call a hotel to change your booking. I'm the receptionist.",
        "Let's role-play: you're on a train and need to ask a stranger for help.",
        "Let's role-play: you ask your manager for a day off.",
        "Let's role-play: you return a jacket that's too small. I'm the shop assistant."),
    "used_to": (
        "Tell me about your childhood: what did you use to do in summer?",
        "How was your city different when you were a child?",
        "What did you use to eat that you don't eat anymore?",
        "What did people use to do at work before computers?",
        "Did you use to have a hobby that you gave up? Why?",
        "What did your parents use to say to you all the time?"),
    "first_conditional": (
        "What will you do this weekend if the weather is bad? And if it's good?",
        "If you finish work early this week, what will you do?",
        "What will happen at work if a big problem appears on Friday afternoon?",
        "If you go on holiday this year, where will you go?",
        "What will you do if you have more free time next month?",
        "If your best friend visits you, what will you do together?"),
    "second_conditional": (
        "If you could have any job in the world, what would it be?",
        "What would you do if you didn't have to work for a year?",
        "If you could live in another time, when would you live?",
        "If you were the boss of your company, what would you change?",
        "What would you do if you found a wallet in the street?",
        "If you could speak perfect English tomorrow, what would you do first?"),
    "opinions": (
        "What do you think about working from home? What are the advantages and drawbacks?",
        "In your opinion, is it better to live in a big city or a small town?",
        "Do you agree that people spend too much time on their phones?",
        "What do you think is the best way to learn a language?",
        "Should companies have a four-day week? Why or why not?",
        "What do you think about electric scooters in cities?"),
    "verb_patterns": (
        "What do you enjoy doing on holiday, and what do you avoid doing?",
        "What have you decided to change in your life this year?",
        "What are you looking forward to doing this summer?",
        "What do you need to improve at work?",
        "What things don't you mind doing at home, and which do you hate doing?",
        "What would you like to learn to do one day?"),
    "questions": (
        "Interview me about my job: ask me at least five questions.",
        "Let's role-play: you call a language school to get information about a course.",
        "Ask me about my favourite city and react to my answers.",
        "Let's role-play: you're a tourist asking me for directions politely.",
        "Interview me for a job in your company.",
        "Ask me questions to find out what I did last weekend."),
}

# Criterios para dar una unidad por dominada (todos a la vez).
MASTERY = {
    "min_lessons": 2,  # al menos dos clases de la unidad (una no basta para fijarla)
    "min_correct": 12,  # usos correctos de la estructura (lo de clases anteriores cuenta la mitad en cada clase)
    "min_accuracy": 0.75,  # precisión ponderada (más peso a lo reciente)
    "min_last_accuracy": 0.8,  # y en la última clase
    "min_words": 6,  # palabras de la unidad usadas por el alumno (de 8)
}


def unit_index(key: str) -> int:
    for i, u in enumerate(ROUTE):
        if u.key == key:
            return i
    return 0


def accuracy(correct: float, major: float, minor: float) -> float | None:
    total = correct + major + 0.4 * minor
    return None if total <= 0 else correct / total


def unit_status(p: dict, words_used: int) -> dict:
    """Estado de dominio de una unidad a partir de su progreso acumulado (fila de unit_progress)."""
    m = MASTERY
    acc = accuracy(p.get("correct", 0), p.get("major", 0), p.get("minor", 0))
    last = accuracy(p.get("last_correct", 0), p.get("last_major", 0), p.get("last_minor", 0))
    checks = {
        "lessons": p.get("lessons", 0) >= m["min_lessons"],
        "correct": p.get("correct", 0) >= m["min_correct"],
        "accuracy": acc is not None and acc >= m["min_accuracy"],
        "last_accuracy": last is not None and last >= m["min_last_accuracy"],
        "words": words_used >= m["min_words"],
    }
    volume = min(p.get("correct", 0) / m["min_correct"], 1.0)
    score = 100 * (0.45 * volume * (acc or 0) + 0.2 * (last or 0) * min(p.get("lessons", 0) / m["min_lessons"], 1.0)
                   + 0.35 * min(words_used / m["min_words"], 1.0))
    if all(checks.values()):
        score = 100.0
    else:
        score = min(score, 95.0)
    missing = []
    if not checks["correct"]:
        missing.append(f"{m['min_correct'] - p.get('correct', 0)} frases correctas más con la estructura")
    if not checks["accuracy"] or not checks["last_accuracy"]:
        missing.append("menos errores con la estructura (≥80 % de acierto en la clase, ≥75 % en las últimas)")
    if not checks["words"]:
        missing.append(f"usar {m['min_words'] - words_used} palabras más de la unidad")
    if not checks["lessons"]:
        missing.append("una clase más para fijarla")
    return {"mastered": all(checks.values()), "score": round(score), "accuracy": None if acc is None else round(acc, 2),
            "last_accuracy": None if last is None else round(last, 2), "words_used": words_used, "checks": checks,
            "missing": missing}


_STOP = {"the", "and", "you", "your", "tell", "about", "what", "with", "use", "this", "that", "for", "me", "start",
         "one", "two", "three", "sentence", "say", "ask", "like", "do", "does", "did", "are", "how", "can", "it"}


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if len(w) > 2 and w not in _STOP}


def drill_asked(drill: str, tutor_text: str) -> bool:
    """¿El profesor ha planteado este ejercicio (aunque lo diga con otras palabras)?"""
    want = content_words(drill)
    if not want:
        return False
    return len(want & content_words(tutor_text)) / len(want) >= 0.5


def attempts_structure(unit: Unit, text: str) -> bool:
    return re.search(unit.detect, text.lower().replace("’", "'"), re.M) is not None


def error_is_unit_related(unit: Unit, err: dict) -> bool:
    if err.get("skill_key") == unit.skill:
        return True
    original = (err.get("original") or "").lower()
    return any(wrong.lower() in original or original in wrong.lower() for wrong, _ in unit.watch if len(original) > 3)
