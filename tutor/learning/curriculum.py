"""Contenido base: catálogo de escenarios, estructuras gramaticales y plan general por semanas.

El planner no sigue este catálogo en orden: lo usa como "menú" y elige según el perfil del usuario.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Scenario:
    key: str
    domain: str  # everyday | work
    title: str
    situation: str  # instrucción para el tutor
    difficulty: int  # 1 = B1, 2 = B1+, 3 = B2
    grammar_fit: tuple[str, ...]  # estructuras que este escenario hace aparecer de forma natural
    seed_vocab: tuple[tuple[str, str], ...] = field(default_factory=tuple)  # (término, significado_es)
    roleplay: bool = False


SCENARIOS: tuple[Scenario, ...] = (
    # ---------------- Everyday ----------------
    Scenario("intro_yourself", "everyday", "Getting to know each other",
             "Casual chat to get to know the learner: where they live, what they do, what they enjoy.", 1,
             ("present_simple", "question_formation"),
             (("to be into something", "ser aficionado a algo"), ("hometown", "ciudad natal"), ("nowadays", "hoy en día"))),
    Scenario("weekend", "everyday", "Last weekend and next weekend",
             "Talk about what the learner did last weekend and their plans for the next one.", 1,
             ("past_simple", "future_forms"),
             (("to hang out", "quedar / pasar el rato"), ("to catch up", "ponerse al día"), ("laid-back", "tranquilo, relajado"))),
    Scenario("hobbies", "everyday", "Hobbies and free time",
             "Explore the learner's hobbies: how they started, how often, why they like them.", 1,
             ("present_perfect", "frequency_adverbs"),
             (("to take up", "empezar (una afición)"), ("to be keen on", "tener muchas ganas de / gustar"), ("in my spare time", "en mi tiempo libre"))),
    Scenario("family", "everyday", "Family and people close to you",
             "Talk about the learner's family or close friends; describe people and relationships.", 1,
             ("present_simple", "comparatives"),
             (("to get along with", "llevarse bien con"), ("to take after", "parecerse a (un familiar)"), ("close-knit", "muy unido"))),
    Scenario("travel_story", "everyday", "A memorable trip",
             "Ask the learner to tell the story of a memorable trip; dig into details and feelings.", 2,
             ("past_simple", "past_continuous"),
             (("itinerary", "itinerario"), ("to get lost", "perderse"), ("breathtaking", "impresionante"))),
    Scenario("restaurant", "everyday", "Ordering at a restaurant", "Role-play: you are a waiter, the learner orders dinner and has a small problem with the order.", 1,
             ("polite_requests", "modal_verbs"),
             (("to book a table", "reservar mesa"), ("could I get", "¿me podría traer…?"), ("the bill", "la cuenta")), True),
    Scenario("shopping_return", "everyday", "Returning something in a shop",
             "Role-play: the learner returns a faulty product to a shop; you are the shop assistant.", 2,
             ("present_perfect", "polite_requests"),
             (("refund", "reembolso"), ("receipt", "ticket / recibo"), ("faulty", "defectuoso")), True),
    Scenario("weather_smalltalk", "everyday", "Small talk: weather and the week",
             "Light small talk about the weather, the season and how the week is going.", 1,
             ("present_continuous", "future_forms"),
             (("chilly", "fresco, frío"), ("it's pouring", "está diluviando"), ("so far", "hasta ahora"))),
    Scenario("daily_problem", "everyday", "A small everyday problem",
             "The learner explains an everyday problem (a broken appliance, a noisy neighbour…) and how to solve it.", 2,
             ("present_perfect", "modal_verbs"),
             (("to sort out", "solucionar"), ("to put up with", "aguantar, soportar"), ("annoying", "molesto"))),
    Scenario("opinions_city", "everyday", "Opinions: city life vs. countryside",
             "Discuss and compare city life and the countryside; ask for reasons and examples.", 2,
             ("comparatives", "giving_opinions"),
             (("on the other hand", "por otro lado"), ("crowded", "abarrotado"), ("cost of living", "coste de vida"))),
    Scenario("health_routine", "everyday", "Habits and health",
             "Talk about the learner's routines, sleep, exercise and habits they'd like to change.", 2,
             ("frequency_adverbs", "used_to"),
             (("to cut down on", "reducir (consumo)"), ("to work out", "hacer ejercicio"), ("routine", "rutina"))),
    Scenario("future_plans", "everyday", "Dreams and future plans",
             "Talk about the learner's plans for the next year and things they'd love to do one day.", 2,
             ("future_forms", "conditionals"),
             (("to look forward to", "tener ganas de"), ("bucket list", "lista de cosas por hacer"), ("to save up", "ahorrar"))),
    Scenario("hypothetical", "everyday", "What would you do if…?",
             "Fun hypothetical questions (winning the lottery, living abroad…); push for reasons.", 3,
             ("conditionals", "giving_opinions"),
             (("I'd probably", "probablemente…"), ("it depends on", "depende de"), ("to make the most of", "aprovechar al máximo"))),
    # ---------------- Work ----------------
    Scenario("work_day", "work", "How was your day at work?",
             "Natural conversation about the learner's work day: tasks, colleagues, what went well or badly.", 1,
             ("past_simple", "present_continuous"),
             (("deadline", "fecha límite"), ("workload", "carga de trabajo"), ("to be swamped", "estar desbordado"))),
    Scenario("job_description", "work", "Explaining your job",
             "The learner explains their job and responsibilities to someone outside their field.", 1,
             ("present_simple", "present_perfect"),
             (("to be in charge of", "estar a cargo de"), ("to deal with", "ocuparse de / tratar con"), ("responsibilities", "responsabilidades"))),
    Scenario("project_update", "work", "Giving a project update",
             "Role-play: you are the learner's manager in a 1:1; ask for a status update on their current project.", 2,
             ("present_perfect", "present_continuous"),
             (("on track", "en plazo, según lo previsto"), ("milestone", "hito"), ("so far", "hasta ahora")), True),
    Scenario("explain_delay", "work", "Explaining a delay",
             "Role-play: the project is late; the learner explains why to their manager and proposes a new plan.", 2,
             ("past_simple", "present_perfect", "future_forms"),
             (("to postpone", "posponer, aplazar"), ("to run behind schedule", "ir con retraso"), ("bottleneck", "cuello de botella")), True),
    Scenario("meeting", "work", "Team meeting",
             "Role-play a short team meeting: discuss priorities for the week and agree on next steps.", 2,
             ("giving_opinions", "modal_verbs"),
             (("to prioritise", "priorizar"), ("action item", "tarea pendiente / acción"), ("to follow up", "hacer seguimiento")), True),
    Scenario("ask_help", "work", "Asking a colleague for help",
             "Role-play: the learner asks a colleague for help with a problem and explains it clearly.", 1,
             ("polite_requests", "question_formation"),
             (("would you mind", "¿te importaría…?"), ("to give someone a hand", "echar una mano"), ("issue", "problema, incidencia")), True),
    Scenario("job_interview", "work", "Job interview",
             "Role-play: you interview the learner for a job similar to theirs. Ask about experience and achievements.", 3,
             ("present_perfect", "past_simple"),
             (("to be responsible for", "ser responsable de"), ("achievement", "logro"), ("strengths and weaknesses", "fortalezas y debilidades")), True),
    Scenario("client_call", "work", "Phone call with a client",
             "Role-play: a client calls with a complaint; the learner handles the call professionally.", 3,
             ("polite_requests", "future_forms"),
             (("to look into it", "investigarlo, revisarlo"), ("I'll get back to you", "te respondo / te vuelvo a contactar"), ("inconvenience", "molestia")), True),
    Scenario("negotiation", "work", "Negotiating priorities",
             "Role-play: the learner negotiates scope or deadlines with a demanding stakeholder.", 3,
             ("conditionals", "giving_opinions"),
             (("trade-off", "compromiso, contrapartida"), ("to push back", "oponerse, rebatir"), ("realistic", "realista")), True),
    Scenario("presentation", "work", "Presenting an idea",
             "The learner briefly presents an idea or proposal for their team; you ask follow-up questions.", 3,
             ("giving_opinions", "conditionals"),
             (("to sum up", "para resumir"), ("the main benefit", "la principal ventaja"), ("to roll out", "implantar, desplegar"))),
    Scenario("pro_smalltalk", "work", "Professional small talk",
             "Small talk with a colleague at the coffee machine or before a meeting starts.", 1,
             ("present_continuous", "question_formation"),
             (("how's it going", "¿qué tal va?"), ("busy week", "semana ajetreada"), ("to grab a coffee", "tomar un café rápido"))),
    Scenario("email_to_speech", "work", "From email to conversation",
             "The learner explains out loud what they'd write in a work email (request, update or complaint), then you discuss it.", 2,
             ("polite_requests", "present_perfect"),
             (("to reach out", "contactar"), ("as soon as possible", "lo antes posible"), ("attached", "adjunto"))),
)

SCENARIOS_BY_KEY = {s.key: s for s in SCENARIOS}

# Estructuras gramaticales: cómo provocarlas en conversación (sin "clase de gramática").
GRAMMAR: dict[str, dict[str, str]] = {
    "past_simple": {"name": "Past simple", "elicit": "Ask about specific finished events: 'What did you do…?', 'How did it go?'"},
    "past_continuous": {"name": "Past continuous", "elicit": "Ask for background of a story: 'What were you doing when…?'"},
    "present_perfect": {"name": "Present perfect", "elicit": "Ask about life experience and recent results: 'Have you ever…?', 'What have you done so far…?'"},
    "present_simple": {"name": "Present simple (incl. 3rd person -s)", "elicit": "Ask about routines and general facts, including about other people: 'What does your manager do…?'"},
    "present_continuous": {"name": "Present continuous", "elicit": "Ask about current temporary situations: 'What are you working on these days?'"},
    "future_forms": {"name": "Future (will / going to / present continuous)", "elicit": "Ask about plans and predictions: 'What are you going to do…?', 'What do you think will happen…?'"},
    "conditionals": {"name": "Conditionals", "elicit": "Ask hypotheticals: 'What would you do if…?', 'If you had more time, …?'"},
    "comparatives": {"name": "Comparatives & superlatives", "elicit": "Ask them to compare two things: 'Which is better…, and why?'"},
    "modal_verbs": {"name": "Modal verbs (should, could, might, have to)", "elicit": "Ask for advice, obligations and possibilities."},
    "question_formation": {"name": "Question formation", "elicit": "Invite the learner to ask YOU questions: 'Ask me anything about…'."},
    "prepositions": {"name": "Prepositions", "elicit": "Ask about time, place and movement details: 'When / where exactly…?'"},
    "articles": {"name": "Articles (a / an / the / ø)", "elicit": "Ask for descriptions of things and places."},
    "polite_requests": {"name": "Polite requests", "elicit": "Create role-play moments where they must ask for something politely."},
    "giving_opinions": {"name": "Giving and justifying opinions", "elicit": "Ask 'What do you think about…? Why?' and gently push for a reason."},
    "frequency_adverbs": {"name": "Adverbs of frequency & word order", "elicit": "Ask 'How often do you…?'"},
    "used_to": {"name": "Used to (past habits)", "elicit": "Ask how things were different in the past: 'What did you use to do…?'"},
    "word_order": {"name": "Word order", "elicit": "Ask for longer descriptive answers."},
    "verb_patterns": {"name": "Verb patterns (to + inf / -ing)", "elicit": "Ask about likes, plans and decisions: 'What do you enjoy…?', 'What did you decide…?'"},
}

# Plan de largo plazo (por semanas). El planner lo usa como objetivo, pero lo adapta al progreso real.
WEEKLY_ARC: tuple[dict[str, str], ...] = (
    {"focus": "basic_conversation", "goal": "Feel comfortable talking: everyday topics, short and clear answers."},
    {"focus": "vocabulary", "goal": "Expand vocabulary: introduce and recycle more useful words and expressions."},
    {"focus": "longer_answers", "goal": "Longer responses: push for reasons, examples and details."},
    {"focus": "spontaneity", "goal": "Spontaneity: fewer scaffolds, more open questions, quicker responses."},
    {"focus": "professional", "goal": "Professional situations: meetings, updates, explaining problems."},
    {"focus": "complex_ideas", "goal": "Complex ideas: opinions, hypotheticals, negotiation, nuance."},
)

FILLER_WORDS = ("um", "umm", "uh", "uhh", "uhm", "er", "erm", "hmm", "you know", "i mean", "like", "kind of", "sort of")
