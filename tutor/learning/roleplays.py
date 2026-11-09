"""Role-plays profesionales: práctica libre fuera de la ruta de unidades (situaciones reales de trabajo).

Una clase de role-play dura menos (10 min) y tiene otros pasos: preparación (el profesor explica la escena y el
objetivo, con apoyo en español) → role-play (el profesor hace de su personaje) → feedback (sale del personaje) → cierre.
Los objetivos se comprueban con reglas sobre lo que dijo el alumno; las frases útiles se ven en pantalla.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Goal:
    en: str
    es: str
    check: str  # regex sobre lo que dijo el alumno durante el role-play (sin distinguir mayúsculas)


@dataclass(frozen=True)
class RolePlay:
    key: str
    title: str
    title_es: str
    tutor_role: str  # el personaje del profesor (en inglés, para el prompt)
    learner_role_es: str  # el papel del alumno, en español (pantalla)
    scene: str  # la situación, para el prompt
    scene_es: str  # la situación, en español (pantalla y explicación inicial)
    goals: tuple[Goal, ...]
    phrases: tuple[tuple[str, str], ...]  # frases útiles (inglés, español)
    beats: tuple[str, ...]  # lo que pregunta el personaje, en orden (se adapta a lo que diga el alumno)
    level: str = "B1"
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["goals"] = [asdict(g) for g in self.goals]
        d["phrases"] = [list(p) for p in self.phrases]
        return d


ROLEPLAYS: tuple[RolePlay, ...] = (
    RolePlay(
        "job_interview", "Job interview: software developer", "Entrevista de trabajo (desarrollador)",
        "Alex, a friendly engineering manager at a US software company, interviewing the learner for a remote "
        "Python developer position",
        "Eres el candidato a un puesto remoto de desarrollador Python.",
        "A remote job interview for a Python developer role. The interviewer asks about experience, a project, "
        "teamwork and why this job.",
        "Una entrevista remota para un puesto de desarrollador Python en una empresa de EE. UU.: te preguntan por tu "
        "experiencia, un proyecto, el trabajo en equipo y por qué quieres el puesto.",
        (Goal("Introduce yourself and your experience", "Presentarte y contar tu experiencia",
              r"\b(i'?ve been|i have been|i have \w+ years|\d+|years? of experience|i work(ed)? (as|at|for|with))\b"),
         Goal("Describe a project you worked on", "Describir un proyecto en el que trabajaste",
              r"\b(project|built|developed|created|implemented|designed)\b"),
         Goal("Explain how you work in a team", "Explicar cómo trabajas en equipo",
              r"\b(team|teammates?|colleagues?|together|collaborat\w*)\b"),
         Goal("Ask the interviewer a question", "Hacerle una pregunta al entrevistador",
              r"\b(what|how|when|which|could you|can you|do you|is there|are there)\b[^.!]*\?|"
              r"^(what|how|could|can|do|is|are)\b")),
        (("I've been working as a … for … years", "Llevo … años trabajando como …"),
         ("I was responsible for …", "Era responsable de …"),
         ("One of the projects I'm most proud of is …", "Uno de los proyectos de los que estoy más orgulloso es …"),
         ("I'm used to working with …", "Estoy acostumbrado a trabajar con …"),
         ("Could you tell me more about the team?", "¿Podría contarme más sobre el equipo?")),
        ("Thanks for joining. To start, could you tell me a bit about yourself and your experience?",
         "Tell me about a project you're proud of. What was your role?",
         "What was the hardest problem in that project, and how did you solve it?",
         "How do you usually work with your team? For example, when you disagree with someone?",
         "Why are you interested in this position?",
         "Do you have any questions for me?"),
        tags=("trabajo", "entrevista")),
    RolePlay(
        "daily_standup", "Daily stand-up meeting", "Daily stand-up con tu equipo",
        "Maya, the team's scrum master, running a quick daily stand-up on a video call",
        "Eres un desarrollador del equipo y te toca dar tu actualización.",
        "A daily stand-up: the learner says what they did yesterday, what they'll do today and any blockers. Maya "
        "asks short follow-up questions.",
        "La reunión diaria del equipo: cuentas qué hiciste ayer, qué harás hoy y si tienes algún bloqueo.",
        (Goal("Say what you did yesterday", "Decir qué hiciste ayer",
              r"\byesterday\b|\b(i|we) (finished|worked|fixed|started|did|had|was|were|made|wrote|reviewed)\b"),
         Goal("Say what you're going to do today", "Decir qué vas a hacer hoy",
              r"\btoday\b|\b(going to|i'?ll|i will|i'm planning|plan to|have to|need to)\b"),
         Goal("Mention a blocker or say you have none", "Mencionar un bloqueo (o decir que no tienes)",
              r"\b(block\w*|stuck|problem|issue|waiting for|no blockers?|nothing blocking)\b")),
        (("Yesterday I worked on …", "Ayer trabajé en …"),
         ("Today I'm going to …", "Hoy voy a …"),
         ("I'm blocked by …", "Estoy bloqueado por …"),
         ("I'm waiting for … to …", "Estoy esperando a que … "),
         ("No blockers for now.", "Por ahora, sin bloqueos.")),
        ("Morning! Let's start with you. What did you work on yesterday?",
         "Nice. And what's your plan for today?",
         "Anything blocking you?",
         "How long do you think it will take?",
         "Do you need help from anyone on the team?"),
        tags=("trabajo", "reuniones")),
    RolePlay(
        "explain_bug", "Explaining a production bug", "Explicarle un bug a tu jefe",
        "Jordan, the learner's manager, a bit worried because customers are reporting errors in production",
        "Eres el desarrollador que encontró la causa del error.",
        "Customers report errors in production. The manager asks what happened, the impact, the fix and how to "
        "prevent it.",
        "Hay errores en producción y tu jefe quiere saber qué pasó, a quién afecta, cómo lo arreglas y cómo evitarlo.",
        (Goal("Explain what caused the bug", "Explicar qué causó el error",
              r"\b(because|caused|the problem (is|was)|happened|the reason)\b"),
         Goal("Explain the impact", "Explicar a quién afecta",
              r"\b(users?|customers?|clients?|affect\w*|impact)\b"),
         Goal("Propose a fix and a time", "Proponer una solución y un plazo",
              r"\b(fix|deploy|patch|solve|hotfix|by (tomorrow|today|tonight|monday|friday)|in \w+ (hours?|minutes?|days?))\b"),
         Goal("Say how to prevent it next time", "Decir cómo evitarlo la próxima vez",
              r"\b(tests?|monitor\w*|alert\w*|prevent|next time|review|check)\b")),
        (("The problem was caused by …", "El problema lo causó …"),
         ("It only affects users who …", "Solo afecta a los usuarios que …"),
         ("I've already pushed a fix to …", "Ya subí una corrección a …"),
         ("It should be fixed by …", "Debería estar arreglado para …"),
         ("To prevent this, we should …", "Para evitarlo, deberíamos …")),
        ("Hi, I've got a few customers saying the app is failing. Do you know what's going on?",
         "OK. How many users are affected?",
         "What's the plan to fix it, and when can we have it in production?",
         "How can we make sure this doesn't happen again?",
         "Should I tell the customers anything?"),
        tags=("trabajo", "problemas")),
    RolePlay(
        "client_requirements", "Meeting with a client", "Reunión con un cliente: requisitos",
        "Sam Taylor, a non-technical client who wants a new feature in their online store and isn't very clear "
        "about the details",
        "Eres el desarrollador que se reúne con el cliente para entender lo que necesita.",
        "A client wants a new feature for their online store. The learner must ask questions to clarify the "
        "requirements, the deadline and the budget, and summarise the agreement.",
        "Un cliente quiere una funcionalidad nueva para su tienda online. Tienes que hacer preguntas para aclarar qué "
        "necesita, el plazo y el presupuesto, y resumir lo acordado.",
        (Goal("Ask clarifying questions", "Hacer preguntas para aclarar",
              r"\b(what|how|who|which|when|do you|does it|would you|could you)\b[^.!]*\?|"
              r"^(what|how|who|which|when|do|does|would|could)\b"),
         Goal("Ask about the deadline", "Preguntar por el plazo",
              r"\b(deadline|when do you need|by when|how soon|date|launch)\b"),
         Goal("Summarise what you agreed", "Resumir lo acordado",
              r"\b(so,? (to sum up|basically|you need|you want)|let me (summari[sz]e|confirm|check)|"
              r"just to confirm|if i understand)\b")),
        (("Just to make sure I understand, …", "Para asegurarme de que entiendo, …"),
         ("Could you give me an example of …?", "¿Me podría dar un ejemplo de …?"),
         ("When do you need it by?", "¿Para cuándo lo necesita?"),
         ("Let me summarise what we agreed.", "Déjeme resumir lo que acordamos."),
         ("I'll send you an estimate by …", "Le enviaré un presupuesto para …")),
        ("Hi! Thanks for meeting me. So, I want customers to be able to save their favourite products. Can you "
         "do that?",
         "Hmm, I'm not sure. What do you need to know?",
         "Ideally it should be ready before Black Friday. Is that possible?",
         "And how much would it cost, more or less?",
         "Great. So what are the next steps?"),
        tags=("trabajo", "clientes")),
    RolePlay(
        "code_review", "Code review with a teammate", "Code review con un compañero",
        "Chris, a teammate who wrote a pull request and is a little defensive about their code",
        "Revisas el pull request de un compañero y le das feedback con educación.",
        "The learner reviews a teammate's pull request: some good things, one bug, and a naming problem. They must "
        "give feedback politely and agree on changes.",
        "Revisas el pull request de un compañero: tiene cosas buenas, un bug y nombres poco claros. Debes dar feedback "
        "con educación y acordar los cambios.",
        (Goal("Say something positive first", "Empezar con algo positivo",
              r"\b(good|great|nice|like|clean|well done|clear)\b"),
         Goal("Point out a problem politely", "Señalar un problema con educación",
              r"\b(maybe|might|could|i think|i noticed|what about|would it be|i'm not sure)\b"),
         Goal("Agree on the next steps", "Acordar los siguientes pasos",
              r"\b(could you|can you|let'?s|i'?ll|we can|next|after that|once you)\b")),
        (("Overall it looks good, but …", "En general se ve bien, pero …"),
         ("I noticed that …", "Me di cuenta de que …"),
         ("What do you think about …?", "¿Qué te parece …?"),
         ("Maybe we could rename … to …", "Quizá podríamos renombrar … a …"),
         ("Once you change that, I'll approve it.", "Cuando cambies eso, lo apruebo.")),
        ("Hey, did you have a chance to look at my pull request?",
         "Thanks. Was there anything you didn't like?",
         "Hmm, I'm not sure that's a bug. Why do you think so?",
         "OK, what would you change exactly?",
         "Fair enough. Anything else before you approve it?"),
        tags=("trabajo", "feedback")),
    RolePlay(
        "deadline_negotiation", "Negotiating a deadline", "Negociar un plazo de entrega",
        "Pat, a product manager who wants a feature finished by Friday, which is not realistic",
        "Eres el desarrollador y necesitas más tiempo para hacerlo bien.",
        "The product manager wants a feature by Friday. The learner explains why it needs more time, proposes "
        "options and reaches an agreement.",
        "El product manager quiere una funcionalidad para el viernes, pero no es realista. Explicas por qué necesitas "
        "más tiempo, propones opciones y llegáis a un acuerdo.",
        (Goal("Explain why you need more time", "Explicar por qué necesitas más tiempo",
              r"\b(because|since|the problem is|it takes|we need to|there are)\b"),
         Goal("Propose an alternative", "Proponer una alternativa",
              r"\b(what if|how about|we could|i suggest|instead|option|first version|by (monday|tuesday|wednesday|next week))\b"),
         Goal("Reach an agreement", "Llegar a un acuerdo",
              r"\b(deal|agree|sounds good|that works|ok(ay)?,? (so|then)|let'?s do)\b")),
        (("I'm afraid Friday is too tight because …", "Me temo que el viernes es muy justo porque …"),
         ("What if we deliver a first version by …?", "¿Y si entregamos una primera versión para …?"),
         ("We could leave … for the next sprint.", "Podríamos dejar … para el siguiente sprint."),
         ("If we skip the tests, we risk …", "Si nos saltamos las pruebas, nos arriesgamos a …"),
         ("That works for me.", "Eso me funciona.")),
        ("Hi! Quick question: can we have the new payment feature ready by Friday?",
         "Why not? It doesn't look that big to me.",
         "Hmm. The client really wants it soon. What can you do?",
         "And if you work a bit faster, maybe without so many tests?",
         "OK. So what's the final plan?"),
        tags=("trabajo", "negociación")),
    RolePlay(
        "meeting_small_talk", "Small talk before a video call", "Small talk antes de una videollamada",
        "Robin, a colleague from the US office, chatting while waiting for others to join a call",
        "Esperas a que empiece una reunión y charlas con un compañero de otro país.",
        "Two colleagues chat for a few minutes before a meeting starts: the weekend, the weather, plans, work.",
        "Charlas unos minutos con un compañero de la oficina de EE. UU. mientras esperáis a los demás: el fin de "
        "semana, el tiempo, planes, el trabajo.",
        (Goal("Answer and add a detail", "Responder y añadir un detalle",
              r"\b(\w+ed|went|had|was|were|saw|took|made)\b.{15,}"),
         Goal("Ask your colleague a question back", "Devolverle una pregunta",
              r"\b(what about you|and you|how about you|how was your|did you|do you|are you)\b"),
         Goal("Use an expression to keep the chat going", "Usar una expresión para mantener la charla",
              r"\b(oh really|that sounds|no way|lucky you|i know what you mean|that's (great|nice|cool|funny))\b")),
        (("How was your weekend?", "¿Qué tal tu fin de semana?"),
         ("Not bad, I just …", "Nada mal, solo …"),
         ("That sounds great!", "¡Suena genial!"),
         ("What about you?", "¿Y tú?"),
         ("Lucky you!", "¡Qué suerte!")),
        ("Hey! Looks like we're the first ones here. How's it going?",
         "Did you do anything fun last weekend?",
         "Nice! What's the weather like over there right now?",
         "Any plans for the holidays?",
         "Oh, here come the others. Shall we start?"),
        tags=("trabajo", "conversación")),
)

ROLEPLAYS_BY_KEY = {r.key: r for r in ROLEPLAYS}


def goals_reached(rp: RolePlay, learner_texts: list[str]) -> list[bool]:
    """Qué objetivos cumplió el alumno (en algún turno del role-play)."""
    return [any(re.search(g.check, t.strip(), re.I) for t in learner_texts) for g in rp.goals]
