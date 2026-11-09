"""Respuestas del profesor escritas a mano para contextos reales de las clases simuladas en los que el maestro (9B)
lo hizo mal (no dio el ejercicio pedido, tradujo preguntas, elogio falso…). Clave = posición del registro en
data/raw/r1.jsonl + r2.jsonl concatenados (collect.py sólo añade al final, así que no cambia)."""

GOLD = {
    15: "OK, so you go to work by car. Good, a clear sentence! Next one: how do you commute to work? Tell me in one "
        "full sentence. <elicit term=\"to commute\"/>",
    22: "No problem, let me say it more simply. I want to know what one person in your family does every day. For "
        "example: \"My mother works in a shop.\" <es>¿Qué hace cada día alguien de tu familia?</es> Now you: what "
        "does your mother or your father do every day?",
    25: "Good: \"my manager deals with the operations\", with -s, because it's he. Well done! And the next one is "
        "similar: what does a colleague deal with at work? <elicit term=\"to deal with\"/>",
    29: "In English, \"lejos\" is \"far away\". <vocab term=\"far away\" es=\"lejos\"/> So: \"My best friend lives "
        "in Cartagena, far away from me.\" And you said \"he lives\" with -s, great! Now, change this sentence to talk "
        "about your best friend: \"I live near the city centre.\" Start with \"My friend…\".",
    33: "Ha, relaxing in bed with Instagram reels sounds nice! Just one thing: we say \"on my day off\". "
        "<fix original=\"When my day off\" corrected=\"On my day off\"/> Now let's do the exercise: change \"I live "
        "near the city centre\" to talk about your friend. Start with \"My friend…\".",
    39: "OK, so your boss checks your work. Let's make the sentence negative together: \"My boss doesn't work on "
        "Saturdays.\" <es>Para negar con he o she usamos doesn't.</es> Next one: who is in charge of the shopping at "
        "home, and what does that person do? <elicit term=\"to be in charge of\"/>",
    43: "So the first thing you do at work is check your tasks and make a list. A good routine! Small tip: we say "
        "\"make a list\". <fix original=\"realize a list\" corrected=\"make a list\"/> Now it's your turn: ask me a "
        "question about my schedule. Start with \"Do you…\" or \"What time do you…\".",
    46: "Wow, you start at 3 a.m.? That's really early! And on Fridays you work only four hours. Good effort! Now "
        "it's your turn: ask me a question about my schedule. Start with \"Do you…\" or \"What time do you…\".",
    49: "No problem! I want you to ask ME a question about my day. For example: \"What time do you start work?\" "
        "<es>Ahora tú me haces una pregunta sobre mi horario.</es> Try it with \"Do you…\".",
    52: "Ah, so your father makes breakfast, and on Wednesdays he goes to work by car. Remember: he makes, he goes, "
        "with -s. <fix original=\"he make the breakfast\" corrected=\"he makes breakfast\"/> Great work! Now let's use "
        "it in a real conversation. What time does your father leave home on Wednesdays?",
    55: "Forty-three kilometres! That's a long commute for your father. <elicit term=\"to commute\"/> What time "
        "does he get up to get there on time?",
    62: "So on weekends your friend goes to his girlfriend's house, and at night you play games together. Sounds "
        "fun! Now tell me: what does your partner or a friend do for work?",
    65: "Oh no, so today you woke up late because you didn't set the alarm. <fix original=\"I don't put the alarm\" "
        "corrected=\"I didn't set the alarm\"/> We say you were running late. <vocab term=\"to run late\" "
        "es=\"ir con retraso\"/> Do you often run late on weekdays?",
    69: "Hi! I'm Sam, and I'm going to be your English teacher. It's really nice to meet you. Tell me, what's your "
        "name?",
    73: "Yes, that's right: \"my wife works at home\", with -s. Well done! Today we're going to learn the present "
        "continuous, so you can talk about what is happening now and about temporary situations. <es>Para algo que "
        "pasa ahora o estos días usamos be más el verbo con -ing: I'm working on a new project. Para rutinas seguimos "
        "con el presente simple.</es> For example: \"I'm working on a big project this month.\" And: \"My sister is "
        "living in London at the moment.\" Can you say the first example?",
    86: "Ha, looking for your keys! Good: \"I'm looking for\", with am and -ing. Next one, with a new expression: "
        "\"these days\" means <es>estos días</es>. <vocab term=\"these days\" es=\"estos días\"/> What project are "
        "you working on these days?",
    95: "Nice contrast: usually you cook dinner, but this week you're working on a project. That's exactly how we "
        "use it! Next one: what is someone in your family doing right now, do you think?",
    101: "Yes: \"usually she goes to work\", and \"her sister works\", both with -s. Don't worry about mistakes, "
         "that's how we learn! Next one: correct this sentence: \"I working from home today.\"",
    104: "Good: \"I'm helping my daughter with math\" and \"this week she is taking care of the kids\", both "
         "correct. And the corrected sentence is \"I'm working from home today.\" Next one: are you looking for "
         "anything new these days? A flat, a job, a hobby?",
    111: "You're cooking and speaking at the same time, that's real present continuous! Just one thing: \"I'm "
         "trying to learn English\". <fix original=\"i am try learning\" corrected=\"I'm trying to learn\"/> Great "
         "work! Now let's use it in a real conversation. What's going on at work these days?",
    115: "So your daughter's school started last week, and you're busy with accounting work. And your wife goes to "
         "the office every Monday, with -s. <fix original=\"my wife go\" corrected=\"my wife goes\"/> Are you "
         "working on anything special this month?",
    118: "That's a busy week, with work, cooking and the kids! So your wife is doing her work from Monday to Friday. "
         "What are you doing to rest these days?",
    139: "Good: \"I'm looking at my phone now\", with -ing. One more thing from today: you said \"my car key\", but "
         "you have more than one, so we say \"my car keys\". Imagine you can't find them at home. What are you "
         "doing?",
    146: "Hi! Today is a bit different: we're doing a role-play. You and I are colleagues waiting for a video call to "
         "start, and I'm Robin, from the US office. <es>Charlas unos minutos con un compañero de la oficina de EE. "
         "UU. mientras esperáis a los demás: el fin de semana, el tiempo, planes, el trabajo.</es> Your goals and "
         "some useful phrases are on the screen. Are you ready?",
    154: "Oh, so your kids are eleven and eight, and they're playing soccer in the park today. That's so nice! My kids are six and "
         "nine, and they love soccer too. Do your kids play every weekend?",
    169: "Ha, no worries, maybe it'll rain on Friday then! Here in Chicago it's really cold this week. In my house "
         "we're four: me, my wife and our two kids. Any plans for the holidays?",
    170: "Hi! Today is a bit different: we're doing a role-play. You and I are colleagues waiting for a video call to "
         "start, and I'm Robin, from the US office. <es>Charlas unos minutos con un compañero de la oficina de EE. "
         "UU. mientras esperáis a los demás: el fin de semana, el tiempo, planes, el trabajo.</es> Your goals and "
         "some useful phrases are on the screen. Are you ready?",
}
