"""Tests deterministas del detector de final de turno (sin modelos: probabilidades de VAD sintéticas)."""

from tutor.audio.turn_detector import (
    FRAME_S, Ev, PatienceProfile, TurnDetector, combine, expects_long_answer, text_completeness,
)


def run(detector, script, analyses, tutor_text="What did you do yesterday?"):
    """script: [(prob, segundos)]. analyses: lista de (smart_turn_p, texto) entregada en cada PAUSE.
    Devuelve [(evento, tiempo)]."""
    detector.start_listening(0.0, tutor_text)
    t, out, k = 0.0, [], 0
    for prob, dur in script:
        for _ in range(int(round(dur / FRAME_S))):
            t += FRAME_S
            for ev in detector.process(prob, t):
                out.append((ev, round(t, 2)))
                if ev == Ev.PAUSE and k < len(analyses):
                    detector.set_analysis(*analyses[k])
                    k += 1
            if detector._ended:
                return out
    return out


def events(out):
    return [e for e, _ in out]


def test_text_completeness_signals():
    assert text_completeness("Yesterday I went to the office because") < 0.2
    assert text_completeness("I needed to, umm") <= 0.05
    assert text_completeness("let me think") <= 0.05
    assert text_completeness("We had a deadline for a new project.") >= 0.75
    assert text_completeness("Yes.") >= 0.8
    assert text_completeness("Yes.", expects_long_answer=True) < 0.7
    assert text_completeness("Yes, I have.") >= 0.7
    assert text_completeness("I think the problem is") < 0.2
    # Palabras que contienen una muletilla no lo son ("togeth-er", "sum-mer")
    assert text_completeness("We had lunch together.") >= 0.75
    assert text_completeness("It went well.") >= 0.75


def test_expects_long_answer():
    assert expects_long_answer("Nice. Tell me about your weekend?")
    assert expects_long_answer("Why did you choose that job?")
    assert not expects_long_answer("Do you like coffee?")


def test_combine_text_veto():
    # Entonación de final, pero el texto acaba en "because": manda el texto.
    assert combine(0.9, 0.15) <= 0.3
    assert combine(0.9, 0.8) >= 0.9


def test_thinking_pauses_do_not_end_turn():
    """'Yesterday I went to... umm... I went to the office because... I needed to finish a report.'"""
    d = TurnDetector(PatienceProfile())
    script = [
        (0.0, 1.0),  # piensa antes de empezar
        (0.9, 1.2), (0.05, 1.8),  # "Yesterday I went to..." + pausa de 1.8 s
        (0.9, 0.6), (0.05, 2.5),  # "umm" + pausa de 2.5 s
        (0.9, 1.5), (0.05, 2.0),  # "I went to the office because" + pausa 2.0 s
        (0.9, 2.0), (0.05, 6.0),  # "I needed to finish a report." + silencio final
    ]
    analyses = [
        (0.2, "Yesterday I went to..."),
        (0.1, "Yesterday I went to... umm"),
        (0.3, "Yesterday I went to... umm I went to the office because"),
        (0.92, "Yesterday I went to... umm I went to the office because I needed to finish a report."),
    ]
    out = run(d, script, analyses)
    ends = [(e, t) for e, t in out if e in (Ev.END_OF_TURN, Ev.STUCK)]
    assert len(ends) == 1 and ends[0][0] == Ev.END_OF_TURN
    # El final llega DESPUÉS de la última frase (que acaba en t = 12.6 s), nunca en las pausas intermedias
    assert ends[0][1] > 12.6
    # Y no se hace esperar demasiado cuando la frase está claramente terminada
    assert ends[0][1] - 12.6 < 2.0
    assert len(d.stats.pauses) == 3


def test_complete_short_answer_is_fast():
    d = TurnDetector(PatienceProfile())
    out = run(d, [(0.0, 0.5), (0.9, 1.5), (0.05, 5.0)], [(0.9, "We had a deadline for a new project.")],
              tutor_text="What made it so busy?")
    end_t = next(t for e, t in out if e == Ev.END_OF_TURN)
    assert end_t - 2.0 < 1.6  # < 1.6 s de silencio
    assert Ev.SPECULATE in events(out)  # se empezó a generar la respuesta antes de confirmar el final


def test_filler_at_end_waits_long():
    d = TurnDetector(PatienceProfile())
    out = run(d, [(0.9, 1.0), (0.05, 10.0)], [(0.7, "I think, umm")])
    end = [(e, t) for e, t in out if e in (Ev.END_OF_TURN, Ev.STUCK)]
    assert end and end[0][0] == Ev.STUCK  # nunca "contesta" a una muletilla: ofrece ayuda
    assert end[0][1] - 1.0 >= 5.0


def test_stuck_mid_sentence_triggers_help_not_reply():
    d = TurnDetector(PatienceProfile(pause_p90=2.0, multiplier=1.0))
    out = run(d, [(0.9, 1.0), (0.05, 15.0)], [(0.05, "I went to the")])
    kinds = events(out)
    # Una frase abierta sólo termina como STUCK (el tutor ayuda), tras una espera larga.
    final = [(e, t) for e, t in out if e in (Ev.END_OF_TURN, Ev.STUCK)][0]
    assert final[0] == Ev.STUCK
    assert final[1] - 1.0 >= 4.5
    assert Ev.SPECULATE not in kinds


def test_no_endpoint_until_transcript_ready():
    d = TurnDetector(PatienceProfile())
    d.start_listening(0.0, "Hi")
    t = 0.0
    for _ in range(40):  # 1.28 s de voz
        t += FRAME_S
        d.process(0.9, t)
    evs = []
    for _ in range(200):  # 6.4 s de silencio sin análisis
        t += FRAME_S
        evs += d.process(0.05, t)
    assert Ev.END_OF_TURN not in evs


def test_resume_during_grace_reopens_turn():
    d = TurnDetector(PatienceProfile())
    out = run(d, [(0.9, 1.0), (0.05, 3.0)], [(0.95, "That's right.")])
    assert Ev.END_OF_TURN in events(out)
    t = 5.0
    evs = []
    for _ in range(5):
        t += FRAME_S
        evs += d.process(0.9, t)
    assert Ev.RESUMED in evs
    assert d.in_turn and d.false_endpoint


def test_barge_in_requires_sustained_voice():
    d = TurnDetector(PatienceProfile())
    t, evs = 0.0, []
    for p in [0.9, 0.9, 0.2, 0.9, 0.9, 0.9]:  # golpes cortos / eco: no
        t += FRAME_S
        evs += d.process(p, t, tutor_speaking=True)
    assert Ev.BARGE_IN not in evs
    for _ in range(10):
        t += FRAME_S
        evs += d.process(0.95, t, tutor_speaking=True)
    assert Ev.BARGE_IN in evs


def test_profile_learns_slow_speaker():
    p = PatienceProfile()
    before = (p.pause_p50, p.pause_p90)
    for _ in range(60):
        for d in (1.5, 2.0, 2.5, 3.0, 3.5):
            p.observe_pause(d)
    assert p.pause_p50 > before[0] + 0.8
    assert p.pause_p90 > before[1] + 0.6
    # → más paciencia para frases abiertas
    slow = TurnDetector(p)
    fast = TurnDetector(PatienceProfile())
    for det in (slow, fast):
        det.set_analysis(0.3, "and then I")
    assert slow.required_silence() > fast.required_silence() + 1.0


def test_false_endpoints_increase_patience():
    p = PatienceProfile()
    m0 = p.multiplier
    p.observe_endpoint(True, 1.0, 0.8)
    p.observe_endpoint(True, 1.0, 0.8)
    assert p.multiplier > m0 * 1.15
    m1 = p.multiplier
    for _ in range(10):
        p.observe_endpoint(False, 2.5, 0.9)  # esperó de más con frases claras
    assert p.multiplier < m1


def test_no_response_nudge():
    d = TurnDetector(PatienceProfile())
    d.start_listening(0.0, "Have you ever been to London?")
    t, evs = 0.0, []
    for _ in range(int(12 / FRAME_S)):
        t += FRAME_S
        evs += d.process(0.0, t)
    assert evs.count(Ev.NO_RESPONSE) == 1


def test_stale_analysis_does_not_close_turn():
    """El análisis de un segmento antiguo llega cuando ya empezó otra pausa: no debe cerrar el turno."""
    d = TurnDetector(PatienceProfile())
    d.start_listening(0.0, "How are you?")
    t = 0.0
    seq = 0
    for prob, dur in [(0.9, 1.0), (0.05, 0.4), (0.9, 1.5), (0.05, 4.0)]:
        for _ in range(int(dur / FRAME_S)):
            t += FRAME_S
            evs = d.process(prob, t)
            if Ev.PAUSE in evs:
                seq += 1
            if seq == 2 and Ev.PAUSE in evs:
                d.set_analysis(0.95, "I'm fine, thanks.", latest=False)  # resultado del 1er segmento, tarde
            assert Ev.END_OF_TURN not in evs
    d.set_analysis(0.95, "I'm fine, thanks. I'm a bit tired today.", latest=True)
    assert Ev.END_OF_TURN in d.process(0.05, t + FRAME_S)
