from signalforge import gamify as G


def test_levels_and_progress():
    assert G.level(0)[:2] == (1, "Intern")
    assert G.level(300)[1] == "Product Manager"
    assert G.level(10_000)[3] is None and G.progress(10_000) == 1.0
    assert 0 <= G.progress(200) < 1


def test_quest_awards_once_and_visits_complete_tour():
    p = G.Player()
    G.award(p, {"ask"}, 3, now=0)
    xp = p.xp
    G.award(p, {"ask"}, 3, now=1)
    assert p.xp == xp and p.done == ["ask"]
    for pg in "abc":
        G.award(p, {f"visit:{pg}"}, 3, now=1000)
    assert "tour" in p.done and p.visited == ["a", "b", "c"]


def test_combo_multiplies_and_caps():
    p = G.Player()
    G.award(p, {"propose"}, 7, now=0)
    G.award(p, {"approve"}, 7, now=30)
    G.award(p, {"ticket"}, 7, now=60)
    G.award(p, {"ship"}, 7, now=90)
    assert p.combo == G.MAX_COMBO
    assert p.xp == 80 + 100 * 2 + 100 * 3 + 120 * 3
    G.award(p, {"measure"}, 7, now=10_000)
    assert p.combo == 1


def test_badges_and_level_up_notifications():
    p = G.Player(started=0)
    msgs = G.award(p, {"propose", "approve", "ticket", "ship", "measure"}, 7, now=100)
    kinds = [k for k, _ in msgs]
    assert "level" in kinds
    assert {"Closed the loop", "Speedrunner"} <= set(p.badges)
    assert G.next_quest(p)[0] == "tour"
