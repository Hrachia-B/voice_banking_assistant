from app.guardrails.scope import QueryScopeGuard


def test_scope_guard_classifies_supported_topics() -> None:
    guard = QueryScopeGuard()

    credits = guard.classify("Յունիբանկի սպառողական վարկի պայմանները")
    deposits = guard.classify("ԱԵԲ բանկի ավանդի տոկոսադրույքը ինչքա՞ն է")
    branches = guard.classify("ՎՏԲ-ի մասնաճյուղը որտեղ է Երևանում")

    assert credits.scope == "credits"
    assert credits.bank_code == "unibank"
    assert deposits.scope == "deposits"
    assert deposits.bank_code == "aeb"
    assert branches.scope == "branches"
    assert branches.bank_code == "vtb"


def test_scope_guard_refuses_out_of_scope_cards() -> None:
    guard = QueryScopeGuard()

    decision = guard.classify("Քարտի սպասարկման վճարը ինչքա՞ն է")

    assert decision.scope == "out_of_scope"


def test_scope_guard_refuses_comparative_requests() -> None:
    guard = QueryScopeGuard()

    decision = guard.classify("Ո՞ր բանկի վարկն է ավելի լավ")

    assert decision.scope == "out_of_scope"
    assert decision.reason == "comparative_request"


def test_scope_guard_handles_noisy_voice_like_query() -> None:
    guard = QueryScopeGuard()

    decision = guard.classify("ՎՏԲի մասնաճույգը որտեղ ա Երևանում")

    assert decision.scope == "branches"
    assert decision.bank_code == "vtb"
