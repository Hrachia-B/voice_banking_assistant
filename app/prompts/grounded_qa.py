from __future__ import annotations

from app.guardrails.scope import ScopeDecision
from app.retrieval.models import RetrievalResult


def build_grounded_qa_instructions() -> str:
    return (
        "Դու հայկական բանկային աջակցման օգնականի grounded QA շերտն ես։ "
        "Քո միակ գիտելիքի աղբյուրը ստորև տրված retrieved chunk-երն են։ "
        "Արգելվում է օգտագործել արտաքին գիտելիք, նախնական հիշողություն, ընդհանուր բանկային փորձ, "
        "կամ տրամաբանական գուշակություն, եթե դա բացահայտ չի հաստատվում ապացույցներում։ "
        "Պատասխանիր միայն հայերենով, հակիրճ և փաստական ձևով։ "
        "Եթե ապացույցները բավարար չեն, իրար հակասում են, կամ հարցի կոնկրետ պատասխանը "
        "ուղիղ չի երևում ապացույցներից, պետք է հրաժարվես։ "
        "Մի համեմատի՛ր բանկերը, մի խորհուրդ տուր և մի գնահատիր, թե որն է ավելի լավը։ "
        "Արտածիր միայն մեկ JSON օբյեկտ հետևյալ դաշտերով՝ "
        '`decision`, `answer_armenian`, `used_chunk_ids`, `refusal_reason`։ '
        "`decision` պետք է լինի `answer` կամ `refuse`։ "
        "`answer_armenian`-ը միշտ պետք է լինի ոչ դատարկ հայերեն տող և երբեք `null` չլինի։ "
        "Եթե հրաժարվում ես, `answer_armenian`-ում գրիր կարճ հայերեն մերժում։ "
        "`used_chunk_ids`-ում նշիր միայն այն chunk_id-ները, որոնց վրա իրականում հիմնվել ես։ "
        "`refusal_reason`-ը թող լինի `null`, եթե պատասխանում ես, այլապես "
        "`insufficient_evidence` կամ `ambiguous_request`։ "
        "Եթե պատասխանում ես, նշիր միայն այն փաստերը, որոնք ուղղակիորեն հաստատվում են ապացույցներում։"
    )


def build_grounded_qa_input(
    query: str,
    scope_decision: ScopeDecision,
    retrieval_results: list[RetrievalResult],
) -> str:
    evidence_blocks = []
    for result in retrieval_results:
        evidence_blocks.append(
            "\n".join(
                [
                    f"[chunk_id={result.chunk_id}]",
                    f"bank_name={result.bank_name}",
                    f"topic={result.topic}",
                    f"source_url={result.source_url}",
                    f"page_title={result.page_title}",
                    f"score={result.score:.4f}",
                    "text:",
                    result.chunk_text,
                ]
            )
        )

    return (
        f"ՀԱՐՑ:\n{query}\n\n"
        f"ՍԿՈՊ:\n{scope_decision.scope}\n\n"
        f"ՀՆԱՐԱՎՈՐ ԲԱՆԿ:\n{scope_decision.bank_code or 'not_specified'}\n\n"
        "ՊԱՏԱՍԽԱՆԻ ԿԱՆՈՆՆԵՐ:\n"
        "- օգտվիր միայն ստորև տրված ապացույցներից\n"
        "- եթե բավարար ապացույց չկա, ընտրիր refuse\n"
        "- մի ավելացրու որևէ փաստ, որը բացահայտ նշված չէ ապացույցներում\n"
        "- պատասխանիր միայն հայերենով\n\n"
        "ԱՊԱՑՈՒՅՑՆԵՐ:\n"
        + "\n\n".join(evidence_blocks)
        + "\n\nՀիշիր՝ պատասխանիր միայն այս ապացույցների հիման վրա։"
    )
