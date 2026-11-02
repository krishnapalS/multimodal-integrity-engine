"""Risk banding and human-readable explanation generation."""
from integrity_engine.config import Config

_CFG = Config()


def risk_band(peer_pct: float, external_pct: float, ai_pct: float) -> str:
    if (peer_pct >= _CFG.risk_high_peer
            or external_pct >= _CFG.risk_high_external
            or ai_pct >= _CFG.risk_high_ai):
        return "HIGH"
    if (peer_pct >= _CFG.risk_medium_peer
            or external_pct >= _CFG.risk_medium_external
            or ai_pct >= _CFG.risk_medium_ai):
        return "MEDIUM"
    return "LOW"


def explain(doc_id: str, peer_pct: float, external_pct: float,
            ai_pct: float, peer_rows: list, external_rows: list,
            multimodal_rows: list, boilerplate_chunks: int,
            ai_method: str) -> str:
    parts = []
    evidence = peer_rows or multimodal_rows
    if peer_pct >= _CFG.risk_high_peer:
        if evidence:
            top = evidence[0]
            parts.append(
                f"High internal peer similarity ({peer_pct}% of substantive "
                f"chunks): strongest overlap with {top['matched_file']} "
                f"(cosine {top['similarity_score']}, {top['query_location']})."
            )
        else:
            parts.append(
                f"High internal peer similarity ({peer_pct}% of substantive chunks)."
            )
    elif peer_pct >= _CFG.risk_medium_peer:
        parts.append(
            f"Moderate internal peer similarity ({peer_pct}% of substantive chunks); "
            "review the matched passages below."
        )
    else:
        parts.append(f"Low internal peer similarity ({peer_pct}%).")

    if external_pct >= _CFG.risk_medium_external and external_rows:
        top = external_rows[0]
        parts.append(
            f"External reference overlap ({external_pct}%): strongest match "
            f"against {top['matched_file']} at {top['query_location']}."
        )

    if multimodal_rows:
        top = multimodal_rows[0]
        parts.append(
            f"Cross-modal duplication detected: spoken content in {doc_id} matches "
            f"{top['matched_file']} ({top['query_location']} vs "
            f"{top['matched_location']}, cosine {top['similarity_score']})."
        )

    if ai_pct >= _CFG.risk_medium_ai:
        parts.append(
            f"AI-generated content likelihood is {ai_pct}% ({ai_method})."
        )

    if boilerplate_chunks:
        parts.append(
            f"{boilerplate_chunks} chunk(s) matched institutional template text "
            "shared by many submissions and were excluded from scoring."
        )

    parts.append(
        "Verdict basis: coverage of matched non-boilerplate chunks; "
        "every claim above is cross-verifiable from the passages below."
    )
    return " ".join(parts)
