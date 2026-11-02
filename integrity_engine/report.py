"""Report writers: per-document JSON + Markdown, aggregate summary matrix."""
import json
import os
from datetime import datetime, timezone


def write_document_reports(report: dict, md: str, output_dir: str) -> tuple:
    json_path = os.path.join(output_dir, f"{report['document_id']}.json")
    md_path = os.path.join(output_dir, f"{report['document_id']}.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md)
    return json_path, md_path


def render_document_markdown(report: dict) -> str:
    s = report["scores"]
    lines = [
        f"# Integrity Audit: {report['document_id']}",
        "",
        f"**Overall integrity risk: {report['total_integrity_risk']}**",
        "",
        "| Signal | Score |",
        "| --- | --- |",
        f"| Internal peer similarity | {s['peer_similarity_percentage']}% |",
        f"| External reference similarity | {s['external_similarity_percentage']}% |",
        f"| AI-generated likelihood | {s['ai_generated_probability']}% |",
        "",
        "## Explanation",
        "",
        report["explanations"],
        "",
    ]

    def section(title, rows, query_loc_key, matched_loc_key):
        if not rows:
            return
        lines.extend([f"## {title}", ""])
        for i, row in enumerate(rows, 1):
            lines.extend([
                f"### Match {i}: {row['matched_file']} (similarity {row['similarity_score']})",
                "",
                f"- Your document ({row[query_loc_key]}):",
                f"  > {row['query_passage']}",
                f"- Matching source ({row[matched_loc_key]}):",
                f"  > {row['matched_passage']}",
                "",
            ])

    section("Peer Matches (other submissions)", report["peer_matches"],
            "query_location", "matched_location")
    section("Cross-Modal Matches", report["multimodal_matches"],
            "query_location", "matched_location")
    section("External Reference Matches", report["external_matches"],
            "query_location", "matched_location")

    if report.get("boilerplate_excluded"):
        lines.extend([
            "## Excluded Template Content",
            "",
            "The following chunk(s) match template text shared across many "
            "submissions and were excluded from scoring:",
            "",
        ])
        for chunk in report["boilerplate_excluded"]:
            lines.append(f"- ({chunk['location']}) {chunk['excerpt']}")
        lines.append("")

    ai = report.get("ai_analysis", {})
    if ai:
        lines.extend([
            "## AI-Content Analysis",
            "",
            f"- Method: {ai.get('method', 'n/a')}",
            f"- Details: `{json.dumps(ai.get('details', {}))}`",
            "",
        ])

    if report.get("warnings"):
        lines.extend(["## Processing Warnings", ""])
        for w in report["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    lines.extend([
        "---",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by the "
        "Multimodal Submission Integrity Engine.",
        "",
    ])
    return "\n".join(lines)


def write_summary(run_summary: dict, matrix_rows: list, output_dir: str) -> str:
    lines = [
        "# Batch Integrity Summary",
        "",
        f"Documents processed: {run_summary['documents_processed']} "
        f"({run_summary['documents_ok']} ok, "
        f"{run_summary['documents_flagged']} flagged/unprocessable)",
        f"Chunks indexed: {run_summary['chunks_indexed']} "
        f"({run_summary['boilerplate_chunks']} excluded as common template)",
        "",
        "| Document | Risk | Peer % | External % | AI % | Status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in run_summary["documents"]:
        lines.append(
            f"| {row['document_id']} | {row['risk']} | {row['peer']} | "
            f"{row['external']} | {row['ai']} | {row['status']} |"
        )
    lines.extend(["", "## Pairwise Similarity Matrix (max chunk cosine)", ""])
    lines.append("| Document pair | Max chunk similarity |")
    lines.append("| --- | --- |")
    for r in sorted(matrix_rows, key=lambda x: -x["max_similarity"]):
        lines.append(f"| {r['doc_a']} x {r['doc_b']} | {r['max_similarity']:.3f} |")
    lines.append("")
    path = os.path.join(output_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"run": run_summary, "pairwise": matrix_rows}, fh, indent=2)
    return path
