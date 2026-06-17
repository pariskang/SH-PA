"""Validated ingestion and anonymization for real/hybrid official-document records.

仅接受经批准的"脱敏聚合"公文台账，拒绝含个人隐私字段的输入，并默认将机关名称匿名化，
以避免泄露真实机关身份或个人信息。
"""
from __future__ import annotations
import csv, hashlib
from pathlib import Path
from typing import Any

REQUIRED = {"agency_name", "doc_type", "title", "issue_date", "direction"}
# 禁止个人隐私 / 涉密真实标识字段进入基准语料
FORBIDDEN_HINTS = ("身份证", "id_card", "手机号", "mobile", "phone", "家庭住址", "银行卡", "patient", "姓名_明文")


def anonymize_agency(raw: str) -> str:
    number = int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16) % 999 + 1
    return f"GA{number:03d}"


def agency_code_for(raw: str) -> str:
    digest = int(hashlib.sha256(("code|" + raw).encode()).hexdigest()[:8], 16)
    return f"示机{digest % 900 + 100}"


def ingest_csv(path: Path, anonymize: bool = True) -> list[dict[str, Any]]:
    """Load approved aggregate official-document records; reject privacy-bearing inputs."""
    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("input records CSV is empty")
    missing = REQUIRED - set(rows[0])
    if missing:
        raise ValueError(f"input records missing fields: {sorted(missing)}")
    forbidden = {k for k in rows[0] if any(x in k.lower() for x in FORBIDDEN_HINTS)}
    if forbidden:
        raise ValueError(f"privacy-bearing fields are forbidden: {sorted(forbidden)}")
    output = []
    for idx, row in enumerate(rows, 1):
        raw_agency = row["agency_name"]
        output.append({
            "doc_id": f"R{idx:06d}",
            "agency_id": anonymize_agency(raw_agency) if anonymize else row.get("agency_id", raw_agency),
            "agency_name": f"示范机关{anonymize_agency(raw_agency)[2:]}" if anonymize else raw_agency,
            "agency_code": agency_code_for(raw_agency) if anonymize else row.get("agency_code", ""),
            "agency_level": row.get("agency_level", "市级"),
            "agency_category": row.get("agency_category", "政府部门"),
            "doc_type": row["doc_type"],
            "doc_number": row.get("doc_number", ""),
            "title": row["title"],
            "main_recipient": row.get("main_recipient", ""),
            "direction": row["direction"],
            "security_level": row.get("security_level", "公开"),
            "urgency": row.get("urgency", "平件"),
            "issue_date": row["issue_date"],
            "has_attachment": row.get("has_attachment", "0"),
            "cc_count": row.get("cc_count", "0"),
            "page_count": row.get("page_count", "1"),
            "format_flag": row.get("format_flag", "normal"),
            "source_type": "real",
            "scenario_id": row.get("scenario_id", "REAL_BASELINE"),
        })
    return output
